import os
import json
import cv2
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.special import softmax
from scipy.optimize import minimize
from tqdm import tqdm

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
CONFIG_FILE = 'satellite/rtmpose-m_satellite_f.py'
CHECKPOINT_FILE = '/workspace/rtmpose-m_f/epoch_420.pth'

POSE_GT_FILE = '/workspace/speedplusv2/sunlamp/test.json'
IMAGE_META_FILE = '/workspace/speedplusv2/annotations/test_sunlamp.json'
IMG_ROOT = '/workspace/speedplusv2/sunlamp_preprocessed/'
MODEL_3D_POINTS_FILE = '/workspace/speedplusv2/tangoPoints.mat'
CAMERA_FILE = '/workspace/speedplusv2/camera.json'

# 모델 입력 사이즈 (Config에 맞춰 설정 필요, 보통 256 or 224)
MODEL_INPUT_SIZE = (224, 224) 

# ------------------------------------------------------------------------------
# 1. SimCC Hook & Refiner Class
# ------------------------------------------------------------------------------

# 전역 변수로 Logit 캡처
captured_logits = {}

def hook_fn_x(module, input, output):
    captured_logits['x'] = output.detach().cpu().numpy() # Softmax는 나중에 함

def hook_fn_y(module, input, output):
    captured_logits['y'] = output.detach().cpu().numpy()

class SimCCPoseRefiner:
    def __init__(self, K, dist, model_input_size):
        self.K = K
        self.dist = dist
        self.w_model, self.h_model = model_input_size
        self.splines_x = []
        self.splines_y = []
        self.simcc_scale_x = 1.0
        self.simcc_scale_y = 1.0

    def prepare_pdf(self, logits_x, logits_y, sigma=2.0):
        """
        Raw Logits -> Softmax -> Smoothing -> Log -> CubicSpline
        최적화를 위해 연속적인 Log-Probability Function을 만듭니다.
        """
        self.splines_x = []
        self.splines_y = []
        
        # SimCC Bin 개수 확인 및 스케일 계산
        n_kpts, n_bins_x = logits_x.shape
        _, n_bins_y = logits_y.shape
        
        # SimCC 좌표계와 모델 입력 좌표계 사이의 비율 (보통 2배)
        self.simcc_scale_x = n_bins_x / self.w_model
        self.simcc_scale_y = n_bins_y / self.h_model
        
        epsilon = 1e-6  # log(0) 방지

        for i in range(n_kpts):
            # 1. Softmax & Smoothing
            prob_x = gaussian_filter1d(softmax(logits_x[i]), sigma=sigma)
            prob_y = gaussian_filter1d(softmax(logits_y[i]), sigma=sigma)
            
            # 2. Log-Likelihood (Negative Log를 최소화할 것이므로 여기선 그냥 Log)
            # Clip을 통해 수치적 안정성 확보
            log_prob_x = np.log(np.clip(prob_x, epsilon, 1.0))
            log_prob_y = np.log(np.clip(prob_y, epsilon, 1.0))
            
            # 3. Cubic Spline Interpolation (미분 가능하게 만듦)
            # x_indices는 SimCC Bin Index
            x_axis = np.arange(n_bins_x)
            y_axis = np.arange(n_bins_y)
            
            self.splines_x.append(CubicSpline(x_axis, log_prob_x))
            self.splines_y.append(CubicSpline(y_axis, log_prob_y))

    def objective_function(self, params, points_3d, crop_meta):
        """
        Minimize Negative Log-Likelihood with Safety Bounds
        """
        rvec, tvec = params[:3], params[3:]
        
        # 1. Project
        img_points, _ = cv2.projectPoints(points_3d, rvec, tvec, self.K, self.dist)
        img_points = img_points.squeeze()
        
        # 2. Transform (Original -> Crop -> SimCC)
        x1, y1 = crop_meta['crop_box'][:2]
        scale = crop_meta['scale_factor']
        
        # Scale이 array인지 scalar인지 확인하여 안전하게 연산
        if isinstance(scale, (list, np.ndarray)):
            scale = np.array(scale) # shape (2,)
        
        model_pts = (img_points - np.array([x1, y1])) * scale
        
        simcc_pts_x = model_pts[:, 0] * self.simcc_scale_x
        simcc_pts_y = model_pts[:, 1] * self.simcc_scale_y
        
        nll_sum = 0.0
        n_bins_x = len(self.splines_x[0].x) # 448
        n_bins_y = len(self.splines_y[0].x)
        
        penalty_value = 100.0 # 범위 밖으로 나갔을 때 줄 페널티 (Log space이므로 100은 매우 큰 값)

        for i in range(len(self.splines_x)):
            ux, uy = simcc_pts_x[i], simcc_pts_y[i]
            
            # --- Safety Check (Bounds) ---
            # 이미지를 벗어나면 Spline이 이상한 값을 뱉으므로 강제로 페널티 부여
            if not (0 <= ux < n_bins_x):
                # 벗어난 거리만큼 페널티를 주어 안으로 들어오게 유도 (L1 loss style)
                dist_penalty = abs(ux - n_bins_x/2) * 0.1 
                nll_sum += (penalty_value + dist_penalty)
            else:
                nll_sum -= self.splines_x[i](ux) # Maximize Log-Prob

            if not (0 <= uy < n_bins_y):
                dist_penalty = abs(uy - n_bins_y/2) * 0.1
                nll_sum += (penalty_value + dist_penalty)
            else:
                nll_sum -= self.splines_y[i](uy)

        return nll_sum

    def refine(self, rvec_init, tvec_init, points_3d, crop_meta):
        initial_params = np.concatenate([rvec_init.flatten(), tvec_init.flatten()])
        
        # Method 변경 추천: L-BFGS-B는 Gradient가 불안정하면 튈 수 있음.
        # 'Nelder-Mead'는 느리지만 미분값을 안 써서 훨씬 안정적임. 
        # (초기값이 이미 좋다면 수십 번의 iteration으로 충분)
        res = minimize(
            self.objective_function,
            initial_params,
            args=(points_3d, crop_meta),
            method='Nelder-Mead', 
            options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
        )
        
        # 만약 L-BFGS-B를 쓰고 싶다면 bounds 옵션을 줘서 tvec이 너무 튀지 않게 막아야 함
        
        return res.x[:3], res.x[3:]

# ------------------------------------------------------------------------------
# Helper Functions (Data Loading)
# ------------------------------------------------------------------------------

def load_camera_intrinsics(camera_json_path):
    with open(camera_json_path) as f:
        cam = json.load(f)
    K = np.array(cam['cameraMatrix'], dtype=np.float32)
    dist = np.array(cam['distCoeffs'], dtype=np.float32).flatten()
    return K, dist

def load_tango_3d_keypoints(mat_path):
    vertices = loadmat(mat_path)['tango3Dpoints']
    return np.transpose(np.array(vertices, dtype=np.float32))

def compute_errors(q_pred, t_pred, q_gt, t_gt):
    t_err = np.linalg.norm(t_pred - t_gt)
    q_pred = q_pred / np.linalg.norm(q_pred)
    q_gt = q_gt / np.linalg.norm(q_gt)
    dot = np.clip(np.abs(np.dot(q_pred, q_gt)), -1.0, 1.0)
    rad_err = 2 * np.arccos(dot)
    return t_err, np.rad2deg(rad_err)

# ------------------------------------------------------------------------------
# Main Evaluation Logic
# ------------------------------------------------------------------------------

def main():
    register_all_modules()
    
    # 1. 모델 초기화 및 Hook 등록
    print(f"Loading model from {CONFIG_FILE}...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)
    
    # SimCC Head의 X, Y 레이어에 Hook 등록 (정확한 모듈명은 Config 확인 필요, 보통 head.cls_x)
    model.head.cls_x.register_forward_hook(hook_fn_x)
    model.head.cls_y.register_forward_hook(hook_fn_y)

    # 2. 데이터 로드
    points_3d = load_tango_3d_keypoints(MODEL_3D_POINTS_FILE)
    K, dist = load_camera_intrinsics(CAMERA_FILE)
    
    # Refiner 초기화
    refiner = SimCCPoseRefiner(K, dist, MODEL_INPUT_SIZE)

    with open(POSE_GT_FILE, 'r') as f:
        pose_data = json.load(f)
        pose_list = pose_data['images'] if 'images' in pose_data else pose_data

    with open(IMAGE_META_FILE, 'r') as f:
        meta_data = json.load(f)
        original_images_map = meta_data.get('original_images', {})

    t_errors_epnp = []
    q_errors_epnp = []
    t_errors_ref = []
    q_errors_ref = []
    
    print(f"Starting evaluation with Refinement on {len(pose_list)} images...")
    
    for item in tqdm(pose_list):
        # --- (기존 파일명 처리 및 이미지 로드 로직 동일) ---
        raw_filename = item['filename']
        clean_filename = raw_filename[3:] if raw_filename.startswith('img') else raw_filename
        
        if clean_filename not in original_images_map: continue
        img_meta = original_images_map[clean_filename]
        
        img_path = os.path.join(IMG_ROOT, clean_filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(IMG_ROOT, raw_filename)
            if not os.path.exists(img_path): continue

        img_input = cv2.imread(img_path)
        if img_input is None: continue
        
        # GT Pose
        if 'r_Vo2To_vbs_true' not in item: continue
        t_gt = np.array(item['r_Vo2To_vbs_true'], dtype=np.float32)
        q_gt = np.array(item['q_vbs2tango_true'], dtype=np.float32)

        # 3. Inference (SimCC Logits은 Hook에 의해 captured_logits에 저장됨)
        results = inference_topdown(model, img_input)
        pred_instances = results[0].pred_instances
        kpts_224 = pred_instances.keypoints[0]
        
        # 4. 좌표 복원 (For EPnP)
        crop_box = img_meta['crop_box']   # [x1, y1, x2, y2]
        scale_factor = img_meta['scale_factor']
        x1, y1 = crop_box[0], crop_box[1]
        kpts_orig = (kpts_224 / scale_factor) + np.array([x1, y1])
        
        # 5. Step 1: EPnP (Initial Guess)
        points_3d_reshaped = np.ascontiguousarray(points_3d).reshape((-1, 1, 3))
        points_2d_reshaped = np.ascontiguousarray(kpts_orig).reshape((-1, 1, 2))
        
        success, rvec_init, tvec_init = cv2.solvePnP(
            points_3d_reshaped, points_2d_reshaped, K, dist, flags=cv2.SOLVEPNP_EPNP
        )

        if not success:
            continue

        # EPnP 결과 저장
        R_mat, _ = cv2.Rodrigues(rvec_init)
        q_epnp = R.from_matrix(R_mat).as_quat()[[3, 0, 1, 2]]
        t_epnp = np.squeeze(tvec_init)
        
        te, qe = compute_errors(q_epnp, t_epnp, q_gt, t_gt)
        t_errors_epnp.append(te)
        q_errors_epnp.append(qe)

        # 6. Step 2: Distribution-Alignment Refinement (Optimization)
        # Logit 준비 (Gaussian Smoothing + Spline)
        # Hook에서 가져온 Logit은 Batch 차원이 있으므로 [0] 인덱싱
        refiner.prepare_pdf(captured_logits['x'][0], captured_logits['y'][0], sigma=10.0)
        
        # 최적화 수행
        # rvec_init, tvec_init를 초기값으로 사용
        rvec_ref, tvec_ref = refiner.refine(
            rvec_init, tvec_init, 
            points_3d, 
            crop_meta={'crop_box': crop_box, 'scale_factor': scale_factor}
        )
        
        # Refinement 결과 저장
        R_mat_ref, _ = cv2.Rodrigues(rvec_ref)
        q_ref = R.from_matrix(R_mat_ref).as_quat()[[3, 0, 1, 2]]
        t_ref = np.squeeze(tvec_ref)

        te_ref, qe_ref = compute_errors(q_ref, t_ref, q_gt, t_gt)
        t_errors_ref.append(te_ref)
        q_errors_ref.append(qe_ref)

    # 결과 출력
    print("\n" + "="*40)
    print("   EVALUATION RESULTS COMPARISON")
    print("="*40)
    print(f"{'Method':<15} | {'Mean Trans Err (m)':<20} | {'Mean Rot Err (deg)':<20}")
    print("-" * 60)
    print(f"{'EPnP (Base)':<15} | {np.mean(t_errors_epnp):.6f}             | {np.mean(q_errors_epnp):.6f}")
    print(f"{'DA-Refined':<15} | {np.mean(t_errors_ref):.6f}             | {np.mean(q_errors_ref):.6f}")
    print("-" * 60)

if __name__ == '__main__':
    main()

# import os
# import json
# import cv2
# import numpy as np
# import torch
# from scipy.io import loadmat
# from scipy.spatial.transform import Rotation as R
# from scipy.ndimage import gaussian_filter1d
# from scipy.interpolate import CubicSpline
# from scipy.special import softmax
# from scipy.optimize import minimize
# from tqdm import tqdm

# from mmpose.apis import init_model, inference_topdown
# from mmpose.utils import register_all_modules

# # ------------------------------------------------------------------------------
# # Configuration
# # ------------------------------------------------------------------------------
# CONFIG_FILE = 'satellite/rtmpose-m_satellite_f.py'
# CHECKPOINT_FILE = '/workspace/rtmpose-m_f/epoch_250.pth'
# POSE_GT_FILE = '/workspace/speedplusv2/sunlamp/test.json'
# IMAGE_META_FILE = '/workspace/speedplusv2/annotations/test_sunlamp.json'
# IMG_ROOT = '/workspace/speedplusv2/sunlamp_preprocessed/'
# MODEL_3D_POINTS_FILE = '/workspace/speedplusv2/tangoPoints.mat'
# CAMERA_FILE = '/workspace/speedplusv2/camera.json'

# MODEL_INPUT_SIZE = (224, 224) 

# # ------------------------------------------------------------------------------
# # Refiner Class
# # ------------------------------------------------------------------------------
# captured_logits = {}

# def hook_fn_x(module, input, output):
#     captured_logits['x'] = output.detach().cpu().numpy()

# def hook_fn_y(module, input, output):
#     captured_logits['y'] = output.detach().cpu().numpy()

# class SimCCPoseRefiner:
#     def __init__(self, K, dist, model_input_size):
#         self.K = K
#         self.dist = dist
#         self.w_model, self.h_model = model_input_size
#         self.splines_x = []
#         self.splines_y = []
#         self.simcc_scale_x = 1.0
#         self.simcc_scale_y = 1.0

#     def filter_valid_keypoints(self, logits_x, logits_y, threshold=0.5, min_kpts=6):
#         """
#         [New Feature] Confidence 기반 필터링
#         1. Score < threshold 이면 제외
#         2. 남은 개수가 min_kpts보다 적으면 Top-k 강제 선택
#         Returns: valid_indices (np.array)
#         """
#         # 1. Confidence 계산 (Softmax Max Prob)
#         # logits shape: (N, 448)
#         probs_x = softmax(logits_x, axis=1)
#         probs_y = softmax(logits_y, axis=1)
        
#         conf_x = np.max(probs_x, axis=1)
#         conf_y = np.max(probs_y, axis=1)
        
#         # 기하평균 혹은 최소값을 전체 Score로 사용
#         scores = np.sqrt(conf_x * conf_y)
        
#         # 2. Thresholding
#         valid_mask = scores >= threshold
#         valid_count = np.sum(valid_mask)
        
#         # 3. Fallback Logic (최소 개수 보장)
#         if valid_count < min_kpts:
#             # 점수가 높은 순서대로 정렬하여 상위 min_kpts개 인덱스 가져오기
#             # argsort는 오름차순이므로 [::-1]로 뒤집어서 큰 값부터 가져옴
#             sorted_indices = np.argsort(scores)[::-1]
#             valid_indices = sorted_indices[:min_kpts]
#             # 인덱스 정렬 (순서 꼬임 방지)
#             valid_indices = np.sort(valid_indices)
#         else:
#             valid_indices = np.where(valid_mask)[0]
            
#         return valid_indices

#     def prepare_pdf(self, logits_x, logits_y, sigma=2.0):
#         """
#         선별된 Logits에 대해 Spline 생성
#         """
#         self.splines_x = []
#         self.splines_y = []
        
#         n_kpts, n_bins_x = logits_x.shape
#         _, n_bins_y = logits_y.shape
        
#         self.simcc_scale_x = n_bins_x / self.w_model
#         self.simcc_scale_y = n_bins_y / self.h_model
        
#         epsilon = 1e-6

#         for i in range(n_kpts):
#             # Softmax & Smoothing
#             prob_x = gaussian_filter1d(softmax(logits_x[i]), sigma=sigma)
#             prob_y = gaussian_filter1d(softmax(logits_y[i]), sigma=sigma)
            
#             # Log-Prob
#             log_prob_x = np.log(np.clip(prob_x, epsilon, 1.0))
#             log_prob_y = np.log(np.clip(prob_y, epsilon, 1.0))
            
#             # Cubic Spline
#             x_axis = np.arange(n_bins_x)
#             y_axis = np.arange(n_bins_y)
            
#             self.splines_x.append(CubicSpline(x_axis, log_prob_x))
#             self.splines_y.append(CubicSpline(y_axis, log_prob_y))

#     def objective_function(self, params, points_3d, crop_meta):
#         """
#         Filtered 3D Points에 대해서만 수행됨
#         """
#         rvec, tvec = params[:3], params[3:]
        
#         # Project Points
#         img_points, _ = cv2.projectPoints(points_3d, rvec, tvec, self.K, self.dist)
#         img_points = img_points.squeeze()
        
#         # Coordinate Transform
#         x1, y1 = crop_meta['crop_box'][:2]
#         scale = crop_meta['scale_factor']
#         if isinstance(scale, (list, np.ndarray)): scale = np.array(scale)
        
#         model_pts = (img_points - np.array([x1, y1])) * scale
#         simcc_pts_x = model_pts[:, 0] * self.simcc_scale_x
#         simcc_pts_y = model_pts[:, 1] * self.simcc_scale_y
        
#         nll_sum = 0.0
#         n_bins_x = len(self.splines_x[0].x)
#         n_bins_y = len(self.splines_y[0].x)
#         penalty = 100.0

#         for i in range(len(self.splines_x)):
#             ux, uy = simcc_pts_x[i], simcc_pts_y[i]
            
#             # Bounds Check & NLL Calculation
#             if not (0 <= ux < n_bins_x):
#                 nll_sum += (penalty + abs(ux - n_bins_x/2)*0.1)
#             else:
#                 nll_sum -= self.splines_x[i](ux)

#             if not (0 <= uy < n_bins_y):
#                 nll_sum += (penalty + abs(uy - n_bins_y/2)*0.1)
#             else:
#                 nll_sum -= self.splines_y[i](uy)

#         return nll_sum

#     def refine(self, rvec_init, tvec_init, points_3d, crop_meta):
#         initial_params = np.concatenate([rvec_init.flatten(), tvec_init.flatten()])
        
#         # Nelder-Mead for Robustness
#         res = minimize(
#             self.objective_function,
#             initial_params,
#             args=(points_3d, crop_meta),
#             method='Nelder-Mead', 
#             options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
#         )
#         return res.x[:3], res.x[3:]

# # ------------------------------------------------------------------------------
# # Helpers
# # ------------------------------------------------------------------------------
# def load_camera_intrinsics(path):
#     with open(path) as f: cam = json.load(f)
#     return np.array(cam['cameraMatrix'], dtype=np.float32), np.array(cam['distCoeffs'], dtype=np.float32).flatten()

# def load_tango_3d_keypoints(path):
#     return np.transpose(np.array(loadmat(path)['tango3Dpoints'], dtype=np.float32))

# def compute_errors(rvec, tvec, q_gt, t_gt):
#     R_mat, _ = cv2.Rodrigues(rvec)
#     q_pred = R.from_matrix(R_mat).as_quat()[[3, 0, 1, 2]]
#     t_err = np.linalg.norm(tvec - t_gt)
#     q_pred = q_pred / np.linalg.norm(q_pred)
#     q_gt = q_gt / np.linalg.norm(q_gt)
#     rad_err = 2 * np.arccos(np.clip(np.abs(np.dot(q_pred, q_gt)), -1.0, 1.0))
#     return t_err, np.rad2deg(rad_err)

# # ------------------------------------------------------------------------------
# # Main
# # ------------------------------------------------------------------------------
# def main():
#     register_all_modules()
    
#     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#     model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)
#     model.head.cls_x.register_forward_hook(hook_fn_x)
#     model.head.cls_y.register_forward_hook(hook_fn_y)

#     points_3d_all = load_tango_3d_keypoints(MODEL_3D_POINTS_FILE) # (N_all, 3)
#     K, dist = load_camera_intrinsics(CAMERA_FILE)
    
#     refiner = SimCCPoseRefiner(K, dist, MODEL_INPUT_SIZE)

#     with open(POSE_GT_FILE, 'r') as f:
#         pose_data = json.load(f)
#         pose_list = pose_data['images'] if 'images' in pose_data else pose_data
#     with open(IMAGE_META_FILE, 'r') as f:
#         original_images_map = json.load(f).get('original_images', {})

#     t_errors, q_errors = [], []
    
#     print(f"Starting Robust Evaluation on {len(pose_list)} images...")
    
#     for item in tqdm(pose_list):
#         # ... (이미지 로드 및 전처리 생략 - 기존과 동일) ...
#         raw_filename = item['filename']
#         clean_filename = raw_filename[3:] if raw_filename.startswith('img') else raw_filename
#         if clean_filename not in original_images_map: continue
#         img_meta = original_images_map[clean_filename]
        
#         img_path = os.path.join(IMG_ROOT, clean_filename)
#         if not os.path.exists(img_path): img_path = os.path.join(IMG_ROOT, raw_filename)
#         if not os.path.exists(img_path): continue
#         img_input = cv2.imread(img_path)
#         if img_input is None: continue

#         if 'r_Vo2To_vbs_true' not in item: continue
#         t_gt = np.array(item['r_Vo2To_vbs_true'], dtype=np.float32)
#         q_gt = np.array(item['q_vbs2tango_true'], dtype=np.float32)

#         # Inference
#         results = inference_topdown(model, img_input)
#         kpts_224 = results[0].pred_instances.keypoints[0]
        
#         # ----------------------------------------------------------------
#         # [NEW] Keypoint Filtering & Selection
#         # ----------------------------------------------------------------
#         logits_x = captured_logits['x'][0]
#         logits_y = captured_logits['y'][0]
        
#         # 1. 유효한 인덱스 추출 (Threshold 0.5, Min Count 6)
#         valid_indices = refiner.filter_valid_keypoints(
#             logits_x, logits_y, threshold=0.5, min_kpts=6
#         )
        
#         # 2. 데이터 서브셋(Subset) 생성 (Slicing)
#         # 중요: 3D Point, Logit, 2D Point 모두 같은 인덱스로 잘라야 함
#         subset_p3d = points_3d_all[valid_indices]
#         subset_logits_x = logits_x[valid_indices]
#         subset_logits_y = logits_y[valid_indices]
#         subset_kpts_224 = kpts_224[valid_indices]
        
#         # ----------------------------------------------------------------
#         # [Step 1] EPnP (Filtered Data 사용)
#         # ----------------------------------------------------------------
#         crop_box = img_meta['crop_box']
#         scale_factor = img_meta['scale_factor']
#         x1, y1 = crop_box[0], crop_box[1]
        
#         # 2D 좌표 복원 (Filtered)
#         kpts_orig = (subset_kpts_224 / scale_factor) + np.array([x1, y1])
        
#         subset_p3d_reshaped = np.ascontiguousarray(subset_p3d).reshape((-1, 1, 3))
#         kpts_orig_reshaped = np.ascontiguousarray(kpts_orig).reshape((-1, 1, 2))
        
#         success, rvec_init, tvec_init = cv2.solvePnP(
#             subset_p3d_reshaped, kpts_orig_reshaped, K, dist, flags=cv2.SOLVEPNP_EPNP
#         )
        
#         if not success:
#             continue

#         # ----------------------------------------------------------------
#         # [Step 2] Distribution Refinement (Filtered Data 사용)
#         # ----------------------------------------------------------------
#         # PDF 준비 (선별된 Logits만 사용)
#         refiner.prepare_pdf(subset_logits_x, subset_logits_y, sigma=10.0)
        
#         # 최적화 수행 (선별된 3D Point만 사용)
#         rvec_ref, tvec_ref = refiner.refine(
#             rvec_init, tvec_init, 
#             subset_p3d, 
#             crop_meta={'crop_box': crop_box, 'scale_factor': scale_factor}
#         )
        
#         te, qe = compute_errors(rvec_ref, tvec_ref, q_gt, t_gt)
#         t_errors.append(te)
#         q_errors.append(qe)

#     if len(t_errors) > 0:
#         print(f"\nFiltered & Refined Results (Mean Trans/Rot): {np.mean(t_errors):.6f} m / {np.mean(q_errors):.6f} deg")
#     else:
#         print("No valid results.")

# if __name__ == '__main__':
#     main()