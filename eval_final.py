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
# 1. Configuration
# ------------------------------------------------------------------------------
CONFIG_FILE = 'satellite/rtmpose-s_satellite.py'
CHECKPOINT_FILE = '/workspace/rtmpose-s/epoch_280.pth'

POSE_GT_FILE = '/workspace/speedplusv2/lightbox/test.json'
IMAGE_META_FILE = '/workspace/speedplusv2/annotations/test_lightbox.json'
IMG_ROOT = '/workspace/speedplusv2/lightbox_preprocessed/'
MODEL_3D_POINTS_FILE = '/workspace/speedplusv2/tangoPoints.mat'
CAMERA_FILE = '/workspace/speedplusv2/camera.json'

MODEL_INPUT_SIZE = (224, 224) 

# HIL-Specific Thresholds
THETA_T_NORM = 2.173e-3  # Threshold for Normalized Translation Error
THETA_Q_DEG = 0.169      # Threshold for Rotation Error (Degree)

# ------------------------------------------------------------------------------
# 2. Weighted SimCC Refiner Class
# ------------------------------------------------------------------------------

class WeightedSimCCPoseRefiner:
    def __init__(self, K, dist, model_input_size):
        self.K = K
        self.dist = dist
        self.w_model, self.h_model = model_input_size
        self.splines_x = []
        self.splines_y = []
        self.weights = [] 
        self.simcc_scale_x = 1.0
        self.simcc_scale_y = 1.0

    def prepare_pdf_and_weights(self, logits_x, logits_y, sigma=2.0):
        """
        Logits로부터 PDF(Spline)와 Weights를 생성.
        Boundary Filtering: Max Index가 0이나 끝값인 경우 Weight = 0
        """
        self.splines_x = []
        self.splines_y = []
        self.weights = []
        
        n_kpts, n_bins_x = logits_x.shape
        _, n_bins_y = logits_y.shape
        
        # Softmax 적용
        probs_x_all = softmax(logits_x, axis=1)
        probs_y_all = softmax(logits_y, axis=1)
        
        # Confidence & Argmax 계산
        conf_x = np.max(logits_x, axis=1)
        conf_y = np.max(logits_y, axis=1)
        
        idx_x = np.argmax(probs_x_all, axis=1)
        idx_y = np.argmax(probs_y_all, axis=1)

        # Weight Calculation (Soft Weighting + Boundary Filtering)
        raw_weights = np.minimum(conf_x, conf_y)
        
        # 경계(0 또는 Max)에 피크가 있으면 가중치 0 처리
        is_boundary_x = (idx_x <= 5) | (idx_x >= n_bins_x - 6)
        is_boundary_y = (idx_y <= 5) | (idx_y >= n_bins_y - 6)
        is_boundary = is_boundary_x | is_boundary_y
        
        raw_weights[is_boundary] = 0.0
        
        self.weights = raw_weights
        
        # PDF (Spline) 준비
        self.simcc_scale_x = n_bins_x / self.w_model
        self.simcc_scale_y = n_bins_y / self.h_model
        epsilon = 1e-6 

        for i in range(n_kpts):
            prob_x = gaussian_filter1d(probs_x_all[i], sigma=sigma)
            prob_y = gaussian_filter1d(probs_y_all[i], sigma=sigma)
            
            log_prob_x = np.log(np.clip(prob_x, epsilon, 1.0))
            log_prob_y = np.log(np.clip(prob_y, epsilon, 1.0))
            
            x_axis = np.arange(n_bins_x)
            y_axis = np.arange(n_bins_y)
            
            self.splines_x.append(CubicSpline(x_axis, log_prob_x))
            self.splines_y.append(CubicSpline(y_axis, log_prob_y))

    def objective_function(self, params, points_3d, crop_meta):
        """
        Weighted NLL Minimization
        """
        rvec, tvec = params[:3], params[3:]
        
        img_points, _ = cv2.projectPoints(points_3d, rvec, tvec, self.K, self.dist)
        img_points = img_points.squeeze()
        
        x1, y1 = crop_meta['crop_box'][:2]
        scale = crop_meta['scale_factor']
        if isinstance(scale, (list, np.ndarray)): scale = np.array(scale)
        
        model_pts = (img_points - np.array([x1, y1])) * scale
        simcc_pts_x = model_pts[:, 0] * self.simcc_scale_x
        simcc_pts_y = model_pts[:, 1] * self.simcc_scale_y
        
        weighted_nll_sum = 0.0
        n_bins_x = len(self.splines_x[0].x)
        n_bins_y = len(self.splines_y[0].x)
        penalty_base = 100.0

        for i in range(len(self.splines_x)):
            w = self.weights[i]
            if w < 1e-6: continue # Boundary Filtered

            ux, uy = simcc_pts_x[i], simcc_pts_y[i]
            
            if not (0 <= ux < n_bins_x):
                val_x = -(penalty_base + abs(ux - n_bins_x/2)*0.1)
            else:
                val_x = self.splines_x[i](ux)

            if not (0 <= uy < n_bins_y):
                val_y = -(penalty_base + abs(uy - n_bins_y/2)*0.1)
            else:
                val_y = self.splines_y[i](uy)
            
            weighted_nll_sum -= w * (val_x + val_y)

        return weighted_nll_sum

    def refine(self, rvec_init, tvec_init, points_3d, crop_meta):
        initial_params = np.concatenate([rvec_init.flatten(), tvec_init.flatten()])
        
        res = minimize(
            self.objective_function,
            initial_params,
            args=(points_3d, crop_meta),
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
        )
        return res.x[:3], res.x[3:]

# ------------------------------------------------------------------------------
# 3. Metrics Calculation
# ------------------------------------------------------------------------------

def load_camera_intrinsics(path):
    with open(path) as f: cam = json.load(f)
    return np.array(cam['cameraMatrix'], dtype=np.float32), np.array(cam['distCoeffs'], dtype=np.float32).flatten()

def load_tango_3d_keypoints(path):
    return np.transpose(np.array(loadmat(path)['tango3Dpoints'], dtype=np.float32))

def compute_metrics_and_stars(rvec, tvec, q_gt, t_gt):
    """
    모든 Raw Metric과 HIL Threshold가 적용된 e* Metric을 한 번에 계산
    """
    # 1. Rotation Error
    R_mat, _ = cv2.Rodrigues(rvec)
    q_pred = R.from_matrix(R_mat).as_quat()[[3, 0, 1, 2]]
    
    q_pred = q_pred / np.linalg.norm(q_pred)
    q_gt = q_gt / np.linalg.norm(q_gt)
    dot = np.clip(np.abs(np.dot(q_pred, q_gt)), -1.0, 1.0)
    
    e_q_rad = 2 * np.arccos(dot)
    e_q_deg = np.rad2deg(e_q_rad)
    
    # 2. Translation Error
    e_t_m = np.linalg.norm(tvec - t_gt)
    norm_t_gt = np.linalg.norm(t_gt)
    e_t_bar = e_t_m / norm_t_gt if norm_t_gt > 1e-6 else 0.0

    # 3. Apply Thresholds (e* 계산)
    # e*_t
    if e_t_bar < THETA_T_NORM:
        e_star_t = 0.0
    else:
        e_star_t = e_t_bar

    # e*_q (Degree Threshold 확인, 값은 Degree/Rad 각각 저장)
    if e_q_deg < THETA_Q_DEG:
        e_star_q_deg = 0.0
        e_star_q_rad = 0.0
    else:
        e_star_q_deg = e_q_deg
        e_star_q_rad = e_q_rad

    # 4. Pose Error (e_pose) 계산
    # 수정된 요구사항: e_pose = e*_t + e*_q(rad)
    e_pose_star = e_star_t + e_star_q_rad
    
    return {
        'e_t_m': e_t_m,
        'e_t_bar': e_t_bar,
        'e_q_deg': e_q_deg,
        'e_star_t': e_star_t,
        'e_star_q_deg': e_star_q_deg,
        'e_pose_star': e_pose_star # Combined HIL Metric
    }

def apply_hil_thresholds(metrics):
    """
    HIL Threshold 적용하여 e* 메트릭 생성
    """
    # e*_t: Normalized Trans Error에 대해 Threshold 적용
    if metrics['e_t_bar'] < THETA_T_NORM:
        e_star_t = 0.0
    else:
        e_star_t = metrics['e_t_bar']
        
    # e*_q: Rotation Error(Deg)에 대해 Threshold 적용
    if metrics['e_q_deg'] < THETA_Q_DEG:
        e_star_q = 0.0
    else:
        e_star_q = metrics['e_q_deg']
        
    return e_star_t, e_star_q

# ------------------------------------------------------------------------------
# 4. Main Loop
# ------------------------------------------------------------------------------

captured_logits = {}
def hook_fn_x(module, input, output): captured_logits['x'] = output.detach().cpu().numpy()
def hook_fn_y(module, input, output): captured_logits['y'] = output.detach().cpu().numpy()

def main():
    register_all_modules()
    
    device = 'cpu'#'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Loading Model on {device}...")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)
    model.head.cls_x.register_forward_hook(hook_fn_x)
    model.head.cls_y.register_forward_hook(hook_fn_y)

    points_3d = load_tango_3d_keypoints(MODEL_3D_POINTS_FILE)
    K, dist = load_camera_intrinsics(CAMERA_FILE)
    
    refiner = WeightedSimCCPoseRefiner(K, dist, MODEL_INPUT_SIZE)

    with open(POSE_GT_FILE, 'r') as f:
        pose_data = json.load(f)
        pose_list = pose_data['images'] if 'images' in pose_data else pose_data
    with open(IMAGE_META_FILE, 'r') as f:
        original_images_map = json.load(f).get('original_images', {})

    # 결과 저장용
    history = {
        'EPnP':    {'t_m': [], 't_bar': [], 'q_deg': [], 'star_t': [], 'star_q': [], 'pose_star': []},
        'Refined': {'t_m': [], 't_bar': [], 'q_deg': [], 'star_t': [], 'star_q': [], 'pose_star': []}
    }
    
    print(f"Starting Comparison Evaluation on {len(pose_list)} images...")
    
    for item in tqdm(pose_list):
        raw_filename = item['filename']
        clean_filename = raw_filename[3:] if raw_filename.startswith('img') else raw_filename
        if clean_filename not in original_images_map: continue
        img_meta = original_images_map[clean_filename]
        
        img_path = os.path.join(IMG_ROOT, clean_filename)
        if not os.path.exists(img_path): img_path = os.path.join(IMG_ROOT, raw_filename)
        if not os.path.exists(img_path): continue
        img_input = cv2.imread(img_path)
        if img_input is None: continue

        if 'r_Vo2To_vbs_true' not in item: continue
        t_gt = np.array(item['r_Vo2To_vbs_true'], dtype=np.float32)
        q_gt = np.array(item['q_vbs2tango_true'], dtype=np.float32)

        results = inference_topdown(model, img_input)
        kpts_224 = results[0].pred_instances.keypoints[0]
        
        crop_box = img_meta['crop_box']
        scale_factor = img_meta['scale_factor']
        x1, y1 = crop_box[0], crop_box[1]
        kpts_orig = (kpts_224 / scale_factor) + np.array([x1, y1])

        # [Method 1] Pure EPnP
        p3d_reshaped = np.ascontiguousarray(points_3d).reshape((-1, 1, 3))
        kpts_reshaped = np.ascontiguousarray(kpts_orig).reshape((-1, 1, 2))
        
        success, rvec_epnp, tvec_epnp = cv2.solvePnP(
            p3d_reshaped, kpts_reshaped, K, dist, flags=cv2.SOLVEPNP_EPNP
        )
        
        if success:
            m = compute_metrics_and_stars(rvec_epnp, np.squeeze(tvec_epnp), q_gt, t_gt)
            
            history['EPnP']['t_m'].append(m['e_t_m'])
            history['EPnP']['t_bar'].append(m['e_t_bar'])
            history['EPnP']['q_deg'].append(m['e_q_deg'])
            history['EPnP']['star_t'].append(m['e_star_t'])
            history['EPnP']['star_q'].append(m['e_star_q_deg'])
            history['EPnP']['pose_star'].append(m['e_pose_star'])
        else:
            continue

        # [Method 2] Weighted Refinement
        if success:
            logits_x = captured_logits['x'][0]
            logits_y = captured_logits['y'][0]
            
            refiner.prepare_pdf_and_weights(logits_x, logits_y, sigma=10.0)
            
            rvec_ref, tvec_ref = refiner.refine(
                rvec_epnp, tvec_epnp, 
                points_3d, 
                crop_meta={'crop_box': crop_box, 'scale_factor': scale_factor}
            )
            
            m_ref = compute_metrics_and_stars(rvec_ref, np.squeeze(tvec_ref), q_gt, t_gt)
            
            history['Refined']['t_m'].append(m_ref['e_t_m'])
            history['Refined']['t_bar'].append(m_ref['e_t_bar'])
            history['Refined']['q_deg'].append(m_ref['e_q_deg'])
            history['Refined']['star_t'].append(m_ref['e_star_t'])
            history['Refined']['star_q'].append(m_ref['e_star_q_deg'])
            history['Refined']['pose_star'].append(m_ref['e_pose_star'])

    # Final Report
    print("\n" + "="*95)
    print(f"{'EVALUATION METRICS SUMMARY (HIL THRESHOLDS APPLIED)':^95}")
    print("="*95)
    print(f"{'Metric':<25} | {'Pure EPnP':<20} | {'SimCC Refinement':<20} | {'Improvement':<15}")
    print("-" * 95)

    keys_to_print = [
        ('t_m',       'e_t (m, raw)'),
        ('t_bar',     'e_t_bar (norm)'),
        ('q_deg',     'e_q (deg, raw)'),
        ('star_t',    'e*_t (HIL norm)'),
        ('star_q',    'e*_q (HIL deg)'),
        ('pose_star', 'e*_pose (sum)')
    ]

    if len(history['EPnP']['t_m']) > 0:
        for key, label in keys_to_print:
            mean_epnp = np.mean(history['EPnP'][key])
            mean_ref = np.mean(history['Refined'][key])
            
            imp = (mean_epnp - mean_ref) / mean_epnp * 100
            
            print(f"{label:<25} | {mean_epnp:.6f}{' '*11} | {mean_ref:.6f}{' '*11} | {imp:6.2f}%")
    else:
        print("No valid results found.")
    
    print("-" * 95)
    print(f"Total Valid Images Evaluated: {len(history['EPnP']['t_m'])}")

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
# from scipy.optimize import least_squares, minimize
# from tqdm import tqdm

# from mmpose.apis import init_model, inference_topdown
# from mmpose.utils import register_all_modules

# # ------------------------------------------------------------------------------
# # 1. Configuration
# # ------------------------------------------------------------------------------
# CONFIG_FILE = 'satellite/rtmpose-m_satellite_f.py'
# CHECKPOINT_FILE = '/workspace/rtmpose-m_f/epoch_420.pth'

# POSE_GT_FILE = '/workspace/speedplusv2/sunlamp/test.json'
# IMAGE_META_FILE = '/workspace/speedplusv2/annotations/test_sunlamp.json'
# IMG_ROOT = '/workspace/speedplusv2/sunlamp_preprocessed/'
# MODEL_3D_POINTS_FILE = '/workspace/speedplusv2/tangoPoints.mat'
# CAMERA_FILE = '/workspace/speedplusv2/camera.json'

# MODEL_INPUT_SIZE = (224, 224) 

# # HIL-Specific Thresholds
# THETA_T = 2.173e-3  # Normalized Translation Threshold
# THETA_Q = 0.169     # Rotation Threshold (Degree)

# # ------------------------------------------------------------------------------
# # 2. Weighted SimCC Refiner Class (With Boundary Filtering)
# # ------------------------------------------------------------------------------

# class WeightedSimCCPoseRefiner:
#     def __init__(self, K, dist, model_input_size):
#         self.K = K
#         self.dist = dist
#         self.w_model, self.h_model = model_input_size
#         self.splines_x = []
#         self.splines_y = []
#         self.weights = [] 
#         self.simcc_scale_x = 1.0
#         self.simcc_scale_y = 1.0

#     def prepare_pdf_and_weights(self, logits_x, logits_y, sigma=2.0):
#         """
#         Logits로부터 PDF(Spline)와 Weights를 생성.
#         [NEW] Boundary Filtering: Max Index가 0이나 끝값인 경우 Weight = 0
#         """
#         self.splines_x = []
#         self.splines_y = []
#         self.weights = []
        
#         # 1. SimCC Bin 크기 확인
#         n_kpts, n_bins_x = logits_x.shape
#         _, n_bins_y = logits_y.shape
        
#         # 2. Softmax 적용
#         probs_x_all = softmax(logits_x, axis=1)
#         probs_y_all = softmax(logits_y, axis=1)
        
#         # 3. Confidence & Argmax 계산
#         conf_x = np.max(logits_x, axis=1)
#         conf_y = np.max(logits_y, axis=1)
        
#         idx_x = np.argmax(probs_x_all, axis=1)
#         idx_y = np.argmax(probs_y_all, axis=1)

#         # 4. [Core Logic] Weight Calculation with Boundary Filtering
#         # 기본 Score: 둘 중 작은 Confidence 사용
#         raw_weights = np.minimum(conf_x, conf_y)
        
#         # 경계 조건 확인 (0 또는 Max Index)
#         # SimCC가 물체가 짤렸을 때 끝값에 피크를 띄우는 경향을 제거
#         is_boundary_x = (idx_x <= 4) | (idx_x >= n_bins_x - 5)
#         is_boundary_y = (idx_y <= 4) | (idx_y >= n_bins_y - 5)
#         is_boundary = is_boundary_x | is_boundary_y
        
#         # 경계에 있는 점은 가중치를 0으로 설정 (Optimization에서 제외됨)
#         raw_weights[is_boundary] = 0.0
        
#         self.weights = raw_weights
        
#         # 5. PDF (Spline) 준비
#         self.simcc_scale_x = n_bins_x / self.w_model
#         self.simcc_scale_y = n_bins_y / self.h_model
#         epsilon = 1e-6 

#         for i in range(n_kpts):
#             # Smoothing & Spline (가중치가 0이어도 Spline은 생성해둠, 계산시 0 곱해짐)
#             prob_x = gaussian_filter1d(probs_x_all[i], sigma=sigma)
#             prob_y = gaussian_filter1d(probs_y_all[i], sigma=sigma)
            
#             log_prob_x = np.log(np.clip(prob_x, epsilon, 1.0))
#             log_prob_y = np.log(np.clip(prob_y, epsilon, 1.0))
            
#             x_axis = np.arange(n_bins_x)
#             y_axis = np.arange(n_bins_y)
            
#             self.splines_x.append(CubicSpline(x_axis, log_prob_x))
#             self.splines_y.append(CubicSpline(y_axis, log_prob_y))

#     def objective_function(self, params, points_3d, crop_meta):
#         """
#         Weighted NLL Minimization
#         """
#         rvec, tvec = params[:3], params[3:]
        
#         img_points, _ = cv2.projectPoints(points_3d, rvec, tvec, self.K, self.dist)
#         img_points = img_points.squeeze()
        
#         x1, y1 = crop_meta['crop_box'][:2]
#         scale = crop_meta['scale_factor']
#         if isinstance(scale, (list, np.ndarray)): scale = np.array(scale)
        
#         model_pts = (img_points - np.array([x1, y1])) * scale
#         simcc_pts_x = model_pts[:, 0] * self.simcc_scale_x
#         simcc_pts_y = model_pts[:, 1] * self.simcc_scale_y
        
#         weighted_nll_sum = 0.0
#         n_bins_x = len(self.splines_x[0].x)
#         n_bins_y = len(self.splines_y[0].x)
#         penalty_base = 100.0

#         for i in range(len(self.splines_x)):
#             w = self.weights[i]
            
#             # 가중치가 0이면(Boundary Filtered) 계산 건너뜀 -> 속도 및 안정성 향상
#             if w < 1e-6:
#                 continue

#             ux, uy = simcc_pts_x[i], simcc_pts_y[i]
            
#             if not (0 <= ux < n_bins_x):
#                 val_x = -(penalty_base + abs(ux - n_bins_x/2)*0.1)
#             else:
#                 val_x = self.splines_x[i](ux)

#             if not (0 <= uy < n_bins_y):
#                 val_y = -(penalty_base + abs(uy - n_bins_y/2)*0.1)
#             else:
#                 val_y = self.splines_y[i](uy)
            
#             weighted_nll_sum -= w * (val_x + val_y)

#         return weighted_nll_sum

#     def refine(self, rvec_init, tvec_init, points_3d, crop_meta):
#         initial_params = np.concatenate([rvec_init.flatten(), tvec_init.flatten()])
        
#         # 초기값의 신뢰도가 높으므로 Nelder-Mead로 Local Fine-tuning
#         res = minimize(
#             self.objective_function,
#             initial_params,
#             args=(points_3d, crop_meta),
#             method='Nelder-Mead',
#             options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
#         )
#         return res.x[:3], res.x[3:]

# # ------------------------------------------------------------------------------
# # 3. Metrics & Helpers
# # ------------------------------------------------------------------------------

# def load_camera_intrinsics(path):
#     with open(path) as f: cam = json.load(f)
#     return np.array(cam['cameraMatrix'], dtype=np.float32), np.array(cam['distCoeffs'], dtype=np.float32).flatten()

# def load_tango_3d_keypoints(path):
#     return np.transpose(np.array(loadmat(path)['tango3Dpoints'], dtype=np.float32))

# def compute_raw_errors(rvec, tvec, q_gt, t_gt):
#     """
#     기본 Translation error(m)와 Rotation error(deg)를 계산
#     """
#     R_mat, _ = cv2.Rodrigues(rvec)
#     q_pred = R.from_matrix(R_mat).as_quat()[[3, 0, 1, 2]] # [w, x, y, z]
    
#     # 1. Translation Error
#     t_err = np.linalg.norm(tvec - t_gt)
    
#     # 2. Rotation Error
#     q_pred = q_pred / np.linalg.norm(q_pred)
#     q_gt = q_gt / np.linalg.norm(q_gt)
#     dot = np.clip(np.abs(np.dot(q_pred, q_gt)), -1.0, 1.0)
#     rad_err = 2 * np.arccos(dot)
#     deg_err = np.rad2deg(rad_err)
    
#     return t_err, deg_err

# def compute_hil_metrics(t_err, deg_err):
#     """
#     [New] HIL-specific Metric (e*) 적용
#     Threshold보다 작으면 Error를 0으로 간주 (Perfect Prediction in HIL context)
#     """
#     # e*_t
#     if t_err < THETA_T:
#         e_star_t = 0.0
#     else:
#         e_star_t = t_err
        
#     # e*_q
#     if deg_err < THETA_Q:
#         e_star_q = 0.0
#     else:
#         e_star_q = deg_err
        
#     return e_star_t, e_star_q

# # ------------------------------------------------------------------------------
# # 4. Main Loop
# # ------------------------------------------------------------------------------

# captured_logits = {}
# def hook_fn_x(module, input, output): captured_logits['x'] = output.detach().cpu().numpy()
# def hook_fn_y(module, input, output): captured_logits['y'] = output.detach().cpu().numpy()

# def main():
#     register_all_modules()
    
#     device = 'cpu' #'cuda:0' if torch.cuda.is_available() else 'cpu'
#     print(f"Loading Model on {device}...")
#     model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)
#     model.head.cls_x.register_forward_hook(hook_fn_x)
#     model.head.cls_y.register_forward_hook(hook_fn_y)

#     points_3d = load_tango_3d_keypoints(MODEL_3D_POINTS_FILE)
#     K, dist = load_camera_intrinsics(CAMERA_FILE)
    
#     refiner = WeightedSimCCPoseRefiner(K, dist, MODEL_INPUT_SIZE)

#     with open(POSE_GT_FILE, 'r') as f:
#         pose_data = json.load(f)
#         pose_list = pose_data['images'] if 'images' in pose_data else pose_data
#     with open(IMAGE_META_FILE, 'r') as f:
#         original_images_map = json.load(f).get('original_images', {})

#     # 결과 저장을 위한 리스트 (EPnP vs Refined)
#     results_epnp = {'t_star': [], 'q_star': [], 'raw_t': [], 'raw_q': []}
#     results_ref = {'t_star': [], 'q_star': [], 'raw_t': [], 'raw_q': []}
    
#     print(f"Starting Comparison Evaluation on {len(pose_list)} images...")
    
#     for item in tqdm(pose_list):
#         # 1. Image Load & Prep
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

#         # 2. Inference
#         results = inference_topdown(model, img_input)
#         kpts_224 = results[0].pred_instances.keypoints[0]
        
#         # 3. Coordinate Restore
#         crop_box = img_meta['crop_box']
#         scale_factor = img_meta['scale_factor']
#         x1, y1 = crop_box[0], crop_box[1]
#         kpts_orig = (kpts_224 / scale_factor) + np.array([x1, y1])

#         # ---------------------------------------------------------
#         # [Method 1] Pure EPnP
#         # ---------------------------------------------------------
#         p3d_reshaped = np.ascontiguousarray(points_3d).reshape((-1, 1, 3))
#         kpts_reshaped = np.ascontiguousarray(kpts_orig).reshape((-1, 1, 2))
        
#         success, rvec_epnp, tvec_epnp = cv2.solvePnP(
#             p3d_reshaped, kpts_reshaped, K, dist, flags=cv2.SOLVEPNP_EPNP
#         )
        
#         if success:
#             t_err, q_err = compute_raw_errors(rvec_epnp, np.squeeze(tvec_epnp), q_gt, t_gt)
#             t_star, q_star = compute_hil_metrics(t_err, q_err)
            
#             results_epnp['raw_t'].append(t_err)
#             results_epnp['raw_q'].append(q_err)
#             results_epnp['t_star'].append(t_star)
#             results_epnp['q_star'].append(q_star)
#         else:
#             # PnP 실패 시 None 혹은 큰 값 처리 (여기선 생략)
#             pass

#         # ---------------------------------------------------------
#         # [Method 2] Weighted Refinement (SimCC Distribution)
#         # ---------------------------------------------------------
#         if success: # EPnP 성공 시 초기값으로 사용
#             logits_x = captured_logits['x'][0]
#             logits_y = captured_logits['y'][0]
            
#             # PDF 및 Weight 준비 (Boundary Filtering 적용됨)
#             refiner.prepare_pdf_and_weights(logits_x, logits_y, sigma=10.0)
            
#             # Optimization
#             rvec_ref, tvec_ref = refiner.refine(
#                 rvec_epnp, tvec_epnp, 
#                 points_3d, 
#                 crop_meta={'crop_box': crop_box, 'scale_factor': scale_factor}
#             )
            
#             t_err_ref, q_err_ref = compute_raw_errors(rvec_ref, np.squeeze(tvec_ref), q_gt, t_gt)
#             t_star_ref, q_star_ref = compute_hil_metrics(t_err_ref, q_err_ref)
            
#             results_ref['raw_t'].append(t_err_ref)
#             results_ref['raw_q'].append(q_err_ref)
#             results_ref['t_star'].append(t_star_ref)
#             results_ref['q_star'].append(q_star_ref)

#     # 4. Final Report
#     print("\n" + "="*80)
#     print(f"{'METRIC COMPARISON':^80}")
#     print("="*80)
    
#     # Helper to print stats
#     def print_stats(name, data_dict):
#         mean_t = np.mean(data_dict['t_star'])
#         mean_q = np.mean(data_dict['q_star'])
#         raw_mean_t = np.mean(data_dict['raw_t'])
#         raw_mean_q = np.mean(data_dict['raw_q'])
        
#         print(f"Method: {name}")
#         print(f"  > Raw Error (Avg)   | Trans: {raw_mean_t:.6f} m    | Rot: {raw_mean_q:.6f} deg")
#         print(f"  > HIL Error (e*)    | Trans: {mean_t:.6f} (norm) | Rot: {mean_q:.6f} deg")
#         print("-" * 80)

#     if len(results_epnp['t_star']) > 0:
#         print_stats("Pure EPnP (Baseline)", results_epnp)
#         print_stats("SimCC Refinement (Proposed)", results_ref)
        
#         # Improvement Calculation
#         imp_t = (np.mean(results_epnp['t_star']) - np.mean(results_ref['t_star'])) / np.mean(results_epnp['t_star']) * 100
#         imp_q = (np.mean(results_epnp['q_star']) - np.mean(results_ref['q_star'])) / np.mean(results_epnp['q_star']) * 100
        
#         print(f"Improvement (e*): Translation {imp_t:.2f}% | Rotation {imp_q:.2f}%")
#     else:
#         print("No valid results found.")

# if __name__ == '__main__':
#     main()
