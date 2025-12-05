import os
import json
import cv2
import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
from tqdm import tqdm

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
CONFIG_FILE = 'satellite/rtmpose-m_satellite_f.py'
CHECKPOINT_FILE = '/workspace/rtmpose-m_f/epoch_250.pth'

POSE_GT_FILE = '/workspace/speedplusv2/sunlamp/test.json'
IMAGE_META_FILE = '/workspace/speedplusv2/annotations/test_sunlamp.json'
IMG_ROOT = '/workspace/speedplusv2/sunlamp_preprocessed/'
MODEL_3D_POINTS_FILE = '/workspace/speedplusv2/tangoPoints.mat'
CAMERA_FILE = '/workspace/speedplusv2/camera.json'

# 모델 입력 사이즈 (Config에 맞춰 설정 필요, 보통 256 or 224)
MODEL_INPUT_SIZE = (224, 224) 

# ------------------------------------------------------------------------------
# 2. Weighted PnP Solver Class (핵심 로직)
# ------------------------------------------------------------------------------

class WeightedPoseRefiner:
    def __init__(self, K, dist):
        self.K = K
        self.dist = dist

    def _get_weights(self, logits_x, logits_y):        
        # 가장 높은 확률값(Confidence) 추출
        # 확률이 높을수록(1.0에 가까울수록) 해당 점을 강하게 신뢰함 -> 가중치 높음
        conf_x = np.max(logits_x, axis=1)
        conf_y = np.max(logits_y, axis=1)
        
        return conf_x, conf_y

    def _project(self, rvec, tvec, points_3d):
        """Project 3D points to 2D image plane"""
        img_points, _ = cv2.projectPoints(points_3d, rvec, tvec, self.K, self.dist)
        return img_points.squeeze()

    def _residual_function(self, params, points_3d, points_2d_meas, weights_x, weights_y):
        """
        Cost Function for Least Squares
        Residual = sqrt(Weight) * (Projected - Measured)
        Squared Sum of Residuals = Sum( Weight * (Error)^2 )
        """
        rvec = params[:3]
        tvec = params[3:]
        
        # 1. Projection
        points_2d_proj = self._project(rvec, tvec, points_3d)
        
        # 2. Calculate Error (Pixel distance)
        diff = points_2d_proj - points_2d_meas # (N, 2)
        diff_x = diff[:, 0]
        diff_y = diff[:, 1]
        
        # 3. Apply Weights
        # least_squares는 리턴된 벡터의 제곱의 합을 최소화하므로, sqrt(w)를 곱해줌.
        w_diff_x = np.sqrt(weights_x) * diff_x
        w_diff_y = np.sqrt(weights_y) * diff_y
        
        return np.concatenate([w_diff_x, w_diff_y])

    def solve(self, points_3d, points_2d_meas, logits_x, logits_y):
        """
        Main Pipeline: EPnP Init -> Weighted Refinement
        """
        # 1. 가중치 계산
        w_x, w_y = self._get_weights(logits_x, logits_y)
        
        # 안전장치: 가중치가 0에 수렴하면 나눗셈 에러 등이 날 수 있으므로 clipping
        w_x = np.clip(w_x, 1e-5, 2.0) **2
        w_y = np.clip(w_y, 1e-5, 2.0) **2

        # 2. 초기값 추정 (EPnP) - 가중치 미반영
        # OpenCV 포맷에 맞게 reshape
        p3d = np.ascontiguousarray(points_3d).reshape((-1, 1, 3))
        p2d = np.ascontiguousarray(points_2d_meas).reshape((-1, 1, 2))
        
        success, rvec_init, tvec_init = cv2.solvePnP(
            p3d, p2d, self.K, self.dist, flags=cv2.SOLVEPNP_EPNP
        )
        
        if not success:
            return None, None

        # 3. Weighted Refinement (Levenberg-Marquardt)
        initial_params = np.concatenate([rvec_init.flatten(), tvec_init.flatten()])
        
        res = least_squares(
            self._residual_function,
            initial_params,
            args=(points_3d, points_2d_meas, w_x, w_y),
            method='lm', # Levenberg-Marquardt
            xtol=1e-6,
            ftol=1e-6,
            max_nfev=100
        )
        
        final_rvec = res.x[:3]
        final_tvec = res.x[3:]
        
        return final_rvec, final_tvec

# ------------------------------------------------------------------------------
# 3. Helper Functions (Data Loading)
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

def compute_errors(rvec, tvec, q_gt, t_gt):
    # Rvec -> Quaternion
    R_mat, _ = cv2.Rodrigues(rvec)
    q_pred = R.from_matrix(R_mat).as_quat() # [x, y, z, w]
    q_pred = q_pred[[3, 0, 1, 2]]           # [w, x, y, z] (GT 포맷에 맞춤)

    # Translation Error
    t_err = np.linalg.norm(tvec - t_gt)

    # Rotation Error
    q_pred = q_pred / np.linalg.norm(q_pred)
    q_gt = q_gt / np.linalg.norm(q_gt)
    dot = np.clip(np.abs(np.dot(q_pred, q_gt)), -1.0, 1.0)
    rad_err = 2 * np.arccos(dot)
    deg_err = np.rad2deg(rad_err)
    
    return t_err, deg_err

# ------------------------------------------------------------------------------
# 4. Main Loop
# ------------------------------------------------------------------------------

# 전역 변수로 Logit 캡처용 딕셔너리
captured_logits = {}

def hook_fn_x(module, input, output):
    captured_logits['x'] = output.detach().cpu().numpy()

def hook_fn_y(module, input, output):
    captured_logits['y'] = output.detach().cpu().numpy()

def main():
    register_all_modules()
    
    # 4.1 모델 초기화
    print(f"Loading model from {CONFIG_FILE}...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)
    
    # Hook 등록 (SimCC Head의 출력을 가로챔)
    # 실제 모델 구조에 따라 이름이 다를 수 있으나 보통 head.cls_x 입니다.
    model.head.cls_x.register_forward_hook(hook_fn_x)
    model.head.cls_y.register_forward_hook(hook_fn_y)
    
    # 4.2 데이터 로드
    points_3d = load_tango_3d_keypoints(MODEL_3D_POINTS_FILE)
    K, dist = load_camera_intrinsics(CAMERA_FILE)
    
    # Weighted PnP Solver 인스턴스 생성
    solver = WeightedPoseRefiner(K, dist)
    
    with open(POSE_GT_FILE, 'r') as f:
        pose_data = json.load(f)
        pose_list = pose_data['images'] if 'images' in pose_data else pose_data

    with open(IMAGE_META_FILE, 'r') as f:
        meta_data = json.load(f)
        original_images_map = meta_data.get('original_images', {})
        
    t_errors = []
    q_errors = []
    
    print(f"Starting evaluation on {len(pose_list)} images...")
    
    # 4.3 Evaluation Loop
    for item in tqdm(pose_list):
        # --- 파일명 및 경로 처리 ---
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

        # --- GT Pose 로드 ---
        if 'r_Vo2To_vbs_true' not in item: continue
        t_gt = np.array(item['r_Vo2To_vbs_true'], dtype=np.float32)
        q_gt = np.array(item['q_vbs2tango_true'], dtype=np.float32)

        # --- Inference ---
        # 이 과정에서 hook이 발동하여 captured_logits에 값이 채워짐
        results = inference_topdown(model, img_input)
        pred_instances = results[0].pred_instances
        
        # MMPose가 예측한 2D 좌표 (224x224 Scale)
        kpts_224 = pred_instances.keypoints[0] 
        
        # --- Coordinate Restoration (Original Image Scale) ---
        crop_box = img_meta['crop_box']   # [x1, y1, x2, y2]
        scale_factor = img_meta['scale_factor']
        x1, y1 = crop_box[0], crop_box[1]
        
        # 원본 이미지 좌표계로 변환 (Weighted PnP의 입력)
        kpts_orig = (kpts_224 / scale_factor) + np.array([x1, y1])
        
        # --- Weighted PnP Solver Execution ---
        # Hook에서 가져온 Logit (Batch Dimension 0 제거)
        logits_x = captured_logits['x'][0] 
        logits_y = captured_logits['y'][0]
        
        rvec_pred, tvec_pred = solver.solve(
            points_3d, 
            kpts_orig, 
            logits_x, 
            logits_y
        )
        
        if rvec_pred is not None:
            te, qe = compute_errors(rvec_pred, tvec_pred, q_gt, t_gt)
            t_errors.append(te)
            q_errors.append(qe)
    
    # 4.4 결과 출력
    if len(t_errors) > 0:
        print("\n=== Weighted PnP Evaluation Results ===")
        print(f"Total Images: {len(t_errors)}")
        print(f"Mean Translation Error: {np.mean(t_errors):.6f} m")
        print(f"Mean Rotation Error:    {np.mean(q_errors):.6f} deg")
    else:
        print("No valid results.")

if __name__ == '__main__':
    main()