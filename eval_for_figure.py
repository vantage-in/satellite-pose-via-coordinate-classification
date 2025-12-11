import os
import json
import cv2
import numpy as np
import torch
import copy
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.special import softmax
from scipy.optimize import minimize

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
from mmpose.registry import VISUALIZERS
from mmpose.structures import PoseDataSample

# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================
TARGET_IDS = ['000001', '000002', '000003', '000004', '000005', '000006', '000007', '000008', '000009', '000010', '000011', '000012', '000013', '000014', '000015', '000016', '000017', '000018', '000019', '000020', '000021', '000022', '000023', '000024', '000025', '000026', '000027', '000028', '000029', '000030', '000031', '000032', '000033', '000034', '000035', '000036', '000037', '000038', '000039', '000040']

# 경로 설정 (사용자 환경에 맞게 수정)
CONFIG_FILE = 'satellite/rtmpose-s_satellite.py'
CHECKPOINT_FILE = '/workspace/rtmpose-s/epoch_280.pth'

DATA_NAME = 'lightbox'  # 'lightbox' or 'sunlamp'
PREPROCESSED_ROOT = f'/workspace/speedplusv2/{DATA_NAME}_preprocessed/'
GT_POSE_FILE = f'/workspace/speedplusv2/{DATA_NAME}/test.json'
IMAGE_META_FILE = f'/workspace/speedplusv2/annotations/test_{DATA_NAME}.json'
CAMERA_FILE = '/workspace/speedplusv2/camera.json'
MODEL_3D_POINTS_FILE = '/workspace/speedplusv2/tangoPoints.mat'

OUTPUT_DIR = f'figure2/vis_separate_{DATA_NAME}'
METRIC_LOG_FILE = os.path.join(OUTPUT_DIR, 'metrics_report.txt')

MODEL_INPUT_SIZE = (224, 224)
THETA_T_NORM = 2.173e-3
THETA_Q_DEG = 0.169

# ==============================================================================
# 2. HELPER CLASSES (Refiner & Metrics) - 기존과 동일
# ==============================================================================

class WeightedSimCCPoseRefiner:
    def __init__(self, K, dist, model_input_size):
        self.K = K
        self.dist = dist
        self.w_model, self.h_model = model_input_size
        self.splines_x, self.splines_y, self.weights = [], [], []
        self.simcc_scale_x = 1.0
        self.simcc_scale_y = 1.0

    def prepare_pdf_and_weights(self, logits_x, logits_y, sigma=2.0):
        self.splines_x, self.splines_y, self.weights = [], [], []
        n_kpts, n_bins_x = logits_x.shape
        _, n_bins_y = logits_y.shape
        
        probs_x_all = softmax(logits_x, axis=1)
        probs_y_all = softmax(logits_y, axis=1)
        conf_x, conf_y = np.max(logits_x, axis=1), np.max(logits_y, axis=1)
        idx_x, idx_y = np.argmax(probs_x_all, axis=1), np.argmax(probs_y_all, axis=1)

        raw_weights = np.minimum(conf_x, conf_y)
        is_boundary = (idx_x <= 5) | (idx_x >= n_bins_x - 6) | (idx_y <= 5) | (idx_y >= n_bins_y - 6)
        raw_weights[is_boundary] = 0.0
        self.weights = raw_weights
        
        self.simcc_scale_x = n_bins_x / self.w_model
        self.simcc_scale_y = n_bins_y / self.h_model
        epsilon = 1e-6 

        for i in range(n_kpts):
            prob_x = gaussian_filter1d(probs_x_all[i], sigma=sigma)
            prob_y = gaussian_filter1d(probs_y_all[i], sigma=sigma)
            x_axis, y_axis = np.arange(n_bins_x), np.arange(n_bins_y)
            self.splines_x.append(CubicSpline(x_axis, np.log(np.clip(prob_x, epsilon, 1.0))))
            self.splines_y.append(CubicSpline(y_axis, np.log(np.clip(prob_y, epsilon, 1.0))))

    def objective_function(self, params, points_3d, crop_meta):
        rvec, tvec = params[:3], params[3:]
        img_points, _ = cv2.projectPoints(points_3d, rvec, tvec, self.K, self.dist)
        img_points = img_points.squeeze()
        x1, y1 = crop_meta['crop_box'][:2]
        scale = crop_meta['scale_factor']
        model_pts = (img_points - np.array([x1, y1])) * scale
        
        weighted_nll_sum = 0.0
        for i in range(len(self.splines_x)):
            if self.weights[i] < 1e-6: continue 
            ux, uy = model_pts[i, 0] * self.simcc_scale_x, model_pts[i, 1] * self.simcc_scale_y
            val_x = self.splines_x[i](ux) if (0 <= ux < len(self.splines_x[0].x)) else -100.0
            val_y = self.splines_y[i](uy) if (0 <= uy < len(self.splines_y[0].x)) else -100.0
            weighted_nll_sum -= self.weights[i] * (val_x + val_y)
        return weighted_nll_sum

    def refine(self, rvec_init, tvec_init, points_3d, crop_meta):
        initial = np.concatenate([rvec_init.flatten(), tvec_init.flatten()])
        res = minimize(self.objective_function, initial, args=(points_3d, crop_meta),
                       method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'disp': False})
        return res.x[:3], res.x[3:]

def load_camera_intrinsics(path):
    with open(path) as f: cam = json.load(f)
    return np.array(cam['cameraMatrix'], dtype=np.float32), np.array(cam['distCoeffs'], dtype=np.float32).flatten()

def load_tango_3d_keypoints(path):
    return np.transpose(np.array(loadmat(path)['tango3Dpoints'], dtype=np.float32))

def compute_metrics(rvec, tvec, q_gt, t_gt):
    R_mat, _ = cv2.Rodrigues(rvec)
    q_pred = R.from_matrix(R_mat).as_quat()[[3, 0, 1, 2]]
    q_pred /= np.linalg.norm(q_pred)
    q_gt /= np.linalg.norm(q_gt)
    e_q_deg = np.rad2deg(2 * np.arccos(np.clip(np.abs(np.dot(q_pred, q_gt)), -1.0, 1.0)))
    e_t_bar = np.linalg.norm(tvec - t_gt) / np.linalg.norm(t_gt)
    e_star_t = 0.0 if e_t_bar < THETA_T_NORM else e_t_bar
    e_star_q = 0.0 if e_q_deg < THETA_Q_DEG else np.deg2rad(e_q_deg)
    return e_t_bar, e_q_deg, e_star_t + e_star_q

def project_back_to_crop(rvec, tvec, K, dist, points_3d, crop_meta):
    img_points_full, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist)
    x1, y1 = crop_meta['crop_box'][:2]
    return (img_points_full.squeeze() - np.array([x1, y1])) * crop_meta['scale_factor']

# Hook
captured_logits = {}
def hook_fn_x(m, i, o): captured_logits['x'] = o.detach().cpu().numpy()
def hook_fn_y(m, i, o): captured_logits['y'] = o.detach().cpu().numpy()

# ==============================================================================
# 3. MAIN LOGIC WITH MMPOSE VISUALIZER
# ==============================================================================

def main():
    register_all_modules()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Loading Model on {device}...")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=device)
    model.head.cls_x.register_forward_hook(hook_fn_x)
    model.head.cls_y.register_forward_hook(hook_fn_y)

    # Visualizer 초기화 (inference.py 스타일)
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    points_3d = load_tango_3d_keypoints(MODEL_3D_POINTS_FILE)
    K, dist = load_camera_intrinsics(CAMERA_FILE)
    refiner = WeightedSimCCPoseRefiner(K, dist, MODEL_INPUT_SIZE)

    # 메타데이터 로드
    with open(GT_POSE_FILE, 'r') as f:
        pose_data = json.load(f)
        pose_list = pose_data['images'] if 'images' in pose_data else pose_data
        pose_map = {item['filename'].split('.')[0]: item for item in pose_list}
        pose_map.update({item['filename']: item for item in pose_list})

    with open(IMAGE_META_FILE, 'r') as f:
        original_images_map = json.load(f).get('original_images', {})

    with open(METRIC_LOG_FILE, 'w') as f:
        f.write(f"{'ID':<15} | {'Method':<10} | {'e_t_bar':<10} | {'e_q(deg)':<10} | {'e_pose*':<10}\n")
        f.write("-" * 70 + "\n")

    print(f"Processing {len(TARGET_IDS)} target images...")

    for target_id in TARGET_IDS:
        # --- 1. Data Loading ---
        meta_key = next((k for k in original_images_map if target_id in k), None)
        if not meta_key: 
            continue
        
        gt_item = pose_map.get(meta_key, pose_map.get('img' + target_id + '.jpg'))
        if not gt_item: 
            continue
        
        img_path = os.path.join(PREPROCESSED_ROOT, f"{target_id}.jpg")
        if not os.path.exists(img_path):
             # jpg가 없으면 폴더 검색 (확장자 유연성)
            cands = [f for f in os.listdir(PREPROCESSED_ROOT) if target_id in f]
            if cands: img_path = os.path.join(PREPROCESSED_ROOT, cands[0])
            else: continue
            
        img_vis = cv2.imread(img_path) # BGR
        img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB) # Visualizer는 RGB 권장

        # --- 2. Inference (Raw) ---
        results = inference_topdown(model, img_vis)
        raw_sample = results[0] # PoseDataSample
        kpts_raw = raw_sample.pred_instances.keypoints[0] # (11, 2)

        # [Visualization 1] Raw Network Prediction
        out_raw = os.path.join(OUTPUT_DIR, f"vis_{target_id}_01_Raw.jpg")
        visualizer.add_datasample(
            'raw', img_rgb, data_sample=raw_sample,
            draw_gt=False, draw_heatmap=False, draw_bbox=True,
            show_kpt_idx=True, skeleton_style='mmpose',
            show=False, out_file=out_raw, kpt_thr=0.3
        )

        # --- 3. EPnP ---
        img_meta = original_images_map[meta_key]
        crop_meta = {'crop_box': img_meta['crop_box'], 'scale_factor': img_meta['scale_factor']}
        x1, y1 = crop_meta['crop_box'][:2]
        
        kpts_orig = (kpts_raw / crop_meta['scale_factor']) + np.array([x1, y1])
        t_gt = np.array(gt_item['r_Vo2To_vbs_true'], dtype=np.float32)
        q_gt = np.array(gt_item['q_vbs2tango_true'], dtype=np.float32)

        success, rvec_epnp, tvec_epnp = cv2.solvePnP(
            np.ascontiguousarray(points_3d).reshape((-1, 1, 3)),
            np.ascontiguousarray(kpts_orig).reshape((-1, 1, 2)),
            K, dist, flags=cv2.SOLVEPNP_EPNP
        )

        if success:
            tvec_epnp = np.squeeze(tvec_epnp)
            # Re-project for Visualization
            kpts_epnp_crop = project_back_to_crop(rvec_epnp, tvec_epnp, K, dist, points_3d, crop_meta)
            
            # [Visualization 2] EPnP Result
            # 기존 data_sample을 복제한 뒤 keypoints만 EPnP 결과로 교체
            epnp_sample = copy.deepcopy(raw_sample)
            epnp_sample.pred_instances.keypoints[0] = kpts_epnp_crop
            
            out_epnp = os.path.join(OUTPUT_DIR, f"vis_{target_id}_02_EPnP.jpg")
            visualizer.add_datasample(
                'epnp', img_rgb, data_sample=epnp_sample,
                draw_gt=False, draw_heatmap=False, draw_bbox=True,
                show_kpt_idx=True, skeleton_style='mmpose',
                show=False, out_file=out_epnp, kpt_thr=0.3
            )
            
            m_epnp = compute_metrics(rvec_epnp, tvec_epnp, q_gt, t_gt)

            # --- 4. Refinement ---
            logits_x, logits_y = captured_logits['x'][0], captured_logits['y'][0]
            refiner.prepare_pdf_and_weights(logits_x, logits_y, sigma=10.0)
            rvec_ref, tvec_ref = refiner.refine(rvec_epnp, tvec_epnp, points_3d, crop_meta)
            
            # Re-project for Visualization
            kpts_ref_crop = project_back_to_crop(rvec_ref, tvec_ref, K, dist, points_3d, crop_meta)
            
            # [Visualization 3] Refined Result
            ref_sample = copy.deepcopy(raw_sample)
            ref_sample.pred_instances.keypoints[0] = kpts_ref_crop
            
            out_ref = os.path.join(OUTPUT_DIR, f"vis_{target_id}_03_Refined.jpg")
            visualizer.add_datasample(
                'refined', img_rgb, data_sample=ref_sample,
                draw_gt=False, draw_heatmap=False, draw_bbox=True,
                show_kpt_idx=True, skeleton_style='mmpose',
                show=False, out_file=out_ref, kpt_thr=0.3
            )

            m_ref = compute_metrics(rvec_ref, tvec_ref, q_gt, t_gt)
            
            # Log Metrics
            with open(METRIC_LOG_FILE, 'a') as f:
                f.write(f"{target_id:<15} | {'EPnP':<10} | {m_epnp[0]:.4f}     | {m_epnp[1]:.4f}     | {m_epnp[2]:.4f}\n")
                f.write(f"{'':<15} | {'Refined':<10} | {m_ref[0]:.4f}     | {m_ref[1]:.4f}     | {m_ref[2]:.4f}\n")
                f.write("-" * 70 + "\n")
            
            print(f"[{target_id}] Saved 3 images. Improvement: {m_epnp[2] - m_ref[2]:.4f}")

        else:
            print(f"[{target_id}] EPnP Failed.")

    print(f"\nCompleted. Results in {OUTPUT_DIR}")

if __name__ == '__main__':
    main()