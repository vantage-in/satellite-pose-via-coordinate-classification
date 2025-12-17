'''
python tools/evaluate.py --dataset sunlamp --model specc-s
'''
import os
import json
import cv2
import numpy as np
import torch
import argparse
import sys
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

from satellite.refiner import WeightedSimCCPoseRefiner
from configs import project_config as cfg

# ------------------------------------------------------------------------------
# 1. Metrics Logic
# ------------------------------------------------------------------------------

def load_camera_intrinsics(path):
    with open(path) as f: cam = json.load(f)
    return np.array(cam['cameraMatrix'], dtype=np.float32), np.array(cam['distCoeffs'], dtype=np.float32).flatten()

def load_tango_3d_keypoints(path):
    return np.transpose(np.array(loadmat(path)['tango3Dpoints'], dtype=np.float32))

def compute_metrics_and_stars(rvec, tvec, q_gt, t_gt):
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
    if e_t_bar < cfg.THETA_T_NORM:
        e_star_t = 0.0
    else:
        e_star_t = e_t_bar

    if e_q_deg < cfg.THETA_Q_DEG:
        e_star_q_deg = 0.0
        e_star_q_rad = 0.0
    else:
        e_star_q_deg = e_q_deg
        e_star_q_rad = e_q_rad

    # 4. Pose Error (e_pose)
    e_pose_star = e_star_t + e_star_q_rad
    
    return {
        'e_t_m': e_t_m,
        'e_t_bar': e_t_bar,
        'e_q_deg': e_q_deg,
        'e_star_t': e_star_t,
        'e_star_q_deg': e_star_q_deg,
        'e_pose_star': e_pose_star
    }

def log_and_save_results(history, output_file):
    lines = []
    lines.append("="*95)
    lines.append(f"{'EVALUATION METRICS SUMMARY (HIL THRESHOLDS APPLIED)':^95}")
    lines.append("="*95)
    lines.append(f"{'Metric':<25} | {'Pure EPnP':<20} | {'SimCC Refinement':<20} | {'Improvement':<15}")
    lines.append("-" * 95)

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
            
            if mean_epnp != 0:
                imp = (mean_epnp - mean_ref) / mean_epnp * 100
            else:
                imp = 0.0
            
            line = f"{label:<25} | {mean_epnp:.6f}{' '*11} | {mean_ref:.6f}{' '*11} | {imp:6.2f}%"
            lines.append(line)
    else:
        lines.append("No valid results found.")
    
    lines.append("-" * 95)
    lines.append(f"Total Valid Images Evaluated: {len(history['EPnP']['t_m'])}")
    
    for line in lines:
        print(line)
        
    try:
        with open(output_file, 'w') as f:
            f.write("\n".join(lines))
        print(f"\n[Info] Results successfully saved to {output_file}")
    except Exception as e:
        print(f"\n[Error] Failed to save results to file: {e}")

# ------------------------------------------------------------------------------
# 2. Main Loop
# ------------------------------------------------------------------------------

captured_logits = {}
def hook_fn_x(module, input, output): captured_logits['x'] = output.detach().cpu().numpy()
def hook_fn_y(module, input, output): captured_logits['y'] = output.detach().cpu().numpy()

def parse_args():
    parser = argparse.ArgumentParser(description="Satellite Pose Evaluation")
    parser.add_argument('--dataset', type=str, default='lightbox', choices=['lightbox', 'sunlamp'])
    parser.add_argument('--model', type=str, default='specc-s', choices=['specc-s', 'specc-m'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sigma', type=float, default=cfg.DEFAULT_SIGMA)
    parser.add_argument('--output', type=str, default='evaluation_results.txt', help='Output text file path')
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules()
    
    # Config
    dataset_cfg = cfg.DATASETS[args.dataset]
    model_cfg = cfg.MODELS[args.model]
    
    config_file = model_cfg['config_file']
    checkpoint_file = args.checkpoint if args.checkpoint else model_cfg['checkpoint_file']
    model_input_size = model_cfg['input_size']
    
    print(f"Loading Model on {args.device}...")
    model = init_model(config_file, checkpoint_file, device=args.device)
    model.head.cls_x.register_forward_hook(hook_fn_x)
    model.head.cls_y.register_forward_hook(hook_fn_y)

    points_3d = load_tango_3d_keypoints(cfg.MODEL_3D_POINTS_FILE)
    K, dist = load_camera_intrinsics(cfg.CAMERA_FILE)
    
    # Refiner Class
    refiner = WeightedSimCCPoseRefiner(K, dist, model_input_size)

    with open(dataset_cfg['gt_file'], 'r') as f:
        pose_data = json.load(f)
        pose_list = pose_data['images'] if 'images' in pose_data else pose_data
    with open(dataset_cfg['meta_file'], 'r') as f:
        original_images_map = json.load(f).get('original_images', {})
    
    img_root = dataset_cfg['img_root']
    history = {
        'EPnP':    {'t_m': [], 't_bar': [], 'q_deg': [], 'star_t': [], 'star_q': [], 'pose_star': []},
        'Refined': {'t_m': [], 't_bar': [], 'q_deg': [], 'star_t': [], 'star_q': [], 'pose_star': []}
    }
    
    print(f"Starting Evaluation on {args.dataset} ({len(pose_list)} images)...")
    
    for item in tqdm(pose_list):
        raw_filename = item['filename']
        clean_filename = raw_filename[3:] if raw_filename.startswith('img') else raw_filename
        if clean_filename not in original_images_map: continue
        img_meta = original_images_map[clean_filename]
        
        img_path = os.path.join(img_root, clean_filename)
        if not os.path.exists(img_path): img_path = os.path.join(img_root, raw_filename)
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

        # 1. Pure EPnP
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

        # 2. Refinement
        if success:
            logits_x = captured_logits['x'][0]
            logits_y = captured_logits['y'][0]
            
            refiner.prepare_pdf_and_weights(logits_x, logits_y, sigma=args.sigma)
            
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

    log_and_save_results(history, args.output)

if __name__ == '__main__':
    main()