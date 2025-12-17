'''
python tools/visualize.py --images 0000{01..05}.jpg --dataset sunlamp
'''
import os
import json
import cv2
import numpy as np
import torch
import copy
import argparse
import sys
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R

from mmengine.config import Config
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
from mmpose.registry import VISUALIZERS

from satellite.refiner import WeightedSimCCPoseRefiner
from configs import project_config as cfg

# ==============================================================================
# 1. HELPER FUNCTIONS
# ==============================================================================

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
    
    # Rotation Error (Degree)
    dot = np.clip(np.abs(np.dot(q_pred, q_gt)), -1.0, 1.0)
    e_q_deg = np.rad2deg(2 * np.arccos(dot))
    
    # Translation Error (Normalized)
    e_t_bar = np.linalg.norm(tvec - t_gt) / np.linalg.norm(t_gt)
    
    # HIL Thresholds from Config
    e_star_t = 0.0 if e_t_bar < cfg.THETA_T_NORM else e_t_bar
    e_star_q = 0.0 if e_q_deg < cfg.THETA_Q_DEG else np.deg2rad(e_q_deg)
    
    return e_t_bar, e_q_deg, e_star_t + e_star_q

def project_back_to_crop(rvec, tvec, K, dist, points_3d, crop_meta):
    img_points_full, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist)
    x1, y1 = crop_meta['crop_box'][:2]
    return (img_points_full.squeeze() - np.array([x1, y1])) * crop_meta['scale_factor']

# Hook Functions
captured_logits = {}
def hook_fn_x(m, i, o): captured_logits['x'] = o.detach().cpu().numpy()
def hook_fn_y(m, i, o): captured_logits['y'] = o.detach().cpu().numpy()

# ==============================================================================
# 2. MAIN LOGIC
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Satellite Pose Visualization & Evaluation")
    
    # Target Images
    parser.add_argument('--images', nargs='+', required=True, 
                        help='List of image filenames (e.g. 000001.jpg 000002.jpg)')
    
    # Settings
    parser.add_argument('--dataset', type=str, default='lightbox', choices=['lightbox', 'sunlamp'])
    parser.add_argument('--model', type=str, default='specc-s', choices=['specc-s', 'specc-m'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sigma', type=float, default=cfg.DEFAULT_SIGMA)
    parser.add_argument('--output_dir', type=str, default='vis_results_figure', 
                        help='Directory to save results')
    
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules()
    
    # 1. Config Loading
    dataset_cfg = cfg.DATASETS[args.dataset]
    model_cfg = cfg.MODELS[args.model]
    
    config_file = model_cfg['config_file']
    checkpoint_file = args.checkpoint if args.checkpoint else model_cfg['checkpoint_file']
    model_input_size = model_cfg['input_size']
    
    output_dir = f"{args.output_dir}_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    metric_log_file = os.path.join(output_dir, 'metrics_report.txt')

    print(f"Loading Model on {args.device}...")
    
    model_config = Config.fromfile(config_file)
    if hasattr(model_config, 'custom_imports'):
        model_config.custom_imports = None
        
    model = init_model(model_config, checkpoint_file, device=args.device)
    model.head.cls_x.register_forward_hook(hook_fn_x)
    model.head.cls_y.register_forward_hook(hook_fn_y)

    # Visualizer Setup
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    # Load 3D Points & Camera
    points_3d = load_tango_3d_keypoints(cfg.MODEL_3D_POINTS_FILE)
    K, dist = load_camera_intrinsics(cfg.CAMERA_FILE)
    refiner = WeightedSimCCPoseRefiner(K, dist, model_input_size)

    # Load Meta Data
    with open(dataset_cfg['gt_file'], 'r') as f:
        pose_data = json.load(f)
        pose_list = pose_data['images'] if 'images' in pose_data else pose_data
        
        pose_map = {}
        for item in pose_list:
            raw_name = item['filename']
            clean_name = raw_name[3:] if raw_name.startswith('img') else raw_name
            pose_map[raw_name] = item
            pose_map[clean_name] = item

    with open(dataset_cfg['meta_file'], 'r') as f:
        original_images_map = json.load(f).get('original_images', {})

    # Initialize Log File
    with open(metric_log_file, 'w') as f:
        f.write(f"{'Filename':<20} | {'Method':<10} | {'e_t_bar':<10} | {'e_q(deg)':<10} | {'e_pose*':<10}\n")
        f.write("-" * 75 + "\n")

    print(f"Processing {len(args.images)} target images...")

    for target_filename in args.images:
        clean_target = os.path.basename(target_filename)
        
        gt_item = pose_map.get(clean_target)
        
        if not gt_item:
            if not clean_target.startswith('img'):
                gt_item = pose_map.get('img' + clean_target)
            else:
                gt_item = pose_map.get(clean_target[3:])
        
        if not gt_item:
            print(f"[Skip] GT not found for {target_filename}")
            continue
            
        # Metadata
        raw_gt_name = gt_item['filename']
        meta_key = raw_gt_name[3:] if raw_gt_name.startswith('img') else raw_gt_name
        
        if meta_key not in original_images_map:
            print(f"[Skip] Meta not found for {meta_key}")
            continue

        # Load
        img_path = os.path.join(dataset_cfg['img_root'], meta_key)
        if not os.path.exists(img_path):
            img_path = os.path.join(dataset_cfg['img_root'], raw_gt_name)
            
        if not os.path.exists(img_path):
            print(f"[Skip] Image file not found: {img_path}")
            continue

        img_vis = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)

        # Inference 
        results = inference_topdown(model, img_vis)
        raw_sample = results[0]
        kpts_raw = raw_sample.pred_instances.keypoints[0]

        # [Visualization 1] Raw Network Prediction
        out_raw = os.path.join(output_dir, f"vis_{meta_key}_01_Raw.jpg")
        visualizer.add_datasample(
            'raw', img_rgb, data_sample=raw_sample,
            draw_gt=False, draw_heatmap=False, draw_bbox=True,
            show_kpt_idx=True, skeleton_style='mmpose',
            show=False, out_file=out_raw, kpt_thr=0.3
        )

        # EPnP
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
            epnp_sample = copy.deepcopy(raw_sample)
            epnp_sample.pred_instances.keypoints[0] = kpts_epnp_crop
            
            out_epnp = os.path.join(output_dir, f"vis_{meta_key}_02_EPnP.jpg")
            visualizer.add_datasample(
                'epnp', img_rgb, data_sample=epnp_sample,
                draw_gt=False, draw_heatmap=False, draw_bbox=True,
                show_kpt_idx=True, skeleton_style='mmpose',
                show=False, out_file=out_epnp, kpt_thr=0.3
            )
            
            m_epnp = compute_metrics(rvec_epnp, tvec_epnp, q_gt, t_gt)

            # Refinement
            logits_x, logits_y = captured_logits['x'][0], captured_logits['y'][0]
            refiner.prepare_pdf_and_weights(logits_x, logits_y, sigma=args.sigma)
            rvec_ref, tvec_ref = refiner.refine(rvec_epnp, tvec_epnp, points_3d, crop_meta)
            
            # Re-project for Visualization
            kpts_ref_crop = project_back_to_crop(rvec_ref, tvec_ref, K, dist, points_3d, crop_meta)
            
            # [Visualization 3] Refined Result
            ref_sample = copy.deepcopy(raw_sample)
            ref_sample.pred_instances.keypoints[0] = kpts_ref_crop
            
            out_ref = os.path.join(output_dir, f"vis_{meta_key}_03_Refined.jpg")
            visualizer.add_datasample(
                'refined', img_rgb, data_sample=ref_sample,
                draw_gt=False, draw_heatmap=False, draw_bbox=True,
                show_kpt_idx=True, skeleton_style='mmpose',
                show=False, out_file=out_ref, kpt_thr=0.3
            )

            m_ref = compute_metrics(rvec_ref, tvec_ref, q_gt, t_gt)
            
            # Log Metrics
            with open(metric_log_file, 'a') as f:
                f.write(f"{meta_key:<20} | {'EPnP':<10} | {m_epnp[0]:.4f}     | {m_epnp[1]:.4f}     | {m_epnp[2]:.4f}\n")
                f.write(f"{'':<20} | {'Refined':<10} | {m_ref[0]:.4f}     | {m_ref[1]:.4f}     | {m_ref[2]:.4f}\n")
                f.write("-" * 75 + "\n")
            
            imp = m_epnp[2] - m_ref[2]
            print(f"[{meta_key}] Saved. e* Improvement: {imp:.6f} ({'Improved' if imp > 0 else 'Worse'})")

        else:
            print(f"[{meta_key}] EPnP Failed.")

    print(f"\nAll Done. Check '{output_dir}' for images and report.")

if __name__ == '__main__':
    main()