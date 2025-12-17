'''
python tools/verification/plot_logits.py --image /workspace/speedplusv2/sunlamp_preprocessed/001155.jpg  --model specc-s --sigma 10.0
'''
import os
import cv2
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.special import softmax

from mmengine.config import Config
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

from configs import project_config as cfg

# ------------------------------------------------------------------------------
# 1. Global Hooks & Helper Functions
# ------------------------------------------------------------------------------

captured_logits = {}

def hook_fn_x(module, input, output):
    """Hook capturing X-axis outputs"""
    captured_logits['x'] = softmax(output.detach().cpu().numpy(), axis=1)

def hook_fn_y(module, input, output):
    """Hook capturing Y-axis outputs"""
    captured_logits['y'] = softmax(output.detach().cpu().numpy(), axis=1)

def process_and_plot(ax, logits, color_dot, color_line, axis_name, sigma=1.0):
    """
    Plot discrete and smoothed logit data
    """
    # Discrete Points
    x_indices = np.arange(len(logits))
    ax.scatter(x_indices, logits, s=10, color=color_dot, alpha=0.4, label='Raw Discrete')

    # Gaussian Smoothing
    smoothed_logits = gaussian_filter1d(logits, sigma=sigma)

    # Cubic Spline Interpolation
    x_dense = np.linspace(0, len(logits) - 1, num=len(logits) * 10)
    
    cs = CubicSpline(x_indices, smoothed_logits)
    y_dense = cs(x_dense)

    ax.plot(x_dense, y_dense, '-', color=color_line, linewidth=2, label=f'Smooth (Sig={sigma}) + Spline')

    # Display max indices
    max_idx_dense = np.argmax(y_dense)
    max_val = y_dense[max_idx_dense]
    max_loc = x_dense[max_idx_dense] 

    ax.axvline(x=max_loc, color='red', linestyle='--', alpha=0.6)
    ax.text(max_loc, max_val, f" Peak: {max_val:.2f}", color='red', fontweight='bold')
    
    ax.set_title(f"{axis_name} Distribution")
    ax.set_ylabel("Logit Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

# ------------------------------------------------------------------------------
# 2. Main Logic
# ------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SimCC Logits for a Single Image")
    
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to the target image file')
    
    parser.add_argument('--model', type=str, default='specc-s', choices=['specc-s', 'specc-m'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sigma', type=float, default=10.0, help='Sigma for Gaussian smoothing in visualization')
    parser.add_argument('--output_dir', type=str, default='vis_logits_spline', help='Output directory for plots')
    
    return parser.parse_args()

def main():
    args = parse_args()
    register_all_modules()
    
    model_cfg = cfg.MODELS[args.model]
    config_file = model_cfg['config_file']
    checkpoint_file = args.checkpoint if args.checkpoint else model_cfg['checkpoint_file']
    
    print(f"Loading Model on {args.device}...")
    
    model_config = Config.fromfile(config_file)
    if hasattr(model_config, 'custom_imports'):
        model_config.custom_imports = None
        
    model = init_model(model_config, checkpoint_file, device=args.device)
    
    # Hook 
    handle_x = model.head.cls_x.register_forward_hook(hook_fn_x)
    handle_y = model.head.cls_y.register_forward_hook(hook_fn_y)

    # Image Load & Inference
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
        return

    img = mmcv.imread(args.image)
    h, w, _ = img.shape
    bbox = np.array([0, 0, w, h]) 
    
    print(f"Running inference on {os.path.basename(args.image)}...")
    results = inference_topdown(model, img, bboxes=bbox[None])
    
    # Resource cleanup
    handle_x.remove()
    handle_y.remove()

    # Data Extraction
    if 'x' not in captured_logits or 'y' not in captured_logits:
        print("Error: Failed to capture logits!")
        return

    raw_x = captured_logits['x'][0] 
    raw_y = captured_logits['y'][0] 
    
    pred_instance = results[0].pred_instances
    pred_scores = pred_instance.keypoint_scores[0]
    keypoint_info = model.dataset_meta.get('keypoint_info', {})

    # Plotting
    os.makedirs(args.output_dir, exist_ok=True)
    num_keypoints = raw_x.shape[0]
    
    print(f"Plotting interpolated logits for {num_keypoints} keypoints...")

    for i in range(num_keypoints):
        kpt_name = keypoint_info.get(i, {}).get('name', f'Keypoint {i}')
        score = pred_scores[i]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f"ID {i}: {kpt_name} (MM Score: {score:.4f})", fontsize=16)
        
        # X-axis Plot
        process_and_plot(ax1, raw_x[i], color_dot='cornflowerblue', color_line='blue', 
                         axis_name='X-axis', sigma=args.sigma)

        # Y-axis Plot
        process_and_plot(ax2, raw_y[i], color_dot='orange', color_line='darkorange', 
                         axis_name='Y-axis', sigma=args.sigma)

        ax2.set_xlabel("Bin Index (SimCC Space)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        base_name = os.path.splitext(os.path.basename(args.image))[0]
        save_path = os.path.join(args.output_dir, f"{base_name}_kpt_{i:02d}.png")
        
        plt.savefig(save_path)
        plt.close(fig)

    print(f"Success! Saved {num_keypoints} graphs to '{args.output_dir}'")

if __name__ == '__main__':
    main()