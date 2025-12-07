import os
import cv2
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
from scipy.special import softmax

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

# 1. MMPose 모듈 등록
register_all_modules()

# --- 전역 변수로 Raw Logit을 저장할 딕셔너리 ---
captured_logits = {}

def hook_fn_x(module, input, output):
    """X축 출력을 가로채는 Hook"""
    captured_logits['x'] = softmax(output.detach().cpu().numpy())

def hook_fn_y(module, input, output):
    """Y축 출력을 가로채는 Hook"""
    captured_logits['y'] = softmax(output.detach().cpu().numpy())

def process_and_plot(ax, logits, color_dot, color_line, axis_name, sigma=1.0):
    """
    Logit 데이터를 처리하여 점(Discrete)과 선(Smoothed+Spline)으로 그리는 함수
    """
    # 1. Discrete Points (이산적인 점)
    x_indices = np.arange(len(logits))
    ax.scatter(x_indices, logits, s=10, color=color_dot, alpha=0.4, label='Raw Discrete')

    # 2. Gaussian Smoothing (약하게 적용)
    smoothed_logits = gaussian_filter1d(logits, sigma=sigma)

    # 3. Cubic Spline Interpolation (부드러운 곡선 생성)
    # 촘촘한 X축 생성 (원본보다 10배 촘촘하게)
    x_dense = np.linspace(0, len(logits) - 1, num=len(logits) * 10)
    
    # 스플라인 함수 생성 및 평가
    cs = CubicSpline(x_indices, smoothed_logits)
    y_dense = cs(x_dense)

    # 4. 곡선 그리기
    ax.plot(x_dense, y_dense, '-', color=color_line, linewidth=2, label=f'Smooth (Sig={sigma}) + Spline')

    # 5. 최대값 표시 (Interpolated Curve 기준)
    max_idx_dense = np.argmax(y_dense)
    max_val = y_dense[max_idx_dense]
    max_loc = x_dense[max_idx_dense] # 소수점 단위 위치

    ax.axvline(x=max_loc, color='red', linestyle='--', alpha=0.6)
    ax.text(max_loc, max_val, f" Peak: {max_val:.2f}", color='red', fontweight='bold')
    
    ax.set_title(f"{axis_name} Distribution")
    ax.set_ylabel("Logit Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

def visualize_logits(config_file, checkpoint_file, img_path, output_dir):
    # 2. 모델 초기화
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing model on {device}...")
    model = init_model(config_file, checkpoint_file, device=device)
    
    # Hook 등록
    handle_x = model.head.cls_x.register_forward_hook(hook_fn_x)
    handle_y = model.head.cls_y.register_forward_hook(hook_fn_y)

    # 3. 이미지 로드 및 추론
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        return

    img = mmcv.imread(img_path)
    h, w, _ = img.shape
    bbox = np.array([0, 0, w, h])
    
    print(f"Running inference on {os.path.basename(img_path)}...")
    results = inference_topdown(model, img, bboxes=bbox[None])
    
    # Hook 제거
    handle_x.remove()
    handle_y.remove()

    # 4. 결과 데이터 추출
    if 'x' not in captured_logits or 'y' not in captured_logits:
        print("Error: Failed to capture logits!")
        return

    raw_x = captured_logits['x'][0] 
    raw_y = captured_logits['y'][0] 
    
    pred_instance = results[0].pred_instances
    pred_scores = pred_instance.keypoint_scores[0]
    keypoint_info = model.dataset_meta.get('keypoint_info', {})

    # 5. 그래프 그리기
    os.makedirs(output_dir, exist_ok=True)
    num_keypoints = raw_x.shape[0]
    
    print(f"Plotting interpolated logits for {num_keypoints} keypoints...")

    for i in range(num_keypoints):
        kpt_name = keypoint_info.get(i, {}).get('name', f'Keypoint {i}')
        score = pred_scores[i]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle(f"ID {i}: {kpt_name} (MM Score: {score:.4f})", fontsize=16)
        
        # X축 Plot (파란색 계열)
        process_and_plot(ax1, raw_x[i], color_dot='cornflowerblue', color_line='blue', axis_name='X-axis', sigma=10.0)

        # Y축 Plot (주황색 계열)
        process_and_plot(ax2, raw_y[i], color_dot='orange', color_line='darkorange', axis_name='Y-axis', sigma=10.0)

        ax2.set_xlabel("Bin Index (SimCC Space)")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        save_path = os.path.join(output_dir, f"logit_spline_kpt_{i:02d}.png")
        plt.savefig(save_path)
        plt.close(fig)

    print(f"Success! Saved 11 graphs to '{output_dir}'")

# --- 실행 설정 ---
if __name__ == '__main__':
    CONFIG_FILE = 'satellite/rtmpose-m_satellite_f.py'
    CHECKPOINT_FILE = '/workspace/rtmpose-m_f/epoch_250.pth'
    
    # 테스트할 이미지 경로 (직접 지정)
    TEST_IMAGE = '/workspace/speedplusv2/sunlamp_preprocessed/001155.jpg' 
    
    OUTPUT_DIR = 'vis_logits_spline/'

    try:
        visualize_logits(CONFIG_FILE, CHECKPOINT_FILE, TEST_IMAGE, OUTPUT_DIR)
    except Exception as e:
        print(f"Critical Error: {e}")
        import traceback
        traceback.print_exc()