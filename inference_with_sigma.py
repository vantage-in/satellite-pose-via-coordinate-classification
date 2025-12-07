import os
import random
import cv2
import mmcv
import torch
import numpy as np
from pathlib import Path

from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
from mmpose.registry import VISUALIZERS

# 1. MMPose 모듈 등록
register_all_modules()

def calculate_uncertainty(pred_instances, simcc_split_ratio=2.0):
    """
    SimCC 1D 분포로부터 불확실성(표준편차)을 계산합니다.
    (배경 노이즈 제거 로직 추가됨)
    """
    if 'keypoint_x_labels' not in pred_instances:
        return None

    x_labels = pred_instances.keypoint_x_labels
    y_labels = pred_instances.keypoint_y_labels
    if x_labels.ndim == 3:
        x_labels = x_labels[0]
        y_labels = y_labels[0]

    dist_x = torch.from_numpy(x_labels).float()
    dist_y = torch.from_numpy(y_labels).float()

    # -------------------------------------------------------------------------
    # [핵심 수정] 노이즈 필터링 (Noise Suppression)
    # 배경에 깔린 작은 값들이 분산을 키우는 것을 방지합니다.
    # 각 키포인트별로 '최대값의 50%'보다 작은 값은 모두 0으로 만듭니다.
    # -------------------------------------------------------------------------
    thr_ratio = 0.5
    
    # (K, 1) 형태의 임계값 생성
    thr_x = dist_x.max(dim=1, keepdim=True).values * thr_ratio
    thr_y = dist_y.max(dim=1, keepdim=True).values * thr_ratio
    
    # 임계값 이하 제거 (In-place clamping)
    dist_x[dist_x < thr_x] = 0
    dist_y[dist_y < thr_y] = 0

    # -------------------------------------------------------------------------
    # 정규화 (L1 Normalize) -> 합이 1인 확률 분포(PDF)로 변환
    # -------------------------------------------------------------------------
    prob_x = dist_x / (dist_x.sum(dim=1, keepdim=True) + 1e-6)
    prob_y = dist_y / (dist_y.sum(dim=1, keepdim=True) + 1e-6)

    # 좌표 Grid 생성
    K, Wx = prob_x.shape
    _, Wy = prob_y.shape
    grid_x = torch.arange(Wx, dtype=torch.float32).expand(K, -1)
    grid_y = torch.arange(Wy, dtype=torch.float32).expand(K, -1)
    
    # 평균(Mean) 계산
    mu_x = (prob_x * grid_x).sum(dim=1)
    mu_y = (prob_y * grid_y).sum(dim=1)
    
    # 분산(Variance) 계산: E[x^2] - E[x]^2 대신 sum(p * (x-mu)^2) 사용 (안정적)
    var_x = (prob_x * (grid_x - mu_x.unsqueeze(1))**2).sum(dim=1)
    var_y = (prob_y * (grid_y - mu_y.unsqueeze(1))**2).sum(dim=1)
    
    # 표준편차(Sigma) 계산 및 Scale 복원 (simcc_split_ratio 고려)
    sigma_x = torch.sqrt(var_x) / simcc_split_ratio
    sigma_y = torch.sqrt(var_y) / simcc_split_ratio
    
    sigmas = torch.stack([sigma_x, sigma_y], dim=1).numpy()
    
    return sigmas

def draw_text_side_panel(img, keypoints, sigmas, scores, keypoint_info):
    """
    이미지 우측에 2열로 텍스트 패널을 생성합니다.
    """
    h, w, _ = img.shape
    # 패널 너비를 조금 더 넓힘 (2열 배치용)
    panel_w = 550 
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8) # 검은색 배경
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4  # 폰트 크기 약간 축소
    line_type = 1
    line_gap = 20     # 줄 간격
    start_y = 25
    
    # 2열 배치를 위한 설정
    col_break_idx = 6 # 6번째 키포인트부터 오른쪽 열로
    col2_x_offset = 280

    # Header
    cv2.putText(panel, "ID:(X,Y)|Sig", (10, 15), font, font_scale, (0, 255, 255), 1)
    cv2.putText(panel, "ID:(X,Y)|Sig", (10 + col2_x_offset, 15), font, font_scale, (0, 255, 255), 1)

    for i, (kpt, score) in enumerate(zip(keypoints, scores)):
        kpt_name = keypoint_info.get(i, {}).get('name', f'{i}')
        
        x, y = kpt
        sx, sy = sigmas[i] if sigmas is not None else (0.0, 0.0)
        
        # 좌표: 정수, Sigma: 소수점 2자리
        text_coords = f"({x:.0f},{y:.0f})"
        text_sigma = f"{sx:.2f},{sy:.2f}" 
        
        # 이름이 길면 자르기
        disp_name = kpt_name[:7]
        text_line = f"{i}:{disp_name} {text_coords}|{text_sigma}"
        
        # 신뢰도 낮은 점은 회색 처리
        color = (255, 255, 255) if score > 0.3 else (100, 100, 100)
        
        if i < col_break_idx:
            # 1열 (왼쪽)
            pos = (10, start_y + i * line_gap)
        else:
            # 2열 (오른쪽)
            pos = (10 + col2_x_offset, start_y + (i - col_break_idx) * line_gap)
            
        cv2.putText(panel, text_line, pos, font, font_scale, color, line_type)

    combined = np.hstack((img, panel))
    return combined

def run_inference_on_image(model, img_path, out_dir, visualizer):
    img = mmcv.imread(img_path)
    h, w, _ = img.shape
    bbox_xyxy = np.array([0, 0, w, h])

    results = inference_topdown(model, img, bboxes=bbox_xyxy[None])
    pred_instance = results[0].pred_instances
    
    keypoints = pred_instance.keypoints[0]
    scores = pred_instance.keypoint_scores[0]

    # 1. Sigma 계산
    simcc_split_ratio = getattr(model.cfg.model.head, 'simcc_split_ratio', 2.0)
    sigmas = calculate_uncertainty(pred_instance, simcc_split_ratio)
    
    # 2. 스켈레톤 그리기
    visualizer.add_datasample(
        'result',
        img,
        data_sample=results[0],
        draw_gt=False,
        draw_heatmap=False,
        draw_bbox=False,
        show_kpt_idx=True,
        skeleton_style='mmpose',
        show=False,
        out_file=None,
        kpt_thr=0.3
    )
    res_img = visualizer.get_image()

    # 3. 텍스트 패널 (2열) 추가
    keypoint_info = model.dataset_meta.get('keypoint_info', {})
    final_img = draw_text_side_panel(res_img, keypoints, sigmas, scores, keypoint_info)

    # 저장
    img_name = os.path.basename(img_path)
    out_path = os.path.join(out_dir, f"vis_{img_name}")
    cv2.imwrite(out_path, final_img)

    return keypoints, scores, out_path

def process_folder(config_file, checkpoint_file, input_folder, output_folder, sample_num=5):
    device = 'cpu'# 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing model on {device}...")
    model = init_model(config_file, checkpoint_file, device=device)
    
    # [중요] Heatmap 출력 활성화
    model.cfg.test_cfg['output_heatmaps'] = True
    
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    visualizer.set_dataset_meta(model.dataset_meta)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_images = [
        os.path.join(input_folder, f) 
        for f in os.listdir(input_folder) 
        if f.lower().endswith(valid_extensions)
    ]

    if not all_images:
        print(f"No images found in {input_folder}")
        return

    num_to_sample = min(len(all_images), sample_num)
    sampled_images = random.sample(all_images, num_to_sample)
    
    print(f"\nProcessing {num_to_sample} images...")
    os.makedirs(output_folder, exist_ok=True)

    for i, img_path in enumerate(sampled_images):
        print(f"[{i+1}/{num_to_sample}] {os.path.basename(img_path)}")
        try:
            run_inference_on_image(model, img_path, output_folder, visualizer)
        except Exception as e:
            print(f"   -> Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    # 경로 설정
    CONFIG_FILE = 'satellite/rtmpose-m_satellite_f.py'
    CHECKPOINT_FILE = '/workspace/rtmpose-m_f/epoch_420.pth'
    INPUT_FOLDER = '/workspace/speedplusv2/sunlamp_preprocessed/'
    OUTPUT_FOLDER = 'vis_results_sigma_sunlamp/'
    SAMPLE_NUM = 50

    try:
        process_folder(CONFIG_FILE, CHECKPOINT_FILE, INPUT_FOLDER, OUTPUT_FOLDER, SAMPLE_NUM)
    except Exception as e:
        print(f"Critical Error: {e}")