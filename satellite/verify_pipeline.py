# verify_pipeline.py

import argparse
import os
import os.path as osp
import torch
import mmcv
import numpy as np
import cv2
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.registry import DATASETS, MODELS, TRANSFORMS
from mmcv.transforms import Compose
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# [사용자 설정] 검증하고 싶은 이미지 ID 리스트 (확장자 제외, 파일명만)
# SPEED+ 데이터셋 폴더 내에 실제로 존재하는 파일명이어야 합니다.
# -----------------------------------------------------------
TARGET_IDS = ['000021', '000012', '000013', '000014', '000015', '000016', '000017', '000018', '000019', '0000020', '000022'] 

def parse_args():
    parser = argparse.ArgumentParser(description='Verify Pipeline Step-by-Step')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--output-dir', default='pipeline_verification', help='output directory')
    parser.add_argument('--device', default='cuda', help='device to run augmentation')
    args = parser.parse_args()
    return args

def denormalize(tensor, mean, std):
    """Normalize된 Tensor를 이미지(0-255, RGB)로 변환"""
    tensor = tensor.detach().cpu().clone()
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0) # (1, C, H, W) -> (C, H, W)
    
    tensor = tensor.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
    
    mean = torch.tensor(mean).view(1, 1, 3)
    std = torch.tensor(std).view(1, 1, 3)
    
    img = tensor * std + mean
    img = img.clamp(0, 255).numpy().astype(np.uint8)
    return img

def save_image(img, folder, filename):
    """이미지 저장 헬퍼 (RGB -> BGR for cv2 saving if needed, or use mmcv)"""
    # mmcv.imwrite는 BGR을 기대하므로 RGB 이미지는 변환 필요
    save_path = osp.join(folder, filename)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mmcv.imwrite(img_bgr, save_path)
    print(f"Saved: {save_path}")

def main():
    args = parse_args()
    
    print(f"Loading config: {args.config}")
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmpose'))

    # 1. 모델(Augmentor) 및 설정 로드
    mean_vals = cfg.model.data_preprocessor.mean
    std_vals = cfg.model.data_preprocessor.std
    mean_tensor = torch.tensor(mean_vals).view(1, 3, 1, 1).to(args.device)
    std_tensor = torch.tensor(std_vals).view(1, 3, 1, 1).to(args.device)

    print("Building Augmentor...")
    aug_cfg = cfg.model.data_preprocessor.batch_augments[0]
    augmentor = MODELS.build(aug_cfg)
    augmentor.to(args.device)
    augmentor.eval() # 훈련 모드가 아니어도 강제로 forward 호출 예정

    # 2. 데이터셋 빌드 (어노테이션 정보 획득용)
    print("Building Dataset...")
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    
    # 데이터셋 인덱싱 (Filename -> Index 매핑)
    # 데이터셋이 클 경우 약간의 시간이 소요될 수 있습니다.
    print("Indexing dataset...")
    filename_to_idx = {}
    for idx in range(len(dataset)):
        info = dataset.get_data_info(idx)
        # img_path에서 파일명만 추출
        fname = osp.splitext(osp.basename(info['img_path']))[0]
        filename_to_idx[fname] = idx

    # 3. CPU 파이프라인 구성 (Loader 제외)
    # Config의 train_pipeline에서 'LoadImage...' 관련 부분을 제외하고 구성합니다.
    # 이미지를 수동으로 로드해서 넣어줄 것이기 때문입니다.
    cpu_transforms = []
    for t in cfg.train_pipeline:
        if 'LoadImage' in t['type']:
            continue
        cpu_transforms.append(TRANSFORMS.build(t))
    cpu_pipeline = Compose(cpu_transforms)

    # 4. SPIN (Aux) 경로 설정 확인
    # Config의 LoadImageFromDualDir 설정에서 aux_dir 경로를 가져옵니다.
    aux_dir = None
    aux_suffix = '.png' # 기본값
    for t in cfg.train_pipeline:
        if t['type'] == 'LoadImageFromDualDir':
            aux_dir = t['aux_dir']
            aux_suffix = t.get('aux_suffix', '.png')
            break
    
    if aux_dir is None:
        print("Warning: LoadImageFromDualDir not found in pipeline. SPIN images might not be loaded.")

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================
    # 메인 루프: 타겟 이미지 처리
    # =========================================================
    for target_id in TARGET_IDS:
        if target_id not in filename_to_idx:
            print(f"[Skip] ID '{target_id}' not found in dataset.")
            continue

        print(f"\nProcessing {target_id}...")
        
        # 저장 폴더 생성
        save_dir = osp.join(args.output_dir, target_id)
        os.makedirs(save_dir, exist_ok=True)

        idx = filename_to_idx[target_id]
        raw_data_info = dataset.get_data_info(idx)
        
        # -----------------------------------------------------
        # A. 원본 이미지 로드 (SPEED+ & SPIN)
        # -----------------------------------------------------
        # 1. SPEED+ (Original)
        speed_path = raw_data_info['img_path']
        img_speed = mmcv.imread(speed_path, channel_order='rgb')
        save_image(img_speed, save_dir, "01_Original_SPEED+.png")

        # 2. SPIN (Aux)
        img_spin = None
        if aux_dir:
            spin_path = osp.join(aux_dir, target_id + aux_suffix)
            if osp.exists(spin_path):
                img_spin = mmcv.imread(spin_path, channel_order='rgb')
                save_image(img_spin, save_dir, "01_Original_SPIN.png")
            else:
                print(f"  -> SPIN image not found at {spin_path}")

        # -----------------------------------------------------
        # B. CPU Augmentation 적용 (SPEED+ & SPIN 각각)
        # -----------------------------------------------------
        
        def run_cpu_pipeline(base_info, image_array, source_name):
            # 데이터 dict 복사 및 이미지 주입
            data = base_info.copy()
            data['img'] = image_array
            data['img_shape'] = image_array.shape[:2]
            data['ori_shape'] = image_array.shape[:2]
            data['img_source'] = source_name # 로깅용 메타데이터 (pipeline에 영향 X)
            
            # 파이프라인 실행
            # (SetFullImageBBox -> SPNAugmentation -> ... -> PackPoseInputs)
            output = cpu_pipeline(data)
            return output

        # SPEED+ CPU Aug
        speed_out = run_cpu_pipeline(raw_data_info, img_speed, 'original')
        # 시각화를 위해 Tensor -> Image 역변환 (Normalize 전 단계라 가정하고 Clamp만 확인)
        # PackPoseInputs 이후 results['inputs']는 보통 0-255 범위의 Tensor (Config의 codec normalize=False 확인)
        # 만약 to_float32 등이 있었다면 범위가 다를 수 있음.
        tensor_speed_cpu = speed_out['inputs'] # (C, H, W)
        
        # 0-255 범위로 가정하고 시각화
        img_speed_cpu = tensor_speed_cpu.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        save_image(img_speed_cpu, save_dir, "02_CPU_Aug_SPEED+.png")

        # SPIN CPU Aug
        tensor_spin_cpu = None
        if img_spin is not None:
            spin_out = run_cpu_pipeline(raw_data_info, img_spin, 'aux')
            tensor_spin_cpu = spin_out['inputs']
            img_spin_cpu = tensor_spin_cpu.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            save_image(img_spin_cpu, save_dir, "02_CPU_Aug_SPIN.png")

        # -----------------------------------------------------
        # C. Tensor Augmentation 적용 (SPEED+ 및 SPIN 모두)
        # -----------------------------------------------------
        
        # 강제로 특정 Augmentation만 적용하는 헬퍼 함수
        def apply_forced_aug(aug_model, tensor_in, mode):
            """
            mode: 'Identity', 'RandConv', 'StyleAug', 'DeepAug'
            CombinedAugmentation 내부의 probs를 조작하여 강제 실행
            """
            # 기존 probs 백업
            original_probs = aug_model.probs.clone()
            
            # 강제 설정 (순서: Identity, RandConv, StyleAug, DeepAug)
            if mode == 'Identity':
                new_probs = [1.0, 0.0, 0.0, 0.0]
            elif mode == 'RandConv':
                new_probs = [0.0, 1.0, 0.0, 0.0]
            elif mode == 'StyleAug':
                new_probs = [0.0, 0.0, 1.0, 0.0]
            elif mode == 'DeepAug':
                new_probs = [0.0, 0.0, 0.0, 1.0]
            
            # 확률 조작 및 모드 변경
            aug_model.probs = torch.tensor(new_probs, device=args.device)
            aug_model.train() # Dropout 등 활성화를 위해 train 모드
            
            with torch.no_grad():
                out_tensor, _ = aug_model(tensor_in, None)
            
            # 원상 복구
            aug_model.probs = original_probs
            aug_model.eval()
            
            return out_tensor

        # 처리할 대상 리스트 생성 (이름, 텐서데이터)
        aug_targets = [('SPEED+', tensor_speed_cpu)]
        if tensor_spin_cpu is not None:
            aug_targets.append(('SPIN', tensor_spin_cpu))

        aug_modes = ['RandConv', 'StyleAug', 'DeepAug']

        # Loop: 각 소스(SPEED+, SPIN)에 대해 수행
        for source_name, src_tensor in aug_targets:
            if src_tensor is None: continue

            # 1. Augmentor 입력 준비 (Normalize)
            # (C, H, W) -> (1, C, H, W) -> Normalize
            input_tensor = src_tensor.unsqueeze(0).float().to(args.device)
            input_normalized = (input_tensor - mean_tensor) / std_tensor

            # 2. 각 Augmentation Mode 적용
            for mode in aug_modes:
                try:
                    # Augmentation 실행
                    out_tensor = apply_forced_aug(augmentor, input_normalized, mode)
                    
                    # 역정규화 (Denormalize) 및 저장
                    img_out = denormalize(out_tensor[0], mean_vals, std_vals)
                    
                    # 파일명 예: 03_Tensor_RandConv_on_SPEED+.png, 03_Tensor_DeepAug_on_SPIN.png
                    save_filename = f"03_Tensor_{mode}_on_{source_name}.png"
                    save_image(img_out, save_dir, save_filename)
                    
                except Exception as e:
                    print(f"  -> Error applying {mode} on {source_name}: {e}")

    print("\nAll Done! Check the output directory.")

if __name__ == '__main__':
    main()