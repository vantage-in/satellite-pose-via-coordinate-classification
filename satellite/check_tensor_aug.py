# check_tensor_aug.py

import argparse
import os
import os.path as osp
import torch
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.registry import DATASETS, MODELS, TRANSFORMS
from mmcv.transforms import Compose

# Headless 환경 지원
import matplotlib
matplotlib.use('Agg')

def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Tensor Augmentation & Dual Loader')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--output-dir', default='tensor_aug_vis', help='output directory')
    parser.add_argument('--num-samples', type=int, default=20, help='number of samples to check')
    parser.add_argument('--device', default='cuda', help='device to run augmentation')
    args = parser.parse_args()
    return args

def denormalize(tensor, mean, std):
    """Normalize된 Tensor를 이미지(0-255, RGB)로 변환"""
    tensor = tensor.detach().cpu().clone()
    tensor = tensor.permute(1, 2, 0) # (C, H, W) -> (H, W, C)
    
    mean = torch.tensor(mean).view(1, 1, 3)
    std = torch.tensor(std).view(1, 1, 3)
    
    img = tensor * std + mean
    img = img.clamp(0, 255).numpy().astype(np.uint8)
    return img

def main():
    args = parse_args()
    
    print(f"Loading config: {args.config}")
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get('default_scope', 'mmpose'))

    # [수정 1] 데이터셋 빌드 (정보만 로드)
    print("Building dataset...")
    # lazy_init=True를 사용하여 무거운 로딩 방지 가능하지만 여기선 기본 빌드
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    
    # [수정 2] 파이프라인 빌드 (LoadImageFromDualDir 동작 확인용)
    # Config에서 train_pipeline 가져오기
    # PackPoseInputs 전까지만 실행해서 Tensor 변환 전 상태를 보고자 함
    # 하지만 여기서는 최종 Augmentor 입력(Tensor)까지 가야 하므로
    # PackPoseInputs 직전까지 돌리고, Tensor 변환은 수동으로 처리하거나
    # 전체 파이프라인을 돌린 뒤 data_sample에서 꺼내는 방식을 씁니다.
    
    # 편의상: 전체 파이프라인을 빌드해서 돌립니다.
    pipeline = Compose(cfg.train_pipeline)

    print("Building Augmentor...")
    if hasattr(cfg.model, 'data_preprocessor') and \
       'batch_augments' in cfg.model.data_preprocessor:
        aug_cfg = cfg.model.data_preprocessor.batch_augments[0]
        try:
            augmentor = MODELS.build(aug_cfg)
        except Exception as e:
            print(f"Error building augmentor: {e}")
            return
        augmentor.to(args.device)
        augmentor.train() 
    else:
        print("Config에 batch_augments가 없습니다.")
        return
    
    # [중요] 최신 tensor_augmentor.py는 'latest_choices'를 저장한다고 가정합니다.
    # 만약 저장하지 않도록 되어 있다면 augmentor 코드에 self.latest_choices = choices 추가 필요.

    mean_vals = cfg.model.data_preprocessor.mean
    std_vals = cfg.model.data_preprocessor.std
    
    # Config 값이 리스트라면 텐서로 변환
    mean_tensor = torch.tensor(mean_vals).view(1, 3, 1, 1).to(args.device)
    std_tensor = torch.tensor(std_vals).view(1, 3, 1, 1).to(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    
    AUG_NAMES = {0: "Identity", 1: "RandConv", 2: "StyleAug", 3: "DeepAug"}

    print(f"Start visualizing {args.num_samples} samples...")
    
    # 데이터셋의 인덱스를 랜덤하게 섞어서 샘플링
    indices = np.random.choice(len(dataset), args.num_samples, replace=False)

    for i, idx in enumerate(indices):
        # 1. Raw Data Info 가져오기
        data_info = dataset.get_data_info(idx)
        
        # 2. 파이프라인 실행 (LoadImageFromDualDir -> ... -> PackPoseInputs)
        # 이 과정에서 'img_source' 키가 results 딕셔너리에 추가될 것입니다.
        data_batch = pipeline(data_info)
        
        # PackPoseInputs 결과는 'inputs'(Tensor)와 'data_samples'를 포함함
        input_tensor = data_batch['inputs'].float().to(args.device)
        input_batch = input_tensor.unsqueeze(0) # (1, C, H, W)
        
        # [소스 확인] LoadImageFromDualDir가 남긴 흔적 찾기
        # PackPoseInputs를 거치면 dict가 분해되므로, data_samples 안에 metainfo로 들어갔는지 확인해야 함.
        # 보통 metainfo에 사용자 정의 키는 안 들어갈 수 있으므로,
        # 파이프라인 중간 결과를 낚아채는 게 가장 확실하지만,
        # 여기서는 data_sample.metainfo를 확인해봅니다. (안 들어있으면 'Unknown')
        
        data_sample = data_batch['data_samples']
        img_source = data_sample.metainfo.get('img_source', 'NA')
        
        # 3. 정규화 (Augmentor 입력 준비)
        # MMPose는 전처리 단계에서 정규화를 하지만, 
        # 여기선 Raw Image -> Pipeline -> Tensor(0-255 range usually if PackPoseInputs does normalize=False)
        # Config의 codec['normalize']=False, mean/std가 data_preprocessor에 있으면
        # Pipeline 출력은 0-255 범위일 가능성이 높음. (확인 필요)
        # 만약 Pipeline 출력 자체가 이미 정규화되어 있다면 이 단계는 건너뛰어야 함.
        # 보통 ToTensor/ImageToTensor가 없으면 0-255 numpy -> tensor 변환만 일어남.
        
        # 가정: Pipeline 출력은 0-255 범위의 Tensor.
        normalized_batch = (input_batch - mean_tensor) / std_tensor
        
        # 4. Augmentation 실행
        # Augmentor는 (Normalized Tensor, DataSamples)를 받음
        augmented_batch, _ = augmentor(normalized_batch, None)

        # 5. 적용된 증강 확인
        aug_name = "Unknown"
        if hasattr(augmentor, 'latest_choices'):
            choice_idx = augmentor.latest_choices[0].item()
            aug_name = AUG_NAMES.get(choice_idx, f"Unknown({choice_idx})")
        
        print(f"[Sample {i:02d}] Source: {img_source}, Aug: {aug_name}")

        # 6. 시각화 (역정규화)
        img_org = denormalize(normalized_batch[0], mean_vals, std_vals)
        img_aug = denormalize(augmented_batch[0], mean_vals, std_vals)
        
        # 7. 저장
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(img_org)
        axes[0].set_title(f"Original\nSource: {img_source}")
        axes[0].axis('off')
        
        axes[1].imshow(img_aug)
        axes[1].set_title(f"Augmented\nType: {aug_name}")
        axes[1].axis('off')
        
        save_path = osp.join(args.output_dir, f"sample_{i:03d}_{img_source}_{aug_name.split()[0]}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    print(f"Done. Results saved in {args.output_dir}")

if __name__ == '__main__':
    main()