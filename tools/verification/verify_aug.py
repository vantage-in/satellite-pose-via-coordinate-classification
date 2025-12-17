'''
python tools/verification/verify_aug.py --images 000021 000012 000015 --model specc-s
'''
import argparse
import os
import os.path as osp
import torch
import mmcv
import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmpose.registry import DATASETS, MODELS, TRANSFORMS
from mmcv.transforms import Compose
from mmpose.utils import register_all_modules

from configs import project_config as cfg

# -----------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Verify Pipeline Step-by-Step')
    
    parser.add_argument('--images', nargs='+', required=True, 
                        help='List of target image IDs (filename without extension)')
    
    parser.add_argument('--model', type=str, default='specc-s', choices=['specc-s', 'specc-m'],
                        help='Model config type')
    parser.add_argument('--config', type=str, default=None, 
                        help='Path to config file (overrides --model default)')
    
    parser.add_argument('--output-dir', default='pipeline_verification', help='output directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='device to run augmentation')
    
    args = parser.parse_args()
    return args

def denormalize(tensor, mean, std):
    """Normalized Tensor -> Image(0-255, RGB)"""
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
    """RGB -> BGR for cv2 saving"""
    save_path = osp.join(folder, filename)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mmcv.imwrite(img_bgr, save_path)
    print(f"Saved: {save_path}")

def main():
    args = parse_args()
    register_all_modules()

    if args.config:
        config_path = args.config
    else:
        config_path = cfg.MODELS[args.model]['config_file']

    print(f"Loading config from: {config_path}")

    mm_cfg = Config.fromfile(config_path)
    if hasattr(mm_cfg, 'custom_imports'):
        mm_cfg.custom_imports = None

    init_default_scope(mm_cfg.get('default_scope', 'mmpose'))

    # Augmentor
    if hasattr(mm_cfg.model, 'data_preprocessor'):
        preprocessor = mm_cfg.model.data_preprocessor
    else:
        print("Error: 'data_preprocessor' not found in model config.")
        return

    mean_vals = preprocessor.mean
    std_vals = preprocessor.std
    mean_tensor = torch.tensor(mean_vals).view(1, 3, 1, 1).to(args.device)
    std_tensor = torch.tensor(std_vals).view(1, 3, 1, 1).to(args.device)

    print("Building Augmentor...")
    # Tensor Augmentation (Batch Augments)
    if hasattr(preprocessor, 'batch_augments') and preprocessor.batch_augments:
        aug_cfg = preprocessor.batch_augments[0]
        augmentor = MODELS.build(aug_cfg)
        augmentor.to(args.device)
        augmentor.eval()
    else:
        print("Warning: No batch_augments found in config.")
        augmentor = None

    # Dataset
    print("Building Dataset...")
    dataset = DATASETS.build(mm_cfg.train_dataloader.dataset)
    
    print("Indexing dataset...")
    filename_to_idx = {}
    for idx in range(len(dataset)):
        info = dataset.get_data_info(idx)
        fname = osp.splitext(osp.basename(info['img_path']))[0]
        
        clean_fname = fname[3:] if fname.startswith('img') else fname
        
        filename_to_idx[fname] = idx
        filename_to_idx[clean_fname] = idx 

    cpu_transforms = []
    for t in mm_cfg.train_pipeline:
        if 'LoadImage' in t['type']:
            continue
        cpu_transforms.append(TRANSFORMS.build(t))
    cpu_pipeline = Compose(cpu_transforms)

    aux_dir = None
    aux_suffix = '.png'
    for t in mm_cfg.train_pipeline:
        if t['type'] == 'LoadImageFromDualDir':
            aux_dir = t['aux_dir']
            aux_suffix = t.get('aux_suffix', '.png')
            break
    
    if aux_dir is None:
        print("Warning: LoadImageFromDualDir not found. SPIN images won't be loaded.")

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================
    # Main loop
    # =========================================================
    for target_id in args.images:
        clean_id = osp.splitext(target_id)[0]
        
        if clean_id not in filename_to_idx:
            print(f"\n[Skip] ID '{clean_id}' not found in dataset.")
            continue

        print(f"\nProcessing {clean_id}...")
        save_dir = osp.join(args.output_dir, clean_id)
        os.makedirs(save_dir, exist_ok=True)

        idx = filename_to_idx[clean_id]
        raw_data_info = dataset.get_data_info(idx)
        
        # -----------------------------------------------------
        # A. Original images
        # -----------------------------------------------------
        # SPEED+
        speed_path = raw_data_info['img_path']
        img_speed = mmcv.imread(speed_path, channel_order='rgb')
        save_image(img_speed, save_dir, "01_Original_SPEED+.png")

        # SPIN
        img_spin = None
        if aux_dir:
            numeric_id = clean_id[3:] if clean_id.startswith('img') else clean_id
            spin_path = osp.join(aux_dir, numeric_id + aux_suffix)
            
            if osp.exists(spin_path):
                img_spin = mmcv.imread(spin_path, channel_order='rgb')
                save_image(img_spin, save_dir, "01_Original_SPIN.png")
            else:
                print(f"  -> SPIN image not found at {spin_path}")

        # -----------------------------------------------------
        # B. CPU Augmentation
        # -----------------------------------------------------
        def run_cpu_pipeline(base_info, image_array, source_name):
            data = base_info.copy()
            data['img'] = image_array
            data['img_shape'] = image_array.shape[:2]
            data['ori_shape'] = image_array.shape[:2]
            data['img_source'] = source_name
            return cpu_pipeline(data)

        # SPEED+ CPU
        speed_out = run_cpu_pipeline(raw_data_info, img_speed, 'original')
        tensor_speed_cpu = speed_out['inputs'] # (C, H, W)
        img_speed_cpu = tensor_speed_cpu.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        save_image(img_speed_cpu, save_dir, "02_CPU_Aug_SPEED+.png")

        # SPIN CPU
        tensor_spin_cpu = None
        if img_spin is not None:
            spin_out = run_cpu_pipeline(raw_data_info, img_spin, 'aux')
            tensor_spin_cpu = spin_out['inputs']
            img_spin_cpu = tensor_spin_cpu.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            save_image(img_spin_cpu, save_dir, "02_CPU_Aug_SPIN.png")

        # -----------------------------------------------------
        # C. Tensor Augmentation
        # -----------------------------------------------------
        if augmentor is None:
            continue

        def apply_forced_aug(aug_model, tensor_in, mode):
            original_probs = aug_model.probs.clone()
            
            if mode == 'Identity':
                new_probs = [1.0, 0.0, 0.0, 0.0]
            elif mode == 'RandConv':
                new_probs = [0.0, 1.0, 0.0, 0.0]
            elif mode == 'StyleAug':
                new_probs = [0.0, 0.0, 1.0, 0.0]
            elif mode == 'DeepAug':
                new_probs = [0.0, 0.0, 0.0, 1.0]
            else:
                return tensor_in

            if len(new_probs) != len(original_probs):
                 new_probs = new_probs[:len(original_probs)]
            
            aug_model.probs = torch.tensor(new_probs, device=args.device)
            aug_model.train()
            
            with torch.no_grad():
                out_tensor, _ = aug_model(tensor_in, None)
            
            aug_model.probs = original_probs
            aug_model.eval()
            return out_tensor

        aug_targets = [('SPEED+', tensor_speed_cpu)]
        if tensor_spin_cpu is not None:
            aug_targets.append(('SPIN', tensor_spin_cpu))

        aug_modes = ['RandConv', 'StyleAug', 'DeepAug']

        for source_name, src_tensor in aug_targets:
            if src_tensor is None: continue

            # Normalize
            input_tensor = src_tensor.unsqueeze(0).float().to(args.device)
            input_normalized = (input_tensor - mean_tensor) / std_tensor

            for mode in aug_modes:
                try:
                    out_tensor = apply_forced_aug(augmentor, input_normalized, mode)
                    img_out = denormalize(out_tensor[0], mean_vals, std_vals)
                    
                    save_filename = f"03_Tensor_{mode}_on_{source_name}.png"
                    save_image(img_out, save_dir, save_filename)
                except Exception as e:
                    print(f"  -> Error applying {mode} on {source_name}: {e}")

    print("\nAll Done! Check the output directory.")

if __name__ == '__main__':
    main()