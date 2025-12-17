import cv2
import numpy as np
import os
import json
import random
from tqdm import tqdm

# -----------------------------------------------------------------------
# (A) Helper Functions (Crop & BBox)
# -----------------------------------------------------------------------

def get_bbox_from_mask(mask_path):
    """
    Calculate BBox from synthetic/masks
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return None
    coords = np.where(mask > 0)
    if len(coords[0]) == 0 or len(coords[1]) == 0: return None
    ymin, ymax = np.min(coords[0]), np.max(coords[0])
    xmin, xmax = np.min(coords[1]), np.max(coords[1])
    return xmin, ymin, xmax, ymax

def get_safe_crop_coords(img_shape, target_center_x, target_center_y, target_size):
    """
    Calculate crop coordinates ensuring not to exceed boundaries
    """
    img_h, img_w = img_shape
    xmin = target_center_x - target_size // 2
    ymin = target_center_y - target_size // 2
    xmax = xmin + target_size
    ymax = ymin + target_size
    
    dx = 0
    if xmin < 0: dx = -xmin
    elif xmax > img_w: dx = img_w - xmax
        
    dy = 0
    if ymin < 0: dy = -ymin
    elif ymax > img_h: dy = img_h - ymax

    final_xmin = int(xmin + dx)
    final_ymin = int(ymin + dy)
    final_xmax = int(xmax + dx)
    final_ymax = int(ymax + dy)
    
    return final_xmin, final_ymin, final_xmax, final_ymax

# -----------------------------------------------------------------------
# (B) Main Processing Logic for SPIN
# -----------------------------------------------------------------------

def main():
    
    root_dir = '/workspace'
    speedplus_dir = os.path.join(root_dir, 'speedplusv2')

    synthetic_dir = os.path.join(speedplus_dir, 'synthetic')
    spin_dir = os.path.join(root_dir, 'SPIN')
    
    output_dir = os.path.join(speedplus_dir, 'SPIN_preprocessed')
    os.makedirs(output_dir, exist_ok=True)
    
    # SPIN data == SPEED+ train set
    train_json_path = os.path.join(synthetic_dir, 'train.json')

    # Random setting for training
    RANDOM_SIZE_RANGE = (1.1, 1.3)
    RANDOM_SHIFT_RATIO = 0.1
    FINAL_RESIZE_SIZE = 224

    print(f"Loading label file from: {train_json_path}")
    try:
        with open(train_json_path, 'r') as f: labels = json.load(f)
    except FileNotFoundError:
        print("Error: train.json not found."); return

    print(f"Processing SPIN dataset based on {len(labels)} entries...")
    print(f"Images will be saved to: {output_dir}")

    for image_data in tqdm(labels):
        
        orig_filename = image_data['filename'] 
        file_stem = os.path.splitext(orig_filename)[0] 

        # Mask
        mask_path = os.path.join(synthetic_dir, 'masks', orig_filename)
        
        # SPIN Image
        spin_image_filename = file_stem + '.png'
        image_path = os.path.join(spin_dir, spin_image_filename)
        if not os.path.exists(image_path):
            continue
        
        new_filename = file_stem.replace('img', '') + '.png'
        output_path = os.path.join(output_dir, new_filename)

        bbox = get_bbox_from_mask(mask_path)
        if bbox is None:
            continue 

        xmin, ymin, xmax, ymax = bbox
        bbox_w = xmax - xmin; bbox_h = ymax - ymin
        bbox_center_x = xmin + bbox_w / 2
        bbox_center_y = ymin + bbox_h / 2
        long_side = max(bbox_w, bbox_h)

        # Random Jitter & Shift 
        size_jitter = random.uniform(RANDOM_SIZE_RANGE[0], RANDOM_SIZE_RANGE[1])
        target_size = int(long_side * size_jitter)
        
        max_shift = int(long_side * RANDOM_SHIFT_RATIO / 2)
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        
        target_center_x = bbox_center_x + shift_x
        target_center_y = bbox_center_y + shift_y

        image = cv2.imread(image_path)
        if image is None:
            continue
            
        img_h, img_w = image.shape[:2]
        target_size = min(target_size, img_h, img_w)

        # Safe Crop
        cx_min, cy_min, cx_max, cy_max = get_safe_crop_coords(
            (img_h, img_w), target_center_x, target_center_y, target_size
        )
        
        crop_size = cx_max - cx_min
        if crop_size == 0: continue

        # Crop & Resize
        cropped_image = image[cy_min:cy_max, cx_min:cx_max]
        cropped_resized_image = cv2.resize(
            cropped_image, 
            (FINAL_RESIZE_SIZE, FINAL_RESIZE_SIZE), 
            interpolation=cv2.INTER_LINEAR
        )

        cv2.imwrite(output_path, cropped_resized_image)

    print("\n--- SPIN dataset processing complete. ---")

if __name__ == '__main__':
    main()