import albumentations as A

# 'pixel' related augmentations
from albumentations.augmentations.pixel.transforms import (
    RandomBrightnessContrast, Posterize, Sharpen,
    Solarize, Spatter, Equalize, RandomGamma, RandomSunFlare, GaussNoise, ISONoise, MultiplicativeNoise
)
# 'blur' related augmentations
from albumentations.augmentations.blur.transforms import (
    GaussianBlur, MedianBlur, Defocus
)
from albumentations.core.composition import OneOf
from mmpose.registry import TRANSFORMS
import numpy as np
import warnings

@TRANSFORMS.register_module(force=True)
class SPNAugmentation:
    """
    Apply RandAugment-like based on SPNv3
    
    1. Grayscale conversion randomly
    2. Choose the number of augmentations and their probabilities
    """
    def __init__(self, gray_prob: float = 0.85):
        self.gray_prob = gray_prob
        
        # 0. Grayscale conversion
        self.gray_transform = A.ToGray(p=1.0)

        # Prob. distribution for choosing N
        self.n_choices = [4, 3, 2, 1, 0]
        self.n_probs = [0.1, 0.2, 0.3, 0.2, 0.2]
        
        # 1. RandomSunFlare
        self.flare_transform = RandomSunFlare(p=1.0, src_radius=100)
        
        # 2. Pool of augmentations
        self.simple_aug_pool = [
            RandomBrightnessContrast(p=1.0),
            OneOf([
                GaussianBlur(blur_limit=(3, 7), p=1.0), 
                MedianBlur(blur_limit=(3, 7), p=1.0), 
                Defocus(radius=(1, 3), p=1.0)
            ], p=1.0),
            OneOf([
                GaussNoise(p=1.0), 
                ISONoise(p=1.0), 
                MultiplicativeNoise(p=1.0)
            ], p=1.0),
            Posterize(num_bits=(4, 8), p=1.0),
            Sharpen(p=1.0),
            Solarize(p=1.0),
            Spatter(p=1.0),
            Equalize(p=1.0),
            RandomGamma(p=1.0),
        ]
        
        # Total augmentations
        self.aug_pool = self.simple_aug_pool + [self.flare_transform]

    def _apply_flare_to_bbox(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Apply RandomSunFlare on the center region of BBox.
        """
        try:
            if bbox.ndim > 1:
                bbox = bbox[0]
            
            x1, y1, x2, y2 = bbox.astype(float)
            w_box = x2 - x1
            h_box = y2 - y1
            
            margin_w = w_box * 0.25
            margin_h = h_box * 0.25
            
            new_x1 = int(x1 + margin_w)
            new_y1 = int(y1 + margin_h)
            new_x2 = int(x2 - margin_w)
            new_y2 = int(y2 - margin_h)
            
            h_img, w_img = img.shape[:2]
            new_x1, new_y1 = max(0, new_x1), max(0, new_y1)
            new_x2, new_y2 = min(w_img, new_x2), min(h_img, new_y2)

            if new_x1 >= new_x2 or new_y1 >= new_y2:
                return img

            bbox_crop = img[new_y1:new_y2, new_x1:new_x2]
            augmented_crop = self.flare_transform(image=bbox_crop)['image']
            
            img_copy = img.copy()
            img_copy[new_y1:new_y2, new_x1:new_x2] = augmented_crop
            return img_copy

        except Exception as e:
            warnings.warn(f"BBox SunFlare failed: {e}. Return original image.")
            return img

    def __call__(self, results: dict) -> dict:
        """Call from MMPose pipeline."""
        
        img = results['img']
        
        # ---------------------------------------------------------
        # Step 1: Grayscale conversion
        # ---------------------------------------------------------
        if np.random.rand() < self.gray_prob:
            # maintain channel 3
            img = self.gray_transform(image=img)['image']

        # ---------------------------------------------------------
        # Step 2: Choose N 
        # ---------------------------------------------------------
        n = np.random.choice(self.n_choices, p=self.n_probs)

        if n == 0:
            results['img'] = img
            return results

        # ---------------------------------------------------------
        # Step 3: Apply augmentations
        # ---------------------------------------------------------

        if 'bbox' not in results:
            warnings.warn("SPNAugmentation: No 'bbox' Key. Apply to the entire image.")
            bbox = None
        else:
            bbox = results['bbox'] 

        chosen_indices = np.random.choice(
            len(self.aug_pool), n, replace=False
        )
        
        for idx in chosen_indices:
            aug_fn = self.aug_pool[idx]

            if aug_fn is self.flare_transform:
                if bbox is not None:
                    img = self._apply_flare_to_bbox(img, bbox)
                else:
                    img = aug_fn(image=img)['image']
            else:
                img = aug_fn(image=img)['image']
        
        results['img'] = img
        return results