# cpu_augmentor.py (modified)

import albumentations as A

#'pixel' 관련 증강
from albumentations.augmentations.pixel.transforms import (
    RandomBrightnessContrast, Posterize, Sharpen,
    Solarize, Spatter, Equalize, RandomGamma, RandomSunFlare, GaussNoise, ISONoise, MultiplicativeNoise
)
# 'blur' 관련 증강
from albumentations.augmentations.blur.transforms import (
    GaussianBlur, MedianBlur, Defocus
)
# 합성(composition) 유틸리티
from albumentations.core.composition import OneOf
from mmpose.registry import TRANSFORMS
import numpy as np
import warnings

@TRANSFORMS.register_module()
class SPNAugmentation:
    """
    선행 연구의 RandAugment 유사 증강(Table 1)을 적용합니다.
    
    [변경 사항]
    1. Grayscale 변환: 0.8 확률로 먼저 적용
    2. N(적용할 증강 개수) 선택: 확률 분포에 따라 N={0,1,2,3,4} 중 선택
       - n=4 (0.1), n=3 (0.1), n=2 (0.3), n=1 (0.3), n=0 (0.2)
    """
    def __init__(self, gray_prob: float = 0.85):
        self.gray_prob = gray_prob
        
        # 0. Grayscale 변환기 (확률에 따라 적용됨)
        # p=1.0으로 초기화하고 호출 시 확률 제어
        self.gray_transform = A.ToGray(p=1.0)

        # N 선택을 위한 확률 분포 정의
        self.n_choices = [4, 3, 2, 1, 0]
        self.n_probs = [0.1, 0.2, 0.3, 0.2, 0.2]
        
        # 1. RandomSunFlare 변환
        self.flare_transform = RandomSunFlare(p=1.0, src_radius=100)
        
        # 2. 증강 함수 풀 정의
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
        
        # 3. 전체 증강 풀 결합
        self.aug_pool = self.simple_aug_pool + [self.flare_transform]

    def _apply_flare_to_bbox(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        RandomSunFlare를 이미지의 BBox 중앙 50% 영역에만 적용합니다.
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
            warnings.warn(f"BBox SunFlare 적용 실패: {e}. 원본 이미지 반환.")
            return img

    def __call__(self, results: dict) -> dict:
        """MMPose 파이프라인에서 호출됩니다."""
        
        img = results['img']
        
        # ---------------------------------------------------------
        # Step 1: Grayscale 적용 (확률 0.8)
        # ---------------------------------------------------------
        if np.random.rand() < self.gray_prob:
            # Albumentations의 ToGray는 3채널을 유지하며 값을 동일하게 만듭니다.
            img = self.gray_transform(image=img)['image']

        # ---------------------------------------------------------
        # Step 2: N (적용할 증강 개수) 선택
        # ---------------------------------------------------------
        n = np.random.choice(self.n_choices, p=self.n_probs)

        # n이 0이면 증강 없이 (Grayscale만 적용되었을 수 있음) 리턴
        if n == 0:
            results['img'] = img
            return results

        # ---------------------------------------------------------
        # Step 3: N개의 증강 선택 및 적용
        # ---------------------------------------------------------
        
        # BBox 정보 가져오기
        if 'bbox' not in results:
            warnings.warn("SPNAugmentation: 'bbox' 키 없음. 전체 이미지에 적용됨.")
            bbox = None
        else:
            bbox = results['bbox'] 

        # 중복 없이 N개 선택
        chosen_indices = np.random.choice(
            len(self.aug_pool), n, replace=False
        )
        
        for idx in chosen_indices:
            aug_fn = self.aug_pool[idx]
            
            # SunFlare는 BBox 내부에만 적용
            if aug_fn is self.flare_transform:
                if bbox is not None:
                    img = self._apply_flare_to_bbox(img, bbox)
                else:
                    img = aug_fn(image=img)['image']
            else:
                img = aug_fn(image=img)['image']
        
        results['img'] = img
        return results

# # my_spn_aug.py

# import albumentations as A

# #'pixel' 관련 증강
# from albumentations.augmentations.pixel.transforms import (
#     RandomBrightnessContrast, Posterize, Sharpen,
#     Solarize, Spatter, Equalize, RandomGamma, RandomSunFlare, GaussNoise, ISONoise, MultiplicativeNoise
# )
# # 'blur' 관련 증강
# from albumentations.augmentations.blur.transforms import (
#     GaussianBlur, MedianBlur, Defocus
# )
# # 합성(composition) 유틸리티
# from albumentations.core.composition import OneOf
# from albumentations.core.composition import OneOf
# from mmpose.registry import TRANSFORMS
# import numpy as np
# import warnings

# @TRANSFORMS.register_module()
# class SPNAugmentation:
#     """
#     선행 연구의 RandAugment 유사 증강(Table 1)을 적용합니다.
#     - N개의 연산을 무작위로 선택하여 순차 적용합니다.
#     - RandomSunFlare는 'bbox' 키를 사용하여 GT BBox 내부에만 적용됩니다.
#     """
#     def __init__(self, n: int = 2, p: float = 0.9):
#         self.n = n
#         self.p = p
        
#         # 1. RandomSunFlare 변환을 별도로 인스턴스화합니다.
#         # (p=1.0으로 설정하여, 일단 선택되면 항상 적용되도록 함)
#         self.flare_transform = RandomSunFlare(p=1.0, src_radius=100)
        
#         # 2. Table 1의 나머지 증강 함수 풀을 정의합니다.
#         self.simple_aug_pool = [
#             RandomBrightnessContrast(p=1.0),
#             OneOf([
#                 GaussianBlur(blur_limit=(3, 7), p=1.0), 
#                 MedianBlur(blur_limit=(3, 7), p=1.0), 
#                 Defocus(radius=(1, 3), p=1.0)
#             ], p=1.0),
#             OneOf([
#                 GaussNoise(p=1.0), 
#                 ISONoise(p=1.0), 
#                 MultiplicativeNoise(p=1.0)
#             ], p=1.0),
#             Posterize(num_bits=(4, 8), p=1.0),
#             Sharpen(p=1.0),
#             Solarize(p=1.0),
#             Spatter(p=1.0),
#             Equalize(p=1.0),
#             RandomGamma(p=1.0),
#         ]
        
#         # 3. 전체 증강 풀을 결합합니다.
#         # (flare_transform을 식별할 수 있도록 리스트에 그대로 추가)
#         self.aug_pool = self.simple_aug_pool + [self.flare_transform]

#     def _apply_flare_to_bbox(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
#         """
#         RandomSunFlare를 이미지의 BBox 중앙 50% 영역에만 적용합니다.
#         (상하좌우에서 각각 25%씩 안쪽으로 들어간 영역)
#         """
#         try:
#             if bbox.ndim > 1:
#                 bbox = bbox[0]
            
#             # 1. 원본 BBox 좌표 확보
#             x1, y1, x2, y2 = bbox.astype(float) # 계산을 위해 float 변환
            
#             # 2. BBox의 너비와 높이 계산
#             w_box = x2 - x1
#             h_box = y2 - y1
            
#             # 3. 중앙 50% 영역 계산
#             # 전체 길이의 25%씩 양쪽에서 줄이면 가운데 50%가 남습니다.
#             margin_w = w_box * 0.25
#             margin_h = h_box * 0.25
            
#             new_x1 = int(x1 + margin_w)
#             new_y1 = int(y1 + margin_h)
#             new_x2 = int(x2 - margin_w)
#             new_y2 = int(y2 - margin_h)
            
#             # 4. 이미지 경계 내로 클리핑 (안전장치)
#             h_img, w_img = img.shape[:2]
#             new_x1, new_y1 = max(0, new_x1), max(0, new_y1)
#             new_x2, new_y2 = min(w_img, new_x2), min(h_img, new_y2)

#             # 5. 유효성 검사 (영역이 너무 작아져서 0이 된 경우 등)
#             if new_x1 >= new_x2 or new_y1 >= new_y2:
#                 return img

#             # 6. 중앙 영역 자르기 (Crop)
#             bbox_crop = img[new_y1:new_y2, new_x1:new_x2]
            
#             # 7. 자른 영역에만 SunFlare 증강 적용
#             # (영역이 작으므로 src_radius 등 파라미터가 민감할 수 있어 기본값 사용 권장)
#             augmented_crop = self.flare_transform(image=bbox_crop)['image']
            
#             # 8. 원본 이미지에 붙여넣기
#             img_copy = img.copy()
#             img_copy[new_y1:new_y2, new_x1:new_x2] = augmented_crop
#             return img_copy

#         except Exception as e:
#             warnings.warn(f"BBox SunFlare 적용 실패: {e}. 원본 이미지 반환.")
#             return img

#     def __call__(self, results: dict) -> dict:
#         """MMPose 파이프라인에서 호출됩니다."""
        
#         # 1. 전체 증강 블록을 적용할지 확률적으로 결정
#         if np.random.rand() > self.p:
#             return results
            
#         img = results['img']
        
#         # 2. BBox 정보 가져오기
#         # 'GetBBoxCenterScale' 이후 'bbox' 키가 있어야 함
#         if 'bbox' not in results:
#             warnings.warn(
#                 "SPNAugmentation: 'bbox'가 results에 없습니다. "
#                 "RandomSunFlare가 전체 이미지에 적용될 수 있습니다."
#             )
#             bbox = None
#         else:
#             # `results['bbox']`는 [x1, y1, w, h] 형식일 수 있습니다.
#             # `albumentations`는 [x1, y1, x2, y2] 형식이 필요합니다.
#             # `GetBBoxCenterScale`은 'bbox'를 [x1, y1, x2, y2]로 저장합니다.
#             bbox = results['bbox'] 

#         # 3. N개의 증강을 무작위로 선택
#         chosen_indices = np.random.choice(
#             len(self.aug_pool), self.n, replace=False
#         )
        
#         # 4. 선택된 증강을 순차적으로 적용
#         for idx in chosen_indices:
#             aug_fn = self.aug_pool[idx]
            
#             # 5. 선택된 증강이 RandomSunFlare인지 확인
#             if aug_fn is self.flare_transform:
#                 if bbox is not None:
#                     # BBox가 있으면 BBox 내부에만 적용
#                     img = self._apply_flare_to_bbox(img, bbox)
#                 else:
#                     # BBox가 없으면 (경고 후) 전체 이미지에 적용
#                     img = aug_fn(image=img)['image']
#             else:
#                 # 다른 모든 "간단한" 증강은 전체 이미지에 적용
#                 img = aug_fn(image=img)['image']
        
#         results['img'] = img
#         return results