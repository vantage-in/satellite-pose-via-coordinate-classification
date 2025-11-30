import os
import numpy as np
import mmcv
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadImageFromDualDir(BaseTransform):
    """
    두 개의 디렉토리 중 하나를 랜덤하게 선택하여 이미지를 로드합니다.
    원본 이미지와 보조 이미지의 확장자가 달라도 파일명(stem)이 같으면 매칭합니다.
    (예: image_01.jpg <-> image_01.png)
    
    Args:
        aux_dir (str): 두 번째(대체) 이미지 디렉토리의 절대 경로
        prob (float): aux_dir에서 이미지를 로드할 확률 (0.0 ~ 1.0). 기본값 0.5.
        aux_suffix (str): 보조 이미지의 확장자 (예: '.png'). None이면 원본 확장자 유지.
    """
    def __init__(self, 
                 aux_dir: str, 
                 prob: float = 0.8,
                 aux_suffix: str = '.png', # [추가] 보조 데이터의 확장자 지정
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2'):
        self.aux_dir = aux_dir
        self.prob = prob
        self.aux_suffix = aux_suffix
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend

    def transform(self, results: dict) -> dict:
        original_path = results['img_path']
        filename = os.path.basename(original_path)
        
        # [수정된 로직] 확장자 교체 처리
        if self.aux_suffix is not None:
            # 파일명에서 확장자 분리 ('image_01', '.jpg')
            file_stem, _ = os.path.splitext(filename)
            # 새 확장자 붙이기 ('image_01.png')
            aux_filename = file_stem + self.aux_suffix
        else:
            aux_filename = filename

        # 1. 보조 디렉토리의 예상 경로 생성
        aux_path = os.path.join(self.aux_dir, aux_filename)
        
        # 2. 로직: "확률 당첨" AND "파일이 실제로 존재함"
        if np.random.rand() < self.prob and os.path.exists(aux_path):
            filepath = aux_path
            results['img_source'] = 'aux'
        else:
            filepath = original_path
            results['img_source'] = 'original'

        # 3. 이미지 로드 (mmcv 사용)
        try:
            img = mmcv.imread(
                filepath,
                flag=self.color_type,
                backend=self.imdecode_backend
            )
            if img is None: raise FileNotFoundError(f"Image read failed: {filepath}")
        except Exception as e:
            # 로드 실패 시 원본으로 폴백
            # print(f"Warning: Failed to load {filepath}, using original. {e}")
            filepath = original_path
            img = mmcv.imread(filepath, flag=self.color_type, backend=self.imdecode_backend)

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        
        # 디버깅 및 추적용 경로 업데이트
        results['img_path'] = filepath 
        
        return results