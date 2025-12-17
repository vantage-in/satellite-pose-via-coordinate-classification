import os
import numpy as np
import mmcv
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadImageFromDualDir(BaseTransform):
    """
    Load images randomly from two directories.
    Match via file stem (e.g. image_01.jpg <-> image_01.png)
    
    Args:
        aux_dir (str): Alternative images.
        prob (float): A probability to load a image from aux_dir.
        aux_suffix (str): File extension of aux. None: maintains original extension.
    """
    def __init__(self, 
                 aux_dir: str, 
                 prob: float = 0.75,
                 aux_suffix: str = '.png',
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

        if self.aux_suffix is not None:
            file_stem, _ = os.path.splitext(filename)
            # New extension
            aux_filename = file_stem + self.aux_suffix
        else:
            aux_filename = filename

        aux_path = os.path.join(self.aux_dir, aux_filename)
        
        if np.random.rand() < self.prob and os.path.exists(aux_path):
            filepath = aux_path
            results['img_source'] = 'aux'
        else:
            filepath = original_path
            results['img_source'] = 'original'

        try:
            img = mmcv.imread(
                filepath,
                flag=self.color_type,
                backend=self.imdecode_backend
            )
            if img is None: raise FileNotFoundError(f"Image read failed: {filepath}")
        except Exception as e:
            filepath = original_path
            img = mmcv.imread(filepath, flag=self.color_type, backend=self.imdecode_backend)

        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        
        results['img_path'] = filepath 
        
        return results