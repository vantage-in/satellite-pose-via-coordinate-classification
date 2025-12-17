import numpy as np
from mmpose.registry import TRANSFORMS

@TRANSFORMS.register_module()
class SetFullImageBBox:
    """
    Set entire region of the image as BBox for GetBBoxCenterScale
    when the images are already cropped and resized.
    """
    def __init__(self):
        pass

    def __call__(self, results):
        img_shape = results['img_shape'] # (h, w)
        h, w = img_shape[:2]
        
        # [x1, y1, x2, y2] = [0, 0, w, h]
        results['bbox'] = np.array([[0, 0, w, h]], dtype=np.float32)
        results['bbox_score'] = np.ones(1, dtype=np.float32)
        
        return results