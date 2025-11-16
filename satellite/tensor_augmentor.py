# my_combined_augmentor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from typing import List, Optional, Tuple

from mmpose.registry import MODELS
from mmpose.utils.typing import DataSample

# ----------------------------------------------------------------------
# 의존성 임포트 (라이브러리 설치 필요)
# ----------------------------------------------------------------------
try:
    from style-augmentation.styleaug import StyleAugmentor
except ImportError:
    raise ImportError(
        'StyleAugment 라이브러리가 필요합니다. '
        'pip install git+https://github.com/roimehrez/style-augmentation.git'
    )

try:
    # DeepAugment(CAE) 모델을 임포트합니다.
    # 'CAE_Model' 디렉터리가 PYTHONPATH에 있거나 이 파일과 같은 위치에 있어야 합니다.
    from CAE_Model.cae_32x32x32_zero_pad_bin import CAE
except ImportError:
    raise ImportError(
        'CAE 모델을 임포트할 수 없습니다. '
        'imagenet-r 레포지토리의 "DeepAugment/CAE_Model" 디렉터리를 '
        'PYTHONPATH에 추가하거나 이 스크립트 위치로 복사하세요.'
    )

# ----------------------------------------------------------------------
# 1. RandConv (lib/networks/rand_conv.py) 모듈 (그대로 복사)
# ----------------------------------------------------------------------
class _RandConvImpl(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias, padding_mode)

    def forward(self, x):
        weight = torch.randn_like(self.weight)
        weight = F.normalize(weight, dim=[1, 2, 3], p=2)
        if self.bias is not None:
            bias = torch.randn_like(self.bias)
            bias = F.normalize(bias, dim=0, p=2)
        else:
            bias = None
        return F.conv2d(x, weight, bias, self.stride, self.padding,
                        self.dilation, self.groups)


# ----------------------------------------------------------------------
# 2. 통합 증강 래퍼 (Wrapper)
# ----------------------------------------------------------------------
@MODELS.register_module()
class CombinedAugmentation(nn.Module):
    """
    StyleAugment, DeepAugment(CAE), RandConv를 순차적으로 적용하는
    통합 텐서 증강 래퍼.
    
    Args:
        mean (list[float]): rtmpose.py의 data_preprocessor mean 값
        std (list[float]): rtmpose.py의 data_preprocessor std 값
        prob_style (float): StyleAugment 적용 확률
        prob_deep (float): DeepAugment(CAE) 적용 확률
        prob_randconv (float): RandConv 적용 확률
        cae_weights_path (str): DeepAugment CAE 모델 가중치(.state) 파일 경로
        deepaug_sigma (float): DeepAugment 잠재 공간 노이즈 강도
        randconv_kernel_size (int): RandConv 커널 크기
    """

    def __init__(self,
                 mean: List[float] = [123.675, 116.28, 103.53],
                 std: List[float] = [58.395, 57.12, 57.375],
                 prob_style: float = 0.5,
                 prob_deep: float = 0.5,
                 prob_randconv: float = 0.5,
                 cae_weights_path: str = 'CAE_Weight/model_final.state',
                 deepaug_sigma: float = 0.1,
                 randconv_kernel_size: int = 3):
        super().__init__()
        self.prob_style = prob_style
        self.prob_deep = prob_deep
        self.prob_randconv = prob_randconv
        self.deepaug_sigma = deepaug_sigma

        # 1. 정규화/역정규화용 mean/std 등록 (GPU 이동을 위해)
        self.mean = nn.Parameter(
            torch.tensor(mean).view(1, 3, 1, 1), requires_grad=False)
        self.std = nn.Parameter(
            torch.tensor(std).view(1, 3, 1, 1), requires_grad=False)

        # 2. StyleAugmentor 초기화
        self.style_augmentor = StyleAugmentor()

        # 3. DeepAugment (CAE) 모델 로드
        print(f"Loading CAE weights from: {cae_weights_path}")
        self.cae_encoder, self.cae_decoder = self._load_cae_model(
            cae_weights_path)

        # 4. RandConv 모듈 초기화
        self.rand_conv = _RandConvImpl(
            in_channels=3,
            out_channels=3,
            kernel_size=randconv_kernel_size,
            stride=1,
            padding=randconv_kernel_size // 2,
            dilation=1,
            groups=3,  # Depth-wise
            bias=False,
            padding_mode='reflect')

    def _load_cae_model(self, weights_path):
        """CAE 모델과 가중치를 로드하는 헬퍼 함수"""
        try:
            cae_model = CAE(in_channels=3, n_class=1000)
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # 가중치 파일 형식에 따라 로드
            if 'model_state' in state_dict:
                cae_model.load_state_dict(state_dict['model_state'])
            else:
                cae_model.load_state_dict(state_dict)
                
            cae_model.eval()  # 증강용이므로 eval 모드로 설정
            print("DeepAugment (CAE) model loaded successfully.")
            return cae_model.encoder, cae_model.decoder
            
        except FileNotFoundError:
            raise FileNotFoundError(
                f"CAE weights not found at: {weights_path}. "
                "CombinedAugmentation의 'cae_weights_path'를 수정하세요."
            )
        except Exception as e:
            raise IOError(f"CAE 모델 로드 중 오류 발생: {e}")

    def to(self, *args, **kwargs):
        """모든 하위 모듈(nn.Module)을 올바른 장치(GPU/CPU)로 이동"""
        super().to(*args, **kwargs)
        self.style_augmentor.to(*args, **kwargs)
        self.cae_encoder.to(*args, **kwargs)
        self.cae_decoder.to(*args, **kwargs)
        self.rand_conv.to(*args, **kwargs)
        return self

    def forward(
        self, inputs: torch.Tensor, data_samples: Optional[List[DataSample]]
    ) -> Tuple[torch.Tensor, Optional[List[DataSample]]]:

        # 훈련 모드에서만 증강 적용
        if not self.training:
            return inputs, data_samples

        device = inputs.device
        B = inputs.shape[0]

        # --- 1. 역정규화 (Mean/Std -> 0~1) ---
        #의 mean/std 값을 사용하여 (0, 255) 범위로 복구
        inputs_unnormalized = inputs * self.std + self.mean
        # (0, 1) 범위로 변경 (StyleAugmentor, CAE 입력 형식)
        current_tensor = (inputs_unnormalized / 255.0).clamp(0.0, 1.0)

        # --- 2. (Online) Style Augmentation 적용 ---
        mask_style = (torch.rand(B, 1, 1, 1, device=device) < self.prob_style).float()
        if mask_style.sum() > 0:
            style_aug_tensor = self.style_augmentor(current_tensor)
            current_tensor = mask_style * style_aug_tensor + (1 - mask_style) * current_tensor

        # --- 3. (Online) Deep Augment (CAE) 적용 ---
        mask_deep = (torch.rand(B, 1, 1, 1, device=device) < self.prob_deep).float()
        if mask_deep.sum() > 0:
            # 인코더 통과 (z = E(x))
            z = self.cae_encoder(current_tensor)
            # 잠재 공간 노이즈 추가 (z' = z + delta)
            delta = torch.randn_like(z) * self.deepaug_sigma
            z_perturbed = z + delta
            # 디코더 통과 (x' = D(z'))
            deep_aug_tensor = self.cae_decoder(z_perturbed)
            
            # 원본과 혼합
            current_tensor = mask_deep * deep_aug_tensor + (1 - mask_deep) * current_tensor

        # --- 4. 다시 정규화 (0~1 -> Mean/Std) ---
        current_tensor = (current_tensor * 255.0 - self.mean) / self.std

        # --- 5. (Online) RandConv 적용 ---
        mask_rand = (torch.rand(B, 1, 1, 1, device=device) < self.prob_randconv).float()
        if mask_rand.sum() > 0:
            rand_conv_tensor = self.rand_conv(current_tensor)
            current_tensor = mask_rand * rand_conv_tensor + (1 - mask_rand) * current_tensor

        # --- (디버깅 코드 추가) ---
        if torch.cuda.current_device() == 0: # 0번 GPU에서만 실행
            try:
                import torchvision
                # 텐서 증강이 완료된 첫 번째 이미지를 저장
                # (다시 0~1로 역정규화)
                img_to_save = (current_tensor[0] * self.std + self.mean) / 255.0
                torchvision.utils.save_image(img_to_save.clamp(0.0, 1.0), 
                                             './tensor_aug_check.png')
            except Exception as e:
                print(f"DEBUG: 이미지 저장 실패 {e}")

        return current_tensor, data_samples