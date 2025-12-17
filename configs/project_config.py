import os

# ==============================================================================
# 1. PATH CONFIGURATION (Customize)
# ==============================================================================
# Dataset path
WORKSPACE_ROOT = '/workspace'
DATASET_ROOT = os.path.join(WORKSPACE_ROOT, 'speedplusv2')

MODEL_3D_POINTS_FILE = os.path.join(DATASET_ROOT, 'tangoPoints.mat')
CAMERA_FILE = os.path.join(DATASET_ROOT, 'camera.json')

DATASETS = {
    'lightbox': {
        'img_root': os.path.join(DATASET_ROOT, 'lightbox_preprocessed'),
        'gt_file': os.path.join(DATASET_ROOT, 'lightbox/test.json'),
        'meta_file': os.path.join(DATASET_ROOT, 'annotations/test_lightbox.json'),
    },
    'sunlamp': {
        'img_root': os.path.join(DATASET_ROOT, 'sunlamp_preprocessed'),
        'gt_file': os.path.join(DATASET_ROOT, 'sunlamp/test.json'),
        'meta_file': os.path.join(DATASET_ROOT, 'annotations/test_sunlamp.json'),
    }
}

# ==============================================================================
# 2. MODEL CONFIGURATION
# ==============================================================================
MODELS = {
    'specc-s': {
        'config_file': 'satellite/specc-s.py',
        'checkpoint_file': os.path.join(WORKSPACE_ROOT, 'specc-s/model_specc-s-a5eced28_20251217.pth'),
        'input_size': (224, 224)
    },
    'specc-m': {
        'config_file': 'satellite/specc-m.py',
        'checkpoint_file': os.path.join(WORKSPACE_ROOT, 'specc-m/model_specc-m-8bd94a4a_20251217.pth'), # 예시 경로
        'input_size': (224, 224)
    }
}

# ==============================================================================
# 3. ALGORITHM HYPERPARAMETERS
# ==============================================================================
# Refinement Sigma
DEFAULT_SIGMA = 10.0

# ==============================================================================
# 4. METRIC PARAMETERS
# ==============================================================================
# HIL-Specific Thresholds
THETA_T_NORM = 2.173e-3  # Normalized Translation Threshold
THETA_Q_DEG = 0.169      # Rotation Threshold (Degree)