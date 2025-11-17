"""
Configuration file for cough detection transfer learning
"""
import os

# Audio processing parameters
SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN = 50
FMAX = 14000

# Model parameters
AUDIOSET_CLASSES_NUM = 527

# Training parameters
BATCH_SIZE = 32
NUM_WORKERS = 8
DEVICE = 'cuda'

# Stage-specific configurations (2-Stage Approach)
STAGE_CONFIGS = {
    'stage1_coughvid': {
        'name': 'COUGHVID',
        'classes_num': 2,  # cough vs no-cough
        'freeze_blocks': 2,  # Freeze first 2 conv blocks
        'learning_rate': 1e-4,
        'epochs': 10,
        'description': 'Learn cough fundamentals from COUGHVID dataset'
    },
    'stage2_icbhi': {
        'name': 'ICBHI',
        'classes_num': 7,  # Multiple disease classes
        'freeze_blocks': 1,  # Freeze first conv block only
        'learning_rate': 5e-5,  # Higher LR for better adaptation
        'epochs': 15,
        'description': 'Multi-class disease classification on ICBHI dataset'
    }
}

# Dataset paths (to be configured by user)
DATASET_PATHS = {
    'coughvid': None,  # Set your COUGHVID dataset path
    'icbhi': None      # Set your ICBHI dataset path
}

# Pretrained model path
PRETRAINED_MODEL_PATH = "pretrained/Cnn14_mAP=0.431.pth"  # Download from Zenodo

# Workspace directory
WORKSPACE = "workspace"

# Logging
LOG_INTERVAL = 200
SAVE_INTERVAL = 2000
