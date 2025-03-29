# config/training_config.py
import logging

class TrainingConfig:
    """Configuration settings for YOLO model training."""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Model settings
    MODEL_PATH = "yolo11n.pt"
    DATA_YAML_PATH = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/dataset_8970/data.yaml"
    
    # Training hyperparameters
    EPOCHS = 200
    IMAGE_SIZE = 640
    BATCH_SIZE = 32
    ENABLE_MIXED_PRECISION = True

    # Learning Rate Scheduling
    LR0 = 0.01
    LRF = 0.2
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.0005
    WARMUP_EPOCHS = 3

    # Early Stopping (Optional)
    EARLY_STOPPING_PATIENCE = 7
    
    # Device configuration
    FORCE_CPU = False

    # Logging and Checkpoints
    SAVE_INTERVAL = 5
    LOGGING_INTERVAL = 200

    # Validation Settings
    VAL_INTERVAL = 1
    
    # Augmentation parameters
    AUG_DEGREES = 10.0  # Rotation range in degrees (0-180)
    AUG_TRANSLATE = 0.1  # Translation range (0-1)
    AUG_SCALE = 0.5  # Scale range (0-1)
    AUG_FLIPLR = 0.5  # Horizontal flip probability (0-1)
    AUG_FLIPUD = 0.0  # Vertical flip probability (0-1)
    AUG_MOSAIC = 1.0  # Mosaic augmentation probability (0-1)
    AUG_MIXUP = 0.1  # Mixup augmentation probability (0-1)
    AUG_COPY_PASTE = 0.1  # Copy-paste augmentation probability (0-1)
    AUG_HSV_H = 0.015  # HSV hue augmentation range (0-1)
    AUG_HSV_S = 0.7  # HSV saturation augmentation range (0-1)
    AUG_HSV_V = 0.4  # HSV value augmentation range (0-1)