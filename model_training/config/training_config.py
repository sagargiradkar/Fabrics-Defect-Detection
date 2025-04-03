# config/training_config.py
import logging

class TrainingConfig:
    """Configuration settings for YOLO model training."""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Model settings
    MODEL_PATH = "yolo11s.pt"  # Switched to YOLOv11s for better accuracy
    DATA_YAML_PATH = "D:/Fabrics-Defect-Detection/dataset_8970/data.yaml"
    
    # Training hyperparameters
    EPOCHS = 200
    IMAGE_SIZE = 800  # Increased for better defect detection
    BATCH_SIZE = 32
    ENABLE_MIXED_PRECISION = True

    # Learning Rate Scheduling
    LR0 = 0.005  # Lowered LR for stability
    LRF = 0.2
    MOMENTUM = 0.937
    WEIGHT_DECAY = 0.01  # Increased for better regularization
    WARMUP_EPOCHS = 5  # Increased warmup

    # Early Stopping (Optional)
    EARLY_STOPPING_PATIENCE = 10  # Increased patience

    # Device configuration
    FORCE_CPU = False

    # Logging and Checkpoints
    SAVE_INTERVAL = 5
    LOGGING_INTERVAL = 200

    # Validation Settings
    VAL_INTERVAL = 1
    
    # Augmentation parameters
    AUG_DEGREES = 10.0
    AUG_TRANSLATE = 0.1
    AUG_SCALE = 0.5
    AUG_FLIPLR = 0.7  # Higher probability for flipping
    AUG_FLIPUD = 0.3  # Enabled vertical flip
    AUG_MOSAIC = 1.0
    AUG_MIXUP = 0.5  # Increased mixup
    AUG_COPY_PASTE = 0.2  # More copy-paste augmentation
    AUG_HSV_H = 0.015
    AUG_HSV_S = 0.7
    AUG_HSV_V = 0.4
