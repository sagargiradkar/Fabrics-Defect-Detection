# config/training_config.py
import logging

class TrainingConfig:
    """Configuration settings for YOLO model training."""
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Model settings
    MODEL_PATH = "yolo11n.pt"
    DATA_YAML_PATH = "D:/Fabrics-Defect-Detection/dataset_8970/data.yaml"
    
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