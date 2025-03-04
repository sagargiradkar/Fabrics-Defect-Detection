# config/training_config.py
class TrainingConfig:
    """Configuration settings for YOLO model training."""
    
    # Model settings
    MODEL_PATH = "yolo11n.pt"
    DATA_YAML_PATH = "C:/Users/vlabs/Desktop/ewaste/Fabric_Defect_5Class/data.yaml"
    
    # Training hyperparameters
    EPOCHS = 10
    IMAGE_SIZE = 640
    BATCH_SIZE = 4
    ENABLE_MIXED_PRECISION = True
    
    # Device configuration
    FORCE_CPU = False  # Set to True to force CPU usage even if CUDA is available
