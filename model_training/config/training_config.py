# config/training_config.py
class TrainingConfig:
    """Configuration settings for YOLO model training."""
    
    # Model settings
    MODEL_PATH = "yolo11n.pt"
    DATA_YAML_PATH = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/defect-dataset/data.yaml"
    
    # Training hyperparameters
    EPOCHS = 100  # Increased epochs for better learning
    IMAGE_SIZE = 640
    BATCH_SIZE = 32  # Optimized for 32GB GPU
    ENABLE_MIXED_PRECISION = True

    # Early Stopping (Optional)
    EARLY_STOPPING_PATIENCE = 7  # Stop if no improvement for 7 epochs
    
    # Device configuration
    FORCE_CPU = False  # Set to True to force CPU usage even if CUDA is available

    # Logging and Checkpoints
    SAVE_INTERVAL = 5  # Save model every 5 epochs
    LOGGING_INTERVAL = 50  # Log metrics every 50 batches
