# trainer/yolo_trainer.py
from ultralytics import YOLO
import logging
from config.training_config import TrainingConfig
from utils.device_manager import DeviceManager

class YOLOTrainer:
    """Handles YOLO model training operations."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer with configuration settings.
        
        Args:
            config: Training configuration object
        """
        self.config = config
        self.model = None
        self.device = DeviceManager.get_device()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load the YOLO model."""
        try:
            self.model = YOLO(self.config.MODEL_PATH)
            self.logger.info(f"Model loaded successfully from {self.config.MODEL_PATH}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def train(self):
        """Execute the training process."""
        if self.model is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            return None
        
        try:
            self.logger.info("Starting training with the following configuration:")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Image size: {self.config.IMAGE_SIZE}")
            self.logger.info(f"Batch size: {self.config.BATCH_SIZE}")
            self.logger.info(f"Epochs: {self.config.EPOCHS}")
            
            results = self.model.train(
                data=self.config.DATA_YAML_PATH,
                epochs=self.config.EPOCHS,
                imgsz=self.config.IMAGE_SIZE,
                device=self.device,
                batch=self.config.BATCH_SIZE,
                half=self.config.ENABLE_MIXED_PRECISION
            )
            
            self.logger.info("Training completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise