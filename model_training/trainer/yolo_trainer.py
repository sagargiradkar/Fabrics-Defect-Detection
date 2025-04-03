# yolo_trainer.py
from ultralytics import YOLO
import logging
import torch.optim as optim
from config.training_config import TrainingConfig
from utils.device_manager import DeviceManager

class YOLOTrainer:
    """Handles YOLO model training operations."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.device = DeviceManager.get_device()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing YOLOTrainer")
    
    def load_model(self):
        """Loads the YOLO model."""
        try:
            self.logger.info(f"Loading model from {self.config.MODEL_PATH}")
            self.model = YOLO(self.config.MODEL_PATH)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def train(self):
        """Trains the YOLO model with updated configurations."""
        if self.model is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            return None
        
        try:
            self.logger.info("Starting training with updated configuration:")
            self.logger.info(f"Device: {self.device}")
            self.logger.info(f"Image size: {self.config.IMAGE_SIZE}")
            self.logger.info(f"Batch size: {self.config.BATCH_SIZE}")
            self.logger.info(f"Epochs: {self.config.EPOCHS}")
            self.logger.info("Data augmentation enabled")

            # Training with updated configurations
            results = self.model.train(
                data=self.config.DATA_YAML_PATH,
                epochs=self.config.EPOCHS,
                imgsz=self.config.IMAGE_SIZE,
                device=self.device,
                batch=self.config.BATCH_SIZE,
                half=self.config.ENABLE_MIXED_PRECISION,
                patience=self.config.EARLY_STOPPING_PATIENCE,
                lr0=self.config.LR0,  # Lower LR for better stability
                lrf=self.config.LRF,
                momentum=self.config.MOMENTUM,
                weight_decay=self.config.WEIGHT_DECAY,
                warmup_epochs=self.config.WARMUP_EPOCHS,
                cos_lr=True,  # Enables cosine LR scheduler
                optimizer=optim.AdamW(self.model.parameters(), lr=self.config.LR0, weight_decay=0.01),  # AdamW for better convergence
                # Augmentation settings
                augment=True,
                degrees=self.config.AUG_DEGREES,
                translate=self.config.AUG_TRANSLATE,
                scale=self.config.AUG_SCALE,
                fliplr=self.config.AUG_FLIPLR,
                flipud=self.config.AUG_FLIPUD,
                mosaic=self.config.AUG_MOSAIC,
                mixup=self.config.AUG_MIXUP,
                copy_paste=self.config.AUG_COPY_PASTE,
                hsv_h=self.config.AUG_HSV_H,
                hsv_s=self.config.AUG_HSV_S,
                hsv_v=self.config.AUG_HSV_V,
            )
            
            self.logger.info("Training completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise
