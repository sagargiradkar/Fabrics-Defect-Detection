# trainer/yolo_trainer.py
from ultralytics import YOLO
import logging
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
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
        try:
            self.logger.info(f"Loading model from {self.config.MODEL_PATH}")
            self.model = YOLO(self.config.MODEL_PATH)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def train(self):
        if self.model is None:
            self.logger.error("Model not loaded. Call load_model() first.")
            return None
        
        try:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.LR0,
                momentum=self.config.MOMENTUM,
                weight_decay=self.config.WEIGHT_DECAY
            )

            self.logger.info("Initializing OneCycleLR Scheduler")
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.config.LR0,
                steps_per_epoch=len(self.model.train_loader),
                epochs=self.config.EPOCHS,
                pct_start=0.2,
                anneal_strategy='cos',
                final_div_factor=1000
            )

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
                half=self.config.ENABLE_MIXED_PRECISION,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                val_period=self.config.VAL_INTERVAL
            )
            
            self.logger.info("Training completed successfully")
            return results
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise