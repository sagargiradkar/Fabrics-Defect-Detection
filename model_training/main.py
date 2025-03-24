from utils.device_manager import DeviceManager
from trainer.yolo_trainer import YOLOTrainer
from config.training_config import TrainingConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        logger.info("Clearing CUDA memory before starting training")
        DeviceManager.clear_cuda_memory()

        logger.info("Initializing YOLOTrainer")
        config = TrainingConfig()
        trainer = YOLOTrainer(config)
        
        logger.info("Loading YOLO model")
        trainer.load_model()

        logger.info("Starting training process")
        trainer.train()
    except Exception as e:
        logger.error(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
