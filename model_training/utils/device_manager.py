# utils/device_manager.py
import torch
import logging
from config.training_config import TrainingConfig

logger = logging.getLogger(__name__)

class DeviceManager:
    @staticmethod
    def get_device():
        if TrainingConfig.FORCE_CPU:
            logger.info("FORCE_CPU is enabled. Using CPU.")
            return 'cpu'
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA not available. Using CPU.")
        return device

    @staticmethod
    def clear_cuda_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")