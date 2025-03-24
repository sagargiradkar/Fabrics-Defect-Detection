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

        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {device_name} (Total GPUs available: {device_count})")
            return 'cuda'
        else:
            logger.info("CUDA not available. Using CPU.")
            return 'cpu'

    @staticmethod
    def clear_cuda_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
