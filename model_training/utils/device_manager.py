import torch
import logging
from config.training_config import TrainingConfig

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DeviceManager:
    """Manages device-related operations and memory."""

    @staticmethod
    def get_device():
        """Determine the available device (CUDA or CPU) and log it."""
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
        """Clear CUDA cache if available and log the action."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared.")
