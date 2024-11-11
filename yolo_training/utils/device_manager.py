# utils/device_manager.py
import torch
from config.training_config import TrainingConfig

class DeviceManager:
    """Manages device-related operations and memory."""
    
    @staticmethod
    def get_device():
        """Determine the available device (CUDA or CPU)."""
        if TrainingConfig.FORCE_CPU:
            return 'cpu'
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @staticmethod
    def clear_cuda_memory():
        """Clear CUDA cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")