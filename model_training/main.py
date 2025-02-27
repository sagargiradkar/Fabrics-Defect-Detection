# main.py
from utils.device_manager import DeviceManager
from trainer.yolo_trainer import YOLOTrainer
from config.training_config import TrainingConfig

def main():
    """Main execution function."""
    try:
        # Clear CUDA memory before starting
        DeviceManager.clear_cuda_memory()
        
        # Initialize trainer
        trainer = YOLOTrainer(TrainingConfig)
        
        # Load and train model
        trainer.load_model()
        results = trainer.train()
        
        # Process results if needed
        if results:
            print("Training completed successfully!")
            # Add any additional results processing here
            
    except Exception as e:
        print(f"An error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()