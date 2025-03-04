import torch

def check_gpu():
    """Check if GPU is available and print details."""
    if torch.cuda.is_available():
        print("✅ GPU is available!")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        print("❌ GPU is NOT available. Using CPU.")

if __name__ == "__main__":
    check_gpu()
