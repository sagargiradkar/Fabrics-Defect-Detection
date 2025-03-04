from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Load a model
    model = YOLO("yolo11n.pt")

    # Training parameters
    data_path = "C:/Users/spgir/OneDrive/Documents/ewaste/E-waste_detection/data.yaml"
    epochs = 100
    imgsz = 640  # Reduced image size
    batch_size = 4  # Reduced batch size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use CUDA if available

    # Train the model
    train_results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=imgsz,
        device=device,
        batch=batch_size,
        half=True,  # Enable mixed precision
    )

    print("Training complete.")

