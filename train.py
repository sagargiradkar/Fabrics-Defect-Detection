from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Load a model
    model = YOLO("yolo11n.pt")

    # Training parameters
    data_path = "/home/pavan/ewaste/E-waste_detection/data.yaml"
    epochs = 200
    imgsz = 1024  # Reduced image size
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

    # Optional: Evaluate model performance on the validation set
    metrics = model.val()

    # Optional: Perform object detection on an image or directory of images
    results = model("/home/pavan/ewaste/E-waste_detection/test/images/")
    results[0].show()

    # Optional: Export the model to ONNX format
    path = model.export(format="onnx")

    print("Training and evaluation complete.")

