# config/settings.py (Configuration settings)
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/model_training/runs/detect/train/weights/best.pt"
IMAGE_FOLDER = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/dataset_8970/test/images"
CLASS_NAMES = ['Hole', 'Stitch', 'seam', 'Thread_other']
CLASS_MAPPING = {i: name for i, name in enumerate(CLASS_NAMES)}
DETECTED_FOLDER = "./detected_objects"

# core/model.py (Model loading and processing)
from ultralytics import YOLO
import cv2
import os
import random
from config.settings import DEVICE, MODEL_PATH, CLASS_MAPPING, DETECTED_FOLDER

os.makedirs(DETECTED_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH).to(DEVICE)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print("Model loaded successfully on GPU!")

def process_images(folder_path):
    detected_files = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            results = model.predict(source=image, save=False, show=False, device=DEVICE)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes else []

            valid_classes = [CLASS_MAPPING[class_id] for class_id in class_ids if class_id in CLASS_MAPPING]
            for obj_class in valid_classes:
                img_save_path = f"{DETECTED_FOLDER}/{obj_class}_{random.randint(0,9999)}.jpg"
                cv2.imwrite(img_save_path, image)
                detected_files.append((obj_class, img_save_path))
    
    return detected_files