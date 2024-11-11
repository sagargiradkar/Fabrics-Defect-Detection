# ewaste_detection/core/detector.py
import cv2
from ultralytics import YOLO
import os

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def detect_objects(self, image_path):
        image = cv2.imread(image_path)
        results = self.model.predict(source=image, save=False, show=False)
        return image, results[0].boxes.cls.cpu().numpy().astype(int)
