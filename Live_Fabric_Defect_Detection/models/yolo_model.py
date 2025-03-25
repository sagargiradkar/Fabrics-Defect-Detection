# models/yolo_model.py
from ultralytics import YOLO
import cv2
from config.settings import Settings

class LiveEWasteDetector:
    """YOLO model wrapper for live defrct"""
    
    def __init__(self):
        self.model = YOLO(Settings.MODEL_PATH)
        self.class_names = Settings.CLASS_NAMES

    def predict(self, frame):
        """Perform prediction on a single frame"""
        results = self.model.predict(source=frame, save=False, show=False)
        return results[0]

    def get_detected_classes(self, results):
        """Extract detected class names from results"""
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        detected_objects = []
        for class_id in class_ids:
            if class_id < len(self.class_names):
                detected_objects.append(self.class_names[class_id])
        return detected_objects