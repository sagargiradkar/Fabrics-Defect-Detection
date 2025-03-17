# config/settings.py
import os

class Settings:
    # Project name and version
    PROJECT_NAME = "Live Fabric Defect Detector"
    VERSION = "1.0.0"

    # Model settings
    MODEL_PATH = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/model_training/runs/detect/train/weights/best.pt"
    CLASS_NAMES =  ['Hole', 'Stitch', 'seam', 'Thread_other']

    # Camera settings
    CAMERA_SOURCES = {
        'LAPTOP': 0,  # Default laptop webcam
        'IP_CAMERA': "http://192.168.195.198:4747/video"  # IP camera URL
    }

    DEFAULT_CAMERA = 'LAPTOP'  # Set default camera source
    FRAME_RATE = 10  # milliseconds between frame updates

    # UI settings
    CAMERA_WINDOW_SIZE = "650x500"
    CLASSIFICATION_WINDOW_SIZE = "1200x1080"
    FRAME_WIDTH = 300
    FRAME_HEIGHT = 250

    # Storage settings
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DETECTED_OBJECTS_DIR = os.path.join(BASE_DIR, "detected_objects")
    os.makedirs(DETECTED_OBJECTS_DIR, exist_ok=True)
