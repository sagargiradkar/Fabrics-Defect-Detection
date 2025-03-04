# ewaste_detection/config/config.py
class Config:
    CLASS_NAMES = ['Thread_other', 'Thread_out', 'other', 'stains']
    
    FRAME_WIDTH = 300
    FRAME_HEIGHT = 250
    DETECTED_OBJECTS_DIR = "./detected_objects"
    MODEL_PATH = "C:/Users/vlabs/Desktop/ewaste/model_training/runs/detect/train/weights/best.pt"
    TEST_IMAGES_PATH = "C:/Users/vlabs/Desktop/ewaste/Fabric_Defect_5Class/test/images"
