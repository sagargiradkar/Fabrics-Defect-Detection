# ewaste_detection/config/config.py
class Config:
    CLASS_NAMES = ['9V Battery', 'Battery', 'HDD', 'Keyboard', 'NetworkSwitch',
                   'Printed Circuit Board PCB', 'Remote control', 'Router', 
                   'Smart Phone', 'USB Flash Drive', 'cable', 'computer mouse', 'internal HDD']
    
    FRAME_WIDTH = 300
    FRAME_HEIGHT = 250
    DETECTED_OBJECTS_DIR = "./detected_objects"
    MODEL_PATH = "C:/Users/spgir/OneDrive/Documents/ewaste/yolo_training/runs/detect/train2/weights/best.pt"
    TEST_IMAGES_PATH = "C:/Users/spgir/OneDrive/Documents/ewaste/E-waste_detection/test/images"
