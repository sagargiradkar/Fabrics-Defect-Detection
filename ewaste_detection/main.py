import os
import random
from config.config import Config
from core.detector import ObjectDetector
from core.image_processor import ImageProcessor
from ui.gui import ClassificationGUI

def main():
    # Initialize components
    detector = ObjectDetector(Config.MODEL_PATH)
    image_processor = ImageProcessor(Config.DETECTED_OBJECTS_DIR, Config.CLASS_NAMES)
    gui = ClassificationGUI(Config.CLASS_NAMES, Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
    
    def process_folder():
        detected_files = []
        
        # Process each image in the test folder
        for filename in os.listdir(Config.TEST_IMAGES_PATH):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(Config.TEST_IMAGES_PATH, filename)
                image, class_ids = detector.detect_objects(img_path)
                
                # Save detected objects
                for class_id in class_ids:
                    if class_id < len(Config.CLASS_NAMES):
                        obj_class = Config.CLASS_NAMES[class_id]
                        saved_path = image_processor.save_detected_image(
                            image, filename, obj_class)
                        detected_files.append((obj_class, saved_path))
        
        # Display detected images
        random.shuffle(detected_files)
        for obj_class, img_path in detected_files:
            resized_img = image_processor.resize_image(
                img_path, Config.FRAME_WIDTH, Config.FRAME_HEIGHT)
            gui.update_image(obj_class, resized_img)
    
    def cleanup():
        image_processor.cleanup_images()
        gui.window.destroy()
    
    # Configure GUI
    gui.configure_grid(len(Config.CLASS_NAMES))
    gui.set_cleanup_handler(cleanup)
    
    # Process images and start GUI
    process_folder()
    gui.start()

if __name__ == "__main__":
    main()