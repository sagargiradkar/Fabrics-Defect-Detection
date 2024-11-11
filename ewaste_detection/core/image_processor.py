# ewaste_detection/core/image_processor.py
from PIL import Image
import os
import cv2
import random

class ImageProcessor:
    def __init__(self, output_dir, class_names):
        self.output_dir = output_dir
        self.class_names = class_names
        os.makedirs(output_dir, exist_ok=True)
        
    def save_detected_image(self, image, filename, class_name):
        save_path = os.path.join(self.output_dir, f"{class_name}_{filename}")
        cv2.imwrite(save_path, image)
        return save_path
    
    def resize_image(self, image_path, width, height):
        img = Image.open(image_path)
        return img.resize((width, height), Image.LANCZOS)
    
    def cleanup_images(self):
        for filename in os.listdir(self.output_dir):
            try:
                os.remove(os.path.join(self.output_dir, filename))
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
