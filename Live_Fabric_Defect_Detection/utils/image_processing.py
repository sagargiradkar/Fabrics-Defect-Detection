# utils/image_processing.py
import cv2
import os
import time
from PIL import Image, ImageTk
from config.settings import Settings

class ImageProcessor:
    """Handle image processing and storage operations"""
    
    @staticmethod
    def save_detected_object(frame, class_name):
        """Save detected object frame to disk"""
        timestamp = int(time.time())
        img_save_path = os.path.join(Settings.DETECTED_OBJECTS_DIR, 
                                    f"{class_name}_{timestamp}.jpg")
        cv2.imwrite(img_save_path, frame)
        return img_save_path

    @staticmethod
    def create_display_image(image_path):
        """Create a Tkinter-compatible image for display"""
        img = Image.open(image_path)
        img = img.resize((Settings.FRAME_WIDTH, Settings.FRAME_HEIGHT), Image.LANCZOS)
        return ImageTk.PhotoImage(image=img)

    @staticmethod
    def cleanup_detected_objects():
        """Remove all saved detected object images"""
        for filename in os.listdir(Settings.DETECTED_OBJECTS_DIR):
            file_path = os.path.join(Settings.DETECTED_OBJECTS_DIR, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting image {file_path}: {e}")