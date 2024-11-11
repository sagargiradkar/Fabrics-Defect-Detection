# main.py
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from models.yolo_model import LiveEWasteDetector
from utils.camera import Camera
from utils.image_processing import ImageProcessor
from ui.camera_window import CameraWindow
from ui.classification_window import ClassificationWindow
from config.settings import Settings

class LiveEWasteDetectionApp:
    def __init__(self, camera_source=None):
        self.detector = LiveEWasteDetector()
        self.camera = Camera(camera_source)
        self.camera_window = CameraWindow()
        self.classification_window = ClassificationWindow(self.camera_window.window)
        self.image_processor = ImageProcessor()
        
        # Commented out because method is not defined
        # self._setup_window_handlers()
        
        self._setup_camera_controls()

    def _setup_camera_controls(self):
        """Add camera control buttons to the camera window"""
        control_frame = tk.Frame(self.camera_window.window)
        control_frame.pack(pady=5)

        tk.Button(
            control_frame, 
            text="Use Laptop Camera",
            command=lambda: self._switch_camera('LAPTOP')
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            control_frame, 
            text="Use IP Camera",
            command=lambda: self._switch_camera('IP_CAMERA')
        ).pack(side=tk.LEFT, padx=5)

    def _switch_camera(self, source):
        """Switch between camera sources"""
        try:
            self.camera.switch_camera(source)
            print(f"Switched to {source}")
        except Exception as e:
            print(f"Error switching camera: {e}")

    def _update_frame(self):
        """Capture frame, make predictions, and update the UI"""
        try:
            frame = self.camera.read_frame()
            detections = self.detector.predict(frame)
            detected_classes = self.detector.get_detected_classes(detections)

            # Update camera feed in UI
            self.camera_window.update_camera_feed(frame)

            # Update detected objects in classification window
            for class_name in Settings.CLASS_NAMES:
                if class_name in detected_classes:
                    img_path = self.image_processor.save_detected_object(frame, class_name)
                    display_img = self.image_processor.create_display_image(img_path)
                    self.classification_window.update_detection(class_name, display_img, detected=True)
                else:
                    self.classification_window.update_detection(class_name, detected=False)
        except RuntimeError as e:
            print(f"Error capturing frame: {e}")
            # Optionally, stop the loop if capturing fails
            return

        # Schedule the next frame update
        self.camera_window.window.after(Settings.FRAME_RATE, self._update_frame)

    def run(self):
        """Start the application with continuous frame updates"""
        self._update_frame()
        self.camera_window.window.mainloop()

# Usage example:
if __name__ == "__main__":
    # Start with laptop camera
    app = LiveEWasteDetectionApp(camera_source='LAPTOP')
    app.run()
