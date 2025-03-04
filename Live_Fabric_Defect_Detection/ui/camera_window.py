# ui/camera_window.py
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from config.settings import Settings

class CameraWindow:
    """Main window displaying the camera feed"""
    
    def __init__(self):
        self.window = tk.Tk()
        self.window.title(f"{Settings.PROJECT_NAME} - Camera Feed")
        self.window.geometry(Settings.CAMERA_WINDOW_SIZE)
        self.window.configure(bg="#f0f0f0")

        self._setup_ui()

    def _setup_ui(self):
        """Setup UI components"""
        self.camera_frame = tk.Frame(self.window, bg='black')
        self.camera_frame.pack(fill=tk.BOTH, expand=True)

        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack(fill=tk.BOTH, expand=True)

    def update_camera_feed(self, frame):
        """Update the camera feed display"""
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img)
        self.camera_label.config(image=img_tk)
        self.camera_label.image = img_tk