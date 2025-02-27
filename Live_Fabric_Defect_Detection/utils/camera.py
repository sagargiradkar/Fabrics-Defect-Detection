# utils/camera.py
import cv2
from config.settings import Settings

class Camera:
    """Handle video capture from multiple camera sources"""
    
    def __init__(self, source=None):
        """
        Initialize camera with specified source
        Args:
            source: Either 'LAPTOP' or 'IP_CAMERA' or None (uses default)
        """
        self.source = source or Settings.DEFAULT_CAMERA
        self.cap = self._initialize_camera()

    def _initialize_camera(self):
        """Initialize the camera with the selected source, with fallback to alternate source if needed"""
        camera_source = Settings.CAMERA_SOURCES[self.source]
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            # Try the alternative source if the primary one fails
            alternative_source = 'IP_CAMERA' if self.source == 'LAPTOP' else 'LAPTOP'
            alternative_camera_source = Settings.CAMERA_SOURCES[alternative_source]
            cap = cv2.VideoCapture(alternative_camera_source)
            
            if not cap.isOpened():
                raise RuntimeError("Error: Could not open any video source. Please check the camera connections and settings.")
            else:
                print(f"Warning: Primary camera source ({self.source}) failed, using alternative source ({alternative_source})")
        
        return cap

    def read_frame(self):
        """Read a single frame from the video source"""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Error: Failed to capture image")
        return frame

    def release(self):
        """Release the video capture resource"""
        self.cap.release()

    def switch_camera(self, new_source):
        """
        Switch to a different camera source
        Args:
            new_source: Either 'LAPTOP' or 'IP_CAMERA'
        """
        if new_source in Settings.CAMERA_SOURCES:
            self.release()
            self.source = new_source
            self.cap = self._initialize_camera()
        else:
            raise ValueError(f"Invalid camera source. Choose from: {list(Settings.CAMERA_SOURCES.keys())}")
