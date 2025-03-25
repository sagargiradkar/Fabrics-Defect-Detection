import os
import logging
import torch
import cv2
from ultralytics import YOLO
from PIL import Image, ImageTk
import tkinter as tk

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Settings:
    PROJECT_NAME = "Live Fabric Defect Detector"
    VERSION = "1.0.0"
    MODEL_PATH = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/model_training/runs/detect/train/weights/best.pt"
    CLASS_NAMES = ['Hole', 'Stitch', 'Seam']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CAMERA_SOURCES = {'LAPTOP': 0, 'IP_CAMERA': "http://192.168.195.198:4747/video"}
    DEFAULT_CAMERA = 'LAPTOP'
    FRAME_RATE = 10
    CAMERA_WINDOW_SIZE = "960x540"
    CLASSIFICATION_WINDOW_SIZE = "960x540"

    @staticmethod
    def check_cuda():
        logging.info(f"Using device: {Settings.DEVICE}")
        if torch.cuda.is_available():
            logging.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logging.warning("CUDA not available. Using CPU.")
Settings.check_cuda()

# YOLO Model Class
class LiveFabricDefectDetector:
    def __init__(self):
        self.model = YOLO(Settings.MODEL_PATH)
        self.class_names = Settings.CLASS_NAMES

    def predict(self, frame):
        results = self.model.predict(source=frame, save=False, show=False)
        return results[0]

# Application Class
class LiveFabricDetectionApp:
    def __init__(self):
        self.detector = LiveFabricDefectDetector()
        self.cap = cv2.VideoCapture(Settings.CAMERA_SOURCES[Settings.DEFAULT_CAMERA])
        self.root = tk.Tk()
        self.root.title(Settings.PROJECT_NAME)
        self.root.geometry("1920x540")
        
        self.camera_label = tk.Label(self.root)
        self.camera_label.grid(row=0, column=0)

        self.class_label = tk.Label(self.root)
        self.class_label.grid(row=0, column=1)

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            detections = self.detector.predict(frame)
            for box in detections.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls)
                label = f"{Settings.CLASS_NAMES[class_id]} ({box.conf[0]:.2f})"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert frame for Tkinter display
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)
            self.camera_label.config(image=img_tk)
            self.camera_label.image = img_tk

            # Update classification window
            detected_labels = [Settings.CLASS_NAMES[int(box.cls)] for box in detections.boxes]
            classification_text = "Detected Defects: " + ", ".join(set(detected_labels)) if detected_labels else "No Defects Detected"
            self.class_label.config(text=classification_text, font=("Arial", 14))

        self.root.after(Settings.FRAME_RATE, self.update_frame)

    def run(self):
        logging.info("Starting Fabric Defect Detection...")
        self.root.mainloop()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = LiveFabricDetectionApp()
    app.run()
