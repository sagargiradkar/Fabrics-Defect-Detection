# ui/classification_window.py
import tkinter as tk
from config.settings import Settings

class ClassificationWindow:
    """Window displaying classified objects"""
    
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title(f"{Settings.PROJECT_NAME} - Classified Objects")
        self.window.geometry(Settings.CLASSIFICATION_WINDOW_SIZE)
        self.frames = {}
        self._setup_frames()

    def _setup_frames(self):
        """Setup frames for each class"""
        for i, class_name in enumerate(Settings.CLASS_NAMES):
            frame = tk.Frame(self.window, bg='white', borderwidth=2, relief="groove")
            frame.grid(row=i // 5, column=(i % 5), padx=10, pady=10, sticky='nsew')
            
            label = tk.Label(frame, text=class_name.replace('-', ' ').capitalize(), 
                           font=("Arial", 10, 'bold'))
            label.pack(pady=(5, 0))
            
            img_label = tk.Label(frame, bg='white', width=Settings.FRAME_WIDTH, 
                               height=Settings.FRAME_HEIGHT)
            img_label.pack(pady=(0, 5))
            
            self.frames[class_name] = {
                "img_label": img_label,
                "detected": tk.Label(frame, text="Not Detected", 
                                   font=("Arial", 10, 'italic'), fg="red")
            }
            self.frames[class_name]["detected"].pack(pady=(5, 0))

        self._configure_grid()

    def _configure_grid(self):
        """Configure grid layout"""
        for i in range(5):
            self.window.grid_columnconfigure(i, weight=1)
        for i in range((len(Settings.CLASS_NAMES) + 4) // 5):
            self.window.grid_rowconfigure(i, weight=1)

    def update_detection(self, class_name, image=None, detected=False):
        """Update the display for a detected or non-detected class"""
        if detected:
            self.frames[class_name]["img_label"].config(image=image)
            self.frames[class_name]["img_label"].image = image
            self.frames[class_name]["detected"].config(text="Detected", fg="green")
        else:
            self.frames[class_name]["img_label"].config(image='')
            self.frames[class_name]["detected"].config(text="Not Detected", fg="red")
