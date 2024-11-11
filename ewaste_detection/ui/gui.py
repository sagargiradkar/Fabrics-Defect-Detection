# ewaste_detection/ui/gui.py
import tkinter as tk
from PIL import ImageTk

class ClassificationGUI:
    def __init__(self, class_names, frame_width, frame_height):
        self.window = tk.Tk()
        self.window.title("Classified Objects")
        self.window.geometry("1900x1080")
        self.frames = {}
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.setup_frames(class_names)
        
    def setup_frames(self, class_names):
        for i, class_name in enumerate(class_names):
            frame = tk.Frame(self.window, bg='white', borderwidth=2, relief="groove")
            frame.grid(row=i // 5, column=(i % 5), padx=10, pady=10, sticky='nsew')
            
            label = tk.Label(frame, text=class_name.replace('-', ' ').capitalize(), 
                           font=("Arial", 10, 'bold'))
            label.pack(pady=(5, 0))
            
            img_label = tk.Label(frame, bg='white', width=self.frame_width, 
                               height=self.frame_height)
            img_label.pack(pady=(0, 5))
            
            self.frames[class_name] = img_label
    
    def update_image(self, class_name, image):
        try:
            img_tk = ImageTk.PhotoImage(image=image)
            self.frames[class_name].config(image=img_tk)
            self.frames[class_name].image = img_tk
        except Exception as e:
            print(f"Error updating image for {class_name}: {e}")
    
    def configure_grid(self, num_classes):
        for i in range(5):
            self.window.grid_columnconfigure(i, weight=1)
        for i in range(num_classes // 5 + 1):
            self.window.grid_rowconfigure(i, weight=1)
    
    def set_cleanup_handler(self, cleanup_func):
        self.window.protocol("WM_DELETE_WINDOW", cleanup_func)
    
    def start(self):
        self.window.mainloop()