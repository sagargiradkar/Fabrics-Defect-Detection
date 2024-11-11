import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ultralytics import YOLO
import cv2
import os
import random
import torch
from PIL import Image, ImageTk
import yaml
import threading

class EWasteDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("E-Waste Detection System")
        self.root.geometry("1900x1080")
        
        # Class names
        self.class_names = ['9V Battery', 'Battery', 'HDD', 'Keyboard', 'NetworkSwitch',
                           'Printed Circuit Board PCB', 'Remote control', 'Router', 
                           'Smart Phone', 'USB Flash Drive', 'cable', 'computer mouse', 
                           'internal HDD']
        
        # Create main frames
        self.create_control_panel()
        self.create_detection_panel()
        
        # Initialize variables
        self.model = None
        self.training_thread = None
        
        # Create detected objects directory
        os.makedirs("./detected_objects", exist_ok=True)

    def create_control_panel(self):
        """Create the control panel with training and detection options"""
        control_frame = ttk.LabelFrame(self.root, text="Control Panel", padding="10")
        control_frame.pack(fill="x", padx=10, pady=5)

        # Training section
        train_frame = ttk.LabelFrame(control_frame, text="Training Settings", padding="5")
        train_frame.pack(fill="x", pady=5)

        # Model path selection
        ttk.Label(train_frame, text="Model Path:").grid(row=0, column=0, padx=5, pady=2)
        self.model_path_var = tk.StringVar()
        ttk.Entry(train_frame, textvariable=self.model_path_var).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(train_frame, text="Browse", command=self.browse_model).grid(row=0, column=2, padx=5, pady=2)

        # Data yaml selection
        ttk.Label(train_frame, text="Data YAML:").grid(row=1, column=0, padx=5, pady=2)
        self.data_path_var = tk.StringVar()
        ttk.Entry(train_frame, textvariable=self.data_path_var).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(train_frame, text="Browse", command=self.browse_yaml).grid(row=1, column=2, padx=5, pady=2)

        # Training parameters
        ttk.Label(train_frame, text="Epochs:").grid(row=2, column=0, padx=5, pady=2)
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(train_frame, textvariable=self.epochs_var).grid(row=2, column=1, padx=5, pady=2)

        ttk.Label(train_frame, text="Batch Size:").grid(row=3, column=0, padx=5, pady=2)
        self.batch_size_var = tk.StringVar(value="4")
        ttk.Entry(train_frame, textvariable=self.batch_size_var).grid(row=3, column=1, padx=5, pady=2)

        # Train button
        self.train_button = ttk.Button(train_frame, text="Start Training", command=self.start_training)
        self.train_button.grid(row=4, column=0, columnspan=3, pady=10)

        # Detection section
        detect_frame = ttk.LabelFrame(control_frame, text="Detection Settings", padding="5")
        detect_frame.pack(fill="x", pady=5)

        # Image folder selection
        ttk.Label(detect_frame, text="Image Folder:").grid(row=0, column=0, padx=5, pady=2)
        self.image_folder_var = tk.StringVar()
        ttk.Entry(detect_frame, textvariable=self.image_folder_var).grid(row=0, column=1, padx=5, pady=2)
        ttk.Button(detect_frame, text="Browse", command=self.browse_image_folder).grid(row=0, column=2, padx=5, pady=2)

        # Detect button
        self.detect_button = ttk.Button(detect_frame, text="Start Detection", command=self.start_detection)
        self.detect_button.grid(row=1, column=0, columnspan=3, pady=10)

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", pady=5)

    def create_detection_panel(self):
        """Create the panel for displaying detected objects"""
        detection_frame = ttk.LabelFrame(self.root, text="Detected Objects", padding="10")
        detection_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Create frames for each class
        self.frames = {}
        frame_width = 300
        frame_height = 250

        for i, class_name in enumerate(self.class_names):
            frame = ttk.Frame(detection_frame, borderwidth=2, relief="groove")
            frame.grid(row=i // 5, column=i % 5, padx=10, pady=10, sticky='nsew')
            
            label = ttk.Label(frame, text=class_name.replace('-', ' ').capitalize())
            label.pack(pady=(5, 0))
            
            img_label = ttk.Label(frame, width=frame_width, height=frame_height)
            img_label.pack(pady=(0, 5))
            
            self.frames[class_name] = img_label

        # Configure grid weights
        for i in range(5):
            detection_frame.grid_columnconfigure(i, weight=1)
        for i in range(len(self.class_names) // 5 + 1):
            detection_frame.grid_rowconfigure(i, weight=1)

    def browse_model(self):
        filename = filedialog.askopenfilename(filetypes=[("PT files", "*.pt")])
        if filename:
            self.model_path_var.set(filename)

    def browse_yaml(self):
        filename = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml"), ("YML files", "*.yml")])
        if filename:
            self.data_path_var.set(filename)

    def browse_image_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.image_folder_var.set(folder)

    def start_training(self):
        if not all([self.model_path_var.get(), self.data_path_var.get()]):
            messagebox.showerror("Error", "Please select model and data paths")
            return

        # Disable buttons during training
        self.train_button.state(['disabled'])
        self.detect_button.state(['disabled'])

        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self.train_model)
        self.training_thread.start()

    def train_model(self):
        try:
            torch.cuda.empty_cache()
            model = YOLO(self.model_path_var.get())
            
            # Training parameters
            train_results = model.train(
                data=self.data_path_var.get(),
                epochs=int(self.epochs_var.get()),
                imgsz=640,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                batch=int(self.batch_size_var.get()),
                half=True
            )
            
            self.root.after(0, lambda: messagebox.showinfo("Success", "Training completed successfully!"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Training failed: {str(e)}"))
        finally:
            self.root.after(0, lambda: self.train_button.state(['!disabled']))
            self.root.after(0, lambda: self.detect_button.state(['!disabled']))

    def start_detection(self):
        if not self.image_folder_var.get():
            messagebox.showerror("Error", "Please select an image folder")
            return

        # Load the model if not already loaded
        if self.model is None:
            try:
                self.model = YOLO(self.model_path_var.get())
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                return

        self.detect_button.state(['disabled'])
        threading.Thread(target=self.process_images).start()

    def process_images(self):
        try:
            folder_path = self.image_folder_var.get()
            detected_files = []

            # Clean up previous detections
            self.cleanup_detected_images()

            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, filename)
                    
                    # Read and process image
                    image = cv2.imread(img_path)
                    results = self.model.predict(source=image, save=False, show=False)
                    
                    # Process detections
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                    for class_id in class_ids:
                        if class_id < len(self.class_names):
                            obj_class = self.class_names[class_id]
                            img_save_path = f"./detected_objects/{obj_class}_{filename}"
                            cv2.imwrite(img_save_path, image)
                            detected_files.append((obj_class, img_save_path))

            # Update UI with detected images
            self.root.after(0, lambda: self.display_random_images(detected_files))
            self.root.after(0, lambda: self.detect_button.state(['!disabled']))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection failed: {str(e)}"))
            self.root.after(0, lambda: self.detect_button.state(['!disabled']))

    def display_random_images(self, detected_files):
        random.shuffle(detected_files)
        
        for obj_class, img_path in detected_files:
            try:
                img = Image.open(img_path)
                img = img.resize((300, 250), Image.LANCZOS)
                img_tk = ImageTk.PhotoImage(image=img)
                
                self.frames[obj_class].configure(image=img_tk)
                self.frames[obj_class].image = img_tk
            except Exception as e:
                print(f"Error displaying image for {obj_class}: {e}")

    def cleanup_detected_images(self):
        for filename in os.listdir("./detected_objects"):
            file_path = os.path.join("./detected_objects", filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    def on_closing(self):
        if self.training_thread and self.training_thread.is_alive():
            if messagebox.askokcancel("Quit", "Training is in progress. Do you want to quit anyway?"):
                self.cleanup_detected_images()
                self.root.destroy()
        else:
            self.cleanup_detected_images()
            self.root.destroy()

def main():
    root = tk.Tk()
    app = EWasteDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()