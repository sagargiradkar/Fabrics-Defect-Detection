import tkinter as tk
from ultralytics import YOLO
import cv2
import os
import random
import torch  # GPU check
from PIL import Image, ImageTk

# ✅ Force GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ✅ Load YOLO model on GPU
model = YOLO("C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/model_training/runs/detect/train/weights/best.pt").to(device)
print("Model loaded successfully on GPU!")

# ✅ Define class names (44 classes)
class_names =['Hole', 'Stitch', 'seam','Thread_other']

# ✅ Map class IDs to names dynamically
class_mapping = {i: name for i, name in enumerate(class_names)}

# ✅ Create directory for detected objects
os.makedirs("./detected_objects", exist_ok=True)

# ✅ Create Tkinter classification window
classification_window = tk.Tk()
classification_window.title("Classified Objects Gallery")
classification_window.geometry("1800x900")

# ✅ Create a scrollable canvas
canvas = tk.Canvas(classification_window)
scrollbar = tk.Scrollbar(classification_window, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

# ✅ Store image references (prevent garbage collection)
image_references = []

# ✅ Function to process images
def process_images(folder_path):
    detected_files = []
    
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            print(f"Processing image: {img_path}")

            # ✅ Read the image using OpenCV CUDA (if available)
            image = cv2.imread(img_path)
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(image)
                image = gpu_frame.download()

            # ✅ Run YOLO detection on GPU
            results = model.predict(source=image, save=False, show=False, device=device)

            # ✅ Extract class IDs (on GPU)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int) if results[0].boxes else []
            print(f"Detected class IDs in {filename}: {class_ids}")  # Debug output

            valid_classes = []
            for class_id in class_ids:
                if class_id in class_mapping:
                    obj_class = class_mapping[class_id]
                    valid_classes.append(obj_class)
                else:
                    print(f"Warning: Class ID {class_id} not found in mapping!")

            # ✅ Save detected objects if valid
            for obj_class in valid_classes:
                img_save_path = f"./detected_objects/{obj_class}_{random.randint(0,9999)}.jpg"
                cv2.imwrite(img_save_path, image)
                detected_files.append((obj_class, img_save_path))
                print(f"Saved detected object as {img_save_path}")

    display_gallery(detected_files)

# ✅ Function to display images in gallery format
def display_gallery(detected_files):
    global image_references
    image_references.clear()

    row, col = 0, 0
    max_columns = 5  # Number of images per row

    for obj_class, img_path in detected_files:
        try:
            img = Image.open(img_path)
            img = img.resize((250, 250), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img)

            label = tk.Label(scrollable_frame, image=img_tk, text=obj_class, compound="top", font=("Arial", 10, "bold"))
            label.grid(row=row, column=col, padx=10, pady=10)

            image_references.append(img_tk)  # Keep reference

            col += 1
            if col >= max_columns:
                col = 0
                row += 1
        except Exception as e:
            print(f"Error displaying image for {obj_class}: {e}")

# ✅ Cleanup detected images on close
def cleanup_detected_images():
    detected_folder = "./detected_objects"
    for filename in os.listdir(detected_folder):
        file_path = os.path.join(detected_folder, filename)
        try:
            os.remove(file_path)
            print(f"Deleted image: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

# ✅ Cleanup when closing the window
def on_closing():
    cleanup_detected_images()
    classification_window.destroy()

classification_window.protocol("WM_DELETE_WINDOW", on_closing)

# ✅ Process images in the folder
folder_path = "C:/Users/vlabs/Desktop/Fabrics-Defect-Detection/defect-dataset/test/images"
process_images(folder_path)

# ✅ Start Tkinter main loop
classification_window.mainloop()
