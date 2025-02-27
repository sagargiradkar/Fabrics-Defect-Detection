import tkinter as tk
from ultralytics import YOLO
import cv2
import os
import random
from PIL import Image, ImageTk

# Load your trained YOLO model
model = YOLO("/home/pavan/ewaste/runs/detect/train19/weights/best.pt")

# Classes of interest
class_names = ['Battery', 'Blood-Pressure-Monitor', 'Boiler', 'Clothes-Iron', 'Coffee-Machine', 'Computer-Keyboard', 'Computer-Mouse', 'Cooling-Display', 'Desktop-PC', 'Digital-Oscilloscope', 'Drone', 'Electric-Guitar', 'Electronic-Keyboard', 'Flashlight', 'Flat-Panel-Monitor', 'Flat-Panel-TV', 'Glucose-Meter', 'HDD', 'Laptop', 'Microwave', 'Music-Player', 'Oven', 'PCB', 'Photovoltaic-Panel', 'Projector', 'Refrigerator', 'Rotary-Mower', 'Router', 'Server', 'Smartphone', 'Smoke-Detector', 'Straight-Tube-Fluorescent-Lamp', 'Street-Lamp', 'TV-Remote-Control', 'Telephone-Set', 'USB-Flash-Drive', 'Washing-Machine']

# Create directories for saving detected objects if they don't exist
os.makedirs("./detected_objects", exist_ok=True)

# Create the classification window
classification_window = tk.Tk()
classification_window.title("Classified Objects")
classification_window.geometry("1900x1080")

# Create frames for detected classes in the classification window
frames = {}
frame_width = 300  # Adjust this to the desired width of each block
frame_height = 250  # Adjust this to the desired height of each block

for i, class_name in enumerate(class_names):
    frame = tk.Frame(classification_window, bg='white', borderwidth=2, relief="groove")
    frame.grid(row=i // 5, column=(i % 5), padx=10, pady=10, sticky='nsew')
    
    label = tk.Label(frame, text=class_name.replace('-', ' ').capitalize(), font=("Arial", 10, 'bold'))
    label.pack(pady=(5, 0))
    
    img_label = tk.Label(frame, bg='white', width=frame_width, height=frame_height)
    img_label.pack(pady=(0, 5))
    
    frames[class_name] = img_label

# Function to process images from a folder
def process_images(folder_path):
    detected_files = []  # To store paths of saved images

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            print(f"Processing image: {img_path}")

            # Read the image
            image = cv2.imread(img_path)
            results = model.predict(source=image, save=False, show=False)

            # Get the class IDs from the detected boxes
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            # Check for specific object classes in the frame
            for class_id in class_ids:
                if class_id < len(class_names):
                    obj_class = class_names[class_id]

                    # Save the frame in the corresponding directory
                    img_save_path = f"./detected_objects/{obj_class}_{filename}"
                    cv2.imwrite(img_save_path, image)
                    detected_files.append((obj_class, img_save_path))  # Store class and path

    # Display random images from the detected objects
    display_random_images(detected_files)

# Function to display random images from the detected objects
def display_random_images(detected_files):
    random.shuffle(detected_files)  # Shuffle the list to select random images

    # Display images for all classes
    for obj_class, img_path in detected_files:
        try:
            img = Image.open(img_path)
            img = img.resize((frame_width, frame_height), Image.LANCZOS)  # Resize to fit the frame
            img_tk = ImageTk.PhotoImage(image=img)

            frames[obj_class].config(image=img_tk)
            frames[obj_class].image = img_tk  # Keep a reference to avoid garbage collection

            print(f"Displayed image for detected object: {obj_class}")  # Debug message
        except Exception as e:
            print(f"Error displaying image for {obj_class}: {e}")  # Error handling

# Function to delete all images from the detected_objects directory
def cleanup_detected_images():
    detected_folder = "./detected_objects"
    for filename in os.listdir(detected_folder):
        file_path = os.path.join(detected_folder, filename)
        try:
            os.remove(file_path)
            print(f"Deleted image: {file_path}")  # Debug message
        except Exception as e:
            print(f"Error deleting image {file_path}: {e}")

# Cleanup function to be called when the window is closed
def on_closing():
    cleanup_detected_images()  # Clean up detected images
    classification_window.destroy()  # Close the window

# Bind the window close event to the cleanup function
classification_window.protocol("WM_DELETE_WINDOW", on_closing)

# Specify the folder containing the images to test
folder_path = "/home/pavan/ewaste/E-waste_detection/test/images"  # Change this to your image folder path
process_images(folder_path)

# Configure grid weights for responsive design
for i in range(5):
    classification_window.grid_columnconfigure(i, weight=1)

for i in range(len(class_names) // 5 + 1):
    classification_window.grid_rowconfigure(i, weight=1)

# Start the Tkinter main loop
classification_window.mainloop()
