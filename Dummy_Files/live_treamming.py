import tkinter as tk
from ultralytics import YOLO
import cv2
import os
import time
import random
from PIL import Image, ImageTk

# Load your trained YOLO model
model = YOLO("C:/Users/spgir/OneDrive/Documents/ewaste/yolo_training/runs/detect/train2/weights/best.pt")

# Classes of interest
class_names = [
    '9V Battery', 'Battery', 'HDD', 'Keyboard', 'NetworkSwitch',
    'Printed Circuit Board PCB', 'Remote control', 'Router',
    'Smart Phone', 'USB Flash Drive', 'cable', 'computer mouse', 'internal HDD'
]

# Create directories for saving detected objects if they don't exist
os.makedirs("./detected_objects", exist_ok=True)

# Dictionary to keep track of all saved image paths for each class
saved_images = {class_name: [] for class_name in class_names}

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Change index if necessary (0, 1, or 2 depending on your setup)

#cap = cv2.VideoCapture("http://192.0.0.2:8080/video")  # Change index if necessary (0, 1, or 2 depending on your setup)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Create the main window for camera feed
camera_window = tk.Tk()
camera_window.title("Camera Feed")
camera_window.geometry("650x500")
camera_window.configure(bg="#f0f0f0")

# Create the classification window
classification_window = tk.Toplevel(camera_window)  # Create a separate window for classification
classification_window.title("Classified Objects")
classification_window.geometry("1200x1080")

# Create a frame for the camera feed
camera_frame = tk.Frame(camera_window, bg='black')
camera_frame.pack(fill=tk.BOTH, expand=True)

# Create a label for the camera feed
camera_label = tk.Label(camera_frame)
camera_label.pack(fill=tk.BOTH, expand=True)

# Create frames for detected classes in the classification window
frames = {}
frame_width = 300  # Width of the frames for detected classes
frame_height = 250  # Height of the frames for detected classes

# Initialize frames for each class
for i, class_name in enumerate(class_names):
    frame = tk.Frame(classification_window, bg='white', borderwidth=2, relief="groove")
    frame.grid(row=i // 5, column=(i % 5), padx=10, pady=10, sticky='nsew')
    
    label = tk.Label(frame, text=class_name.replace('-', ' ').capitalize(), font=("Arial", 10, 'bold'))
    label.pack(pady=(5, 0))
    
    img_label = tk.Label(frame, bg='white', width=frame_width, height=frame_height)
    img_label.pack(pady=(0, 5))
    
    frames[class_name] = {
        "img_label": img_label,
        "detected": tk.Label(frame, text="Not Detected", font=("Arial", 10, 'italic'), fg="red")
    }
    frames[class_name]["detected"].pack(pady=(5, 0))  # Initially show "Not Detected"

# Function to update the video frame and perform detection
def update_frame():
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture image.")
        return

    # Run the trained YOLO model on the frame
    results = model.predict(source=frame, save=False, show=False)

    # Get the class IDs from the detected boxes
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

    detected_objects = []

    # Check for specific object classes in the frame
    for class_id in class_ids:
        if class_id < len(class_names):
            obj_class = class_names[class_id]
            detected_objects.append(obj_class)

            # Save the frame in the corresponding directory
            timestamp = int(time.time())  # For unique file naming
            img_save_path = f"./detected_objects/{obj_class}_{timestamp}.jpg"

            # Save the new image
            cv2.imwrite(img_save_path, frame)
            saved_images[obj_class].append(img_save_path)  # Store the saved image path

            # Update the class frame to show the detected image
            random_image_path = img_save_path  # Use the saved path directly
            try:
                img = Image.open(random_image_path)
                img = img.resize((frame_width, frame_height), Image.LANCZOS)  # Resize to fit the frame
                img_tk = ImageTk.PhotoImage(image=img)

                frames[obj_class]["img_label"].config(image=img_tk)
                frames[obj_class]["img_label"].image = img_tk  # Keep a reference

                # Update detected message
                frames[obj_class]["detected"].config(text="Detected", fg="green")
            except Exception as e:
                print(f"Error displaying image for {obj_class}: {e}")  # Error handling

    # Update frames for classes not detected
    for class_name in class_names:
        if class_name not in detected_objects:
            frames[class_name]["img_label"].config(image='')  # Clear the image
            frames[class_name]["detected"].config(text="Not Detected", fg="red")  # Update message

    # Convert the frame to a format that Tkinter can use
    annotated_frame = results[0].plot()
    img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
    img_tk = ImageTk.PhotoImage(image=img)

    # Display the annotated frame in the camera feed label
    camera_label.config(image=img_tk)
    camera_label.image = img_tk  # Keep a reference

    # Call this function again after 10 ms
    camera_label.after(10, update_frame)

# Function to clean up detected objects
def cleanup_detected_objects():
    detected_folder = "./detected_objects"
    for filename in os.listdir(detected_folder):
        file_path = os.path.join(detected_folder, filename)
        try:
            os.remove(file_path)
            print(f"Deleted image: {file_path}")  # Debug message
        except Exception as e:
            print(f"Error deleting image {file_path}: {e}")

# Cleanup function to be called when the classification window is closed
def on_classification_window_close():
    cleanup_detected_objects()  # Clean up detected images
    classification_window.destroy()  # Close the classification window

# Bind the window close event to the cleanup function
classification_window.protocol("WM_DELETE_WINDOW", on_classification_window_close)

# Start the update loop
update_frame()

# Configure grid weights for responsive design
for i in range(5):
    classification_window.grid_columnconfigure(i, weight=1)

for i in range((len(class_names) + 4) // 5):  # Use ceiling division
    classification_window.grid_rowconfigure(i, weight=1)

# Start the Tkinter main loop
camera_window.mainloop()

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
