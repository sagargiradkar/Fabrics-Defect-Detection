# ui/app.py (Tkinter UI and gallery display)
import tkinter as tk
from tkinter import Scrollbar, Canvas, Frame, Label
from PIL import Image, ImageTk
from core.model import process_images
from config.settings import IMAGE_FOLDER

def run_app():
    classification_window = tk.Tk()
    classification_window.title("Classified Objects Gallery")
    classification_window.geometry("1800x900")
    
    canvas = Canvas(classification_window)
    scrollbar = Scrollbar(classification_window, orient="vertical", command=canvas.yview)
    scrollable_frame = Frame(canvas)
    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    image_references = []
    detected_files = process_images(IMAGE_FOLDER)
    
    row, col, max_columns = 0, 0, 5
    for obj_class, img_path in detected_files:
        try:
            img = Image.open(img_path)
            img = img.resize((250, 250), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img)
            label = Label(scrollable_frame, image=img_tk, text=obj_class, compound="top", font=("Arial", 10, "bold"))
            label.grid(row=row, column=col, padx=10, pady=10)
            image_references.append(img_tk)
            col += 1
            if col >= max_columns:
                col = 0
                row += 1
        except Exception as e:
            print(f"Error displaying image for {obj_class}: {e}")
    
    classification_window.mainloop()
