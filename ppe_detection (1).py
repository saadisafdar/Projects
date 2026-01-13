import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os

class PPEDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PPE Image Detection using YOLOv8")
        self.root.geometry("900x800")
        self.root.configure(bg="#f0f0f0")

        self.model = None
        self.model_loaded = False
        self.image_path = None

        self.create_gui()

    def create_gui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", pady=15)
        header_frame.pack(fill="x")
        
        title = tk.Label(header_frame, text="PPE Image Detection using YOLOv8", 
                        font=("Helvetica", 18, "bold"), fg="white", bg="#2c3e50")
        title.pack()
        
        subtitle = tk.Label(header_frame, text="AI Lab / University Project", 
                           font=("Helvetica", 12), fg="#ecf0f1", bg="#2c3e50")
        subtitle.pack()

        # Training Workflow Note
        training_frame = tk.LabelFrame(self.root, text="Training Workflow (Recommended on Google Colab)", 
                                      font=("Helvetica", 10, "bold"), padx=10, pady=10)
        training_frame.pack(fill="x", padx=20, pady=10)
        
        training_text = (
            "Training is recommended on Google Colab due to hardware limitations.\n\n"
            "Dataset follows YOLO format:\n"
            "dataset/\n"
            "├── data.yaml\n"
            "├── train/images\n"
            "├── train/labels\n"
            "├── valid/images\n"
            "├── valid/labels\n"
            "├── test/images\n"
            "├── test/labels"
        )
        tk.Label(training_frame, text=training_text, justify="left", bg="#f0f0f0").pack(anchor="w")

        # Model Section
        model_frame = tk.LabelFrame(self.root, text="Model Loading", font=("Helvetica", 10, "bold"), padx=10, pady=10)
        model_frame.pack(fill="x", padx=20, pady=10)
        
        tk.Button(model_frame, text="Load trained_model.pt", command=self.load_model, 
                 bg="#3498db", fg="white", font=("Helvetica", 10)).pack(side="left", padx=10)
        
        self.model_status_label = tk.Label(model_frame, text="Not Loaded ❌", fg="red", font=("Helvetica", 10, "bold"))
        self.model_status_label.pack(side="left")

        # Image Selection Section
        img_frame = tk.LabelFrame(self.root, text="Image Selection", font=("Helvetica", 10, "bold"), padx=10, pady=10)
        img_frame.pack(fill="both", padx=20, pady=10, expand=True)
        
        tk.Button(img_frame, text="Select Image", command=self.select_image,
                 bg="#27ae60", fg="white", font=("Helvetica", 10)).pack(pady=5)
        
        self.original_label = tk.Label(img_frame, text="No image selected", bg="white", width=50, height=20)
        self.original_label.pack(pady=10)

        # Detection Section
        detect_frame = tk.LabelFrame(self.root, text="Detection", font=("Helvetica", 10, "bold"), padx=10, pady=10)
        detect_frame.pack(fill="both", padx=20, pady=10, expand=True)
        
        tk.Button(detect_frame, text="Run Detection", command=self.run_detection,
                 bg="#e74c3c", fg="white", font=("Helvetica", 10)).pack(pady=5)
        
        self.output_label = tk.Label(detect_frame, text="Detection output will appear here", bg="white", width=50, height=20)
        self.output_label.pack(pady=10)

        # Results Panel
        results_frame = tk.LabelFrame(self.root, text="Detection Results", font=("Helvetica", 10, "bold"), padx=10, pady=10)
        results_frame.pack(fill="both", padx=20, pady=10, expand=True)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=8, font=("Helvetica", 10))
        self.results_text.pack(fill="both", expand=True)

        # Status Bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief="sunken", anchor="w", bg="#bdc3c7")
        self.status_bar.pack(fill="x", side="bottom")

    def load_model(self):
        model_path = "trained_model.pt"
        if not os.path.exists(model_path):
            self.status_bar.config(text="Error: trained_model.pt not found in the script folder.")
            self.model_status_label.config(text="Not Loaded ❌", fg="red")
            return
        
        try:
            self.model = YOLO(model_path)
            self.model_loaded = True
            self.model_status_label.config(text="Loaded ✅", fg="green")
            self.status_bar.config(text="Model loaded successfully.")
        except Exception as e:
            self.model_status_label.config(text="Error ❌", fg="red")
            self.status_bar.config(text=f"Model loading error: {str(e)}")

    def select_image(self):
        self.image_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if self.image_path:
            # Display original image
            img = Image.open(self.image_path)
            img = self.resize_to_fit(img, 450, 350)
            self.original_photo = ImageTk.PhotoImage(img)
            self.original_label.config(image=self.original_photo, text="")
            self.status_bar.config(text="Image selected.")
            
            # Clear previous results
            self.output_label.config(image="", text="Detection output will appear here")
            self.results_text.delete(1.0, tk.END)

    def run_detection(self):
        if not self.model_loaded:
            self.status_bar.config(text="Error: Load the model first.")
            return
        if not self.image_path:
            self.status_bar.config(text="Error: Select an image first.")
            return
        
        self.status_bar.config(text="Running detection...")
        self.root.update()
        
        try:
            results = self.model(self.image_path)[0]
            
            # Annotated image
            annotated = results.plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            annotated_pil = Image.fromarray(annotated)
            annotated_pil = self.resize_to_fit(annotated_pil, 450, 350)
            self.annotated_photo = ImageTk.PhotoImage(annotated_pil)
            self.output_label.config(image=self.annotated_photo, text="")
            
            # Text results
            self.results_text.delete(1.0, tk.END)
            if results.boxes is not None and len(results.boxes) > 0:
                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = results.names[cls_id]
                    conf = box.conf[0].item()
                    self.results_text.insert(tk.END, f"{cls_name}: {conf:.2%}\n")
            else:
                self.results_text.insert(tk.END, "No PPE objects detected.\n")
            
            self.status_bar.config(text="Detection completed.")
        except Exception as e:
            self.status_bar.config(text=f"Detection error: {str(e)}")

    def resize_to_fit(self, img, max_width, max_height):
        img.thumbnail((max_width, max_height), Image.LANCZOS)
        return img

if __name__ == "__main__":
    root = tk.Tk()
    app = PPEDetectionApp(root)
    root.mainloop()