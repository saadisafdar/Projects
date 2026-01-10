import tkinter as tk
from tkinter import filedialog, messagebox
from ultralytics import YOLO
import os
from PIL import Image, ImageTk
import cv2
import numpy as np

class HeadGearSentinel:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HeadGear Sentinel")
        self.model = None
        self.classes = ['Person', 'Helmet', 'Goggles', 'Mask', 'Gloves', 'Safety Vest', 'Boots']

        # GUI layout
        tk.Button(self.root, text="Load Dataset", command=self.load_dataset).pack()
        tk.Button(self.root, text="Train Model", command=self.train_model).pack()
        tk.Button(self.root, text="Load Image(s)", command=self.load_images).pack()
        tk.Button(self.root, text="Run Detection", command=self.run_detection).pack()

        self.result_text = tk.Text(self.root, height=10, width=50)
        self.result_text.pack()

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.root.mainloop()

    def load_dataset(self):
        self.dataset_path = filedialog.askdirectory()
        # Create a YAML config for YOLOv8
        with open('dataset.yaml', 'w') as f:
            f.write(f"path: {self.dataset_path}\ntrain: .\nval: .\nnames:\n")
            for i, c in enumerate(self.classes):
                f.write(f"  {i}: {c}\n")
        messagebox.showinfo("Dataset Loaded", "Dataset loaded successfully!")

    def train_model(self):
        if not self.dataset_path:
            messagebox.showerror("Error", "Load dataset first!")
            return
        self.model = YOLO('yolov8n.pt')
        self.model.train(data='dataset.yaml', imgsz=640, epochs=10)
        messagebox.showinfo("Training Done", "Model trained successfully!")

    def load_images(self):
        self.image_paths = filedialog.askopenfilenames(filetypes=[("Image files", ".jpg .png")])
        if self.image_paths:
            self.display_image(self.image_paths[0])

    def display_image(self, path):
        img = Image.open(path)
        img.thumbnail((400, 400))
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img

    def run_detection(self):
        if not self.model:
            messagebox.showerror("Error", "Train model first!")
            return
        results = []
        for path in self.image_paths:
            result = self.model(path)
            results.append((path, result))
            # Annotate image
            img = cv2.imread(path)
            for r in result:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(f"annotated_{os.basename(path)}", img)
        
        # Process results
        output = ""
        for path, result in results:
            detected = set()
            for r in result:
                for c in r.boxes.cls:
                    detected.add(self.classes[int(c)])
            output += f"{path}: {', '.join(detected)}\n"
            safety_status = "SAFE" if 'Person' in detected and all(c in detected for c in self.classes[1:]) else "UNSAFE"
            output += f"Safety Status: {safety_status}\n\n"
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, output)

if __name__ == "__main__":
    HeadGearSentinel()