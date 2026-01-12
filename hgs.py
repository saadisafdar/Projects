#!/usr/bin/env python3
"""
PPE Detection System - Single File Implementation
Complete system for PPE detection with YOLOv8, Tkinter GUI, and safety compliance.
Runs on CPU with support for training and inference.
Customized for Roboflow PPE dataset with 7 classes.
"""

import os
import sys
import subprocess
import json
import yaml
import threading
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Auto-install missing libraries
required_packages = {
    'ultralytics': 'ultralytics',
    'cv2': 'opencv-python',
    'PIL': 'pillow',
    'yaml': 'pyyaml',
    'numpy': 'numpy',
}

for module, package in required_packages.items():
    try:
        __import__(module)
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


class PPEDetectionSystem:
    """Main PPE Detection System with training and inference capabilities."""
    
    def __init__(self):
        """Initialize the system."""
        self.dataset_path = None
        self.model = None
        self.model_path = None
        self.current_image = None
        self.current_image_path = None
        self.training_in_progress = False
        
        # PPE Configuration - Adapted to your 7-class dataset
        self.required_ppe = {'helmet', 'boots', 'gloves', 'vest'}
        self.violation_classes = {'no helmet', 'no vest'}
        self.class_names = ['boots', 'gloves', 'helmet', 'human', 'no helmet', 'no vest', 'vest']
        
        # Colors for visualization
        self.color_safe = (0, 255, 0)      # Green
        self.color_unsafe = (0, 0, 255)    # Red
        
        self.setup_gui()
    
    def setup_gui(self):
        """Setup the Tkinter GUI."""
        self.root = tk.Tk()
        self.root.title("PPE Detection System - 7 Classes")
        self.root.geometry("1400x900")
        self.root.configure(bg="#f0f0f0")
        
        # Create main frames
        left_frame = tk.Frame(self.root, bg="#f0f0f0")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        right_frame = tk.Frame(self.root, bg="#f0f0f0")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        self.setup_control_panel(left_frame)
        
        # Right panel - Image preview and logs
        self.setup_preview_panel(right_frame)
    
    def setup_control_panel(self, parent):
        """Setup control panel with buttons and inputs."""
        # Title
        title = tk.Label(parent, text="PPE Detection System", font=("Arial", 16, "bold"), bg="#f0f0f0")
        title.pack(pady=10)
        
        # Classes info
        classes_info = tk.Label(parent, text="Classes: boots, gloves, helmet, human,\nno helmet, no vest, vest", 
                               font=("Arial", 9), bg="#f0f0f0", fg="#666")
        classes_info.pack(pady=5)
        
        # Dataset Section
        dataset_frame = tk.LabelFrame(parent, text="Dataset Configuration", font=("Arial", 11, "bold"), bg="#f0f0f0", padx=10, pady=10)
        dataset_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(dataset_frame, text="Select Dataset Folder", command=self.select_dataset, width=30, bg="#4CAF50", fg="white").pack(pady=5)
        
        self.dataset_label = tk.Label(dataset_frame, text="No dataset selected", fg="#666", bg="#f0f0f0")
        self.dataset_label.pack(pady=5)
        
        tk.Button(dataset_frame, text="Validate Dataset", command=self.validate_dataset, width=30, bg="#2196F3", fg="white").pack(pady=5)
        
        # Training Section
        training_frame = tk.LabelFrame(parent, text="Model Training", font=("Arial", 11, "bold"), bg="#f0f0f0", padx=10, pady=10)
        training_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(training_frame, text="Epochs:", bg="#f0f0f0").pack(anchor=tk.W, pady=5)
        self.epochs_var = tk.StringVar(value="10")
        tk.Spinbox(training_frame, from_=1, to=100, textvariable=self.epochs_var, width=10).pack(anchor=tk.W, pady=5)
        
        tk.Label(training_frame, text="Image Size:", bg="#f0f0f0").pack(anchor=tk.W, pady=5)
        self.img_size_var = tk.StringVar(value="640")
        tk.Spinbox(training_frame, from_=320, to=1280, increment=64, textvariable=self.img_size_var, width=10).pack(anchor=tk.W, pady=5)
        
        self.train_button = tk.Button(training_frame, text="Start Training (FROM SCRATCH)", command=self.train_model, width=30, bg="#FF9800", fg="white")
        self.train_button.pack(pady=5)
        
        self.train_status = tk.Label(training_frame, text="Ready to train", fg="#666", bg="#f0f0f0")
        self.train_status.pack(pady=5)
        
        # Model Section
        model_frame = tk.LabelFrame(parent, text="Model Management", font=("Arial", 11, "bold"), bg="#f0f0f0", padx=10, pady=10)
        model_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(model_frame, text="Load Trained Model", command=self.load_model, width=30, bg="#9C27B0", fg="white").pack(pady=5)
        
        self.model_label = tk.Label(model_frame, text="No model loaded", fg="#666", bg="#f0f0f0")
        self.model_label.pack(pady=5)
        
        # Detection Section
        detection_frame = tk.LabelFrame(parent, text="Detection", font=("Arial", 11, "bold"), bg="#f0f0f0", padx=10, pady=10)
        detection_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(detection_frame, text="Select Image", command=self.select_image, width=30, bg="#00BCD4", fg="white").pack(pady=5)
        
        self.image_label = tk.Label(detection_frame, text="No image selected", fg="#666", bg="#f0f0f0")
        self.image_label.pack(pady=5)
        
        tk.Button(detection_frame, text="Run Detection", command=self.run_detection, width=30, bg="#E91E63", fg="white").pack(pady=5)
        
        # Results Section
        results_frame = tk.LabelFrame(parent, text="Safety Status", font=("Arial", 11, "bold"), bg="#f0f0f0", padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.results_text = scrolledtext.ScrolledText(results_frame, height=10, width=40, font=("Courier", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.results_text.config(state=tk.DISABLED)
    
    def setup_preview_panel(self, parent):
        """Setup image preview and log panels."""
        # Image Preview
        preview_frame = tk.LabelFrame(parent, text="Image Preview", font=("Arial", 11, "bold"), bg="#f0f0f0", padx=10, pady=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.image_canvas = tk.Canvas(preview_frame, bg="gray20", width=500, height=500)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Log Panel
        log_frame = tk.LabelFrame(parent, text="System Log", font=("Arial", 11, "bold"), bg="#f0f0f0", padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=40, font=("Courier", 8))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        
        self.log("System initialized. Ready to use.")
    
    def log(self, message: str):
        """Add message to log."""
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        print(message)
    
    def update_results(self, message: str):
        """Update results text."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, message)
        self.results_text.config(state=tk.DISABLED)
    
    def select_dataset(self):
        """Select dataset folder."""
        path = filedialog.askdirectory(title="Select Dataset Root Folder")
        if path:
            self.dataset_path = path
            self.dataset_label.config(text=f"Dataset: {Path(path).name}")
            self.log(f"Dataset selected: {path}")
    
    def validate_dataset(self):
        """Validate dataset structure."""
        if not self.dataset_path:
            messagebox.showwarning("Warning", "Please select a dataset first")
            return
        
        self.log("Validating dataset structure...")
        
        required_dirs = [
            'train/images', 'train/labels',
            'valid/images', 'valid/labels',
            'test/images', 'test/labels'
        ]
        
        dataset_root = Path(self.dataset_path)
        missing = []
        
        for dir_path in required_dirs:
            full_path = dataset_root / dir_path
            if not full_path.exists():
                missing.append(dir_path)
        
        # Check data.yaml
        yaml_path = dataset_root / 'data.yaml'
        if not yaml_path.exists():
            missing.append('data.yaml')
        
        if missing:
            msg = f"Missing components:\n" + "\n".join(missing)
            messagebox.showerror("Validation Failed", msg)
            self.log(f"Dataset validation failed: {msg}")
            return
        
        # Check data.yaml content
        try:
            with open(yaml_path) as f:
                data_yaml = yaml.safe_load(f)
            
            required_keys = ['train', 'val', 'test', 'nc', 'names']
            missing_keys = [k for k in required_keys if k not in data_yaml]
            
            if missing_keys:
                msg = f"data.yaml missing keys: {', '.join(missing_keys)}"
                messagebox.showerror("Validation Failed", msg)
                self.log(msg)
                return
            
            # Count files
            train_imgs = len(list((dataset_root / 'train/images').glob('*')))
            valid_imgs = len(list((dataset_root / 'valid/images').glob('*')))
            test_imgs = len(list((dataset_root / 'test/images').glob('*')))
            
            msg = f"Dataset valid!\n\nClasses: {data_yaml['nc']}\n" \
                  f"Names: {', '.join(data_yaml['names'])}\n" \
                  f"Train images: {train_imgs}\n" \
                  f"Valid images: {valid_imgs}\n" \
                  f"Test images: {test_imgs}"
            
            messagebox.showinfo("Validation Success", msg)
            self.log(msg)
        
        except Exception as e:
            messagebox.showerror("Error", f"Error reading data.yaml: {str(e)}")
            self.log(f"Error: {str(e)}")
    
    def train_model(self):
        """Train model from scratch."""
        if not self.dataset_path:
            messagebox.showwarning("Warning", "Please select a dataset first")
            return
        
        if self.training_in_progress:
            messagebox.showinfo("Info", "Training already in progress")
            return
        
        # Run training in separate thread
        thread = threading.Thread(target=self._train_worker)
        thread.daemon = True
        thread.start()
    
    def _train_worker(self):
        """Worker thread for training."""
        self.training_in_progress = True
        self.train_button.config(state=tk.DISABLED)
        self.train_status.config(text="Training in progress...", fg="#FF5722")
        
        try:
            epochs = int(self.epochs_var.get())
            img_size = int(self.img_size_var.get())
            yaml_path = str(Path(self.dataset_path) / 'data.yaml')
            
            self.log(f"Starting training: epochs={epochs}, img_size={img_size}")
            self.log(f"Dataset YAML: {yaml_path}")
            
            # Initialize YOLOv8n model from scratch (no pretrained weights)
            self.log("Loading YOLOv8n (nano) model...")
            model = YOLO('yolov8n.yaml')  # Load from architecture only
            
            # Train
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=img_size,
                device=0 if self._has_gpu() else 'cpu',
                patience=5,
                save=True,
                verbose=False,
                plots=False,
                batch=-1  # Auto batch size
            )
            
            # Save model
            model_save_path = Path('runs/detect/train/weights/best.pt')
            if model_save_path.exists():
                self.model_path = str(model_save_path)
                self.model = model
                self.model_label.config(text=f"Model: Trained")
                self.log(f"Training completed! Model saved: {self.model_path}")
                messagebox.showinfo("Success", "Training completed successfully!")
                self.train_status.config(text="Training complete", fg="#4CAF50")
            else:
                self.log("Training completed but model file not found")
                self.train_status.config(text="Training failed", fg="#F44336")
        
        except Exception as e:
            self.log(f"Training error: {str(e)}")
            messagebox.showerror("Training Error", str(e))
            self.train_status.config(text="Training failed", fg="#F44336")
        
        finally:
            self.training_in_progress = False
            self.train_button.config(state=tk.NORMAL)
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def load_model(self):
        """Load trained model."""
        path = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        
        if path:
            try:
                self.log(f"Loading model: {path}")
                self.model = YOLO(path)
                self.model_path = path
                self.model_label.config(text=f"Model: {Path(path).name}")
                self.log("Model loaded successfully")
                messagebox.showinfo("Success", "Model loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.log(f"Model load error: {str(e)}")
    
    def select_image(self):
        """Select image for detection."""
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")]
        )
        
        if path:
            self.current_image_path = path
            self.image_label.config(text=f"Image: {Path(path).name}")
            self.log(f"Image selected: {path}")
            self.display_image(path)
    
    def display_image(self, image_path: str):
        """Display image in canvas."""
        try:
            img = Image.open(image_path)
            
            # Resize to fit canvas
            max_width, max_height = 500, 500
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(img)
            self.image_canvas.create_image(250, 250, image=photo)
            self.image_canvas.image = photo
            self.current_image = cv2.imread(image_path)
        
        except Exception as e:
            self.log(f"Error displaying image: {str(e)}")
    
    def run_detection(self):
        """Run detection on selected image."""
        if not self.model:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        try:
            self.log("Running detection...")
            
            # Run inference
            results = self.model.predict(self.current_image_path, conf=0.5, device=0 if self._has_gpu() else 'cpu')
            
            if not results or len(results) == 0:
                self.update_results("No detections found")
                self.log("No objects detected")
                return
            
            result = results[0]
            
            # Process detections
            detections = []
            detected_classes = set()
            
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                detections.append((class_name, confidence))
                detected_classes.add(class_name)
            
            # Check safety
            safety_status, safety_details = self.check_safety(detected_classes)
            
            # Draw detections
            annotated_img = self._draw_detections(result, safety_status)
            
            # Display annotated image
            self._display_result_image(annotated_img)
            
            # Show results
            results_msg = self._format_results(safety_status, detections, safety_details)
            self.update_results(results_msg)
            self.log("Detection completed")
        
        except Exception as e:
            messagebox.showerror("Error", f"Detection error: {str(e)}")
            self.log(f"Detection error: {str(e)}")
    
    def check_safety(self, detected_classes: set) -> Tuple[str, str]:
        """Check PPE safety compliance."""
        # Check for violations first
        violations = detected_classes & self.violation_classes
        
        if violations:
            details = f"Violations found: {', '.join(violations)}"
            return "UNSAFE", details
        
        # Check if required PPE is present (human must have helmet, boots, gloves, vest)
        detected_ppe = detected_classes & self.required_ppe
        if len(detected_ppe) < len(self.required_ppe):
            missing = self.required_ppe - detected_ppe
            details = f"Missing required PPE: {', '.join(missing)}"
            return "UNSAFE", details
        
        details = f"All required PPE detected: {', '.join(sorted(self.required_ppe))}"
        return "SAFE", details
    
    def _draw_detections(self, result, safety_status: str):
        """Draw bounding boxes on image."""
        img = self.current_image.copy()
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            
            # Determine color based on class
            if class_name in self.violation_classes:
                box_color = self.color_unsafe
            else:
                box_color = self.color_safe
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw label
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
        
        # Draw safety status
        status_color = self.color_safe if safety_status == "SAFE" else self.color_unsafe
        status_text = f"Status: {safety_status}"
        cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        
        return img
    
    def _display_result_image(self, img):
        """Display result image in canvas."""
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize
            h, w = img_rgb.shape[:2]
            if w > 500 or h > 500:
                scale = min(500 / w, 500 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                img_rgb = cv2.resize(img_rgb, (new_w, new_h))
            
            pil_img = Image.fromarray(img_rgb)
            photo = ImageTk.PhotoImage(pil_img)
            
            self.image_canvas.create_image(250, 250, image=photo)
            self.image_canvas.image = photo
        
        except Exception as e:
            self.log(f"Error displaying result: {str(e)}")
    
    def _format_results(self, safety_status: str, detections: list, details: str) -> str:
        """Format detection results."""
        result_text = f"{'='*40}\n"
        result_text += f"SAFETY STATUS: {safety_status}\n"
        result_text += f"{'='*40}\n\n"
        
        result_text += f"Details: {details}\n\n"
        
        result_text += "Detections:\n"
        result_text += "-" * 40 + "\n"
        
        if detections:
            for class_name, confidence in detections:
                status = "✗" if class_name in self.violation_classes else "✓"
                result_text += f"{status} {class_name}: {confidence:.2%}\n"
        else:
            result_text += "No objects detected\n"
        
        return result_text
    
    def run(self):
        """Run the application."""
        self.log("PPE Detection System started")
        self.root.mainloop()


def main():
    """Main entry point."""
    app = PPEDetectionSystem()
    app.run()


if __name__ == "__main__":
    main()
