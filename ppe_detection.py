#!/usr/bin/env python3
"""
PPE Detection System - YOLOv8 from Scratch
Single file implementation: ppe_detection.py
Author: AI Lab Project
Date: 2026-01-12
Description: Personal Protective Equipment detection using YOLOv8 trained from scratch
"""

import os
import sys
import yaml
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO

class PPEDetectionApp:
    """Main application class for PPE Detection System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PPE Detection System - YOLOv8 from Scratch")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
        # Initialize variables
        self.dataset_path = None
        self.model = None
        self.current_image = None
        self.detection_image = None
        self.class_names = ['boots', 'gloves', 'helmet', 'human', 'no helmet', 'no vest', 'vest']
        self.training_thread = None
        self.training_active = False
        
        # Color scheme for different PPE classes
        self.class_colors = {
            'boots': (255, 0, 0),       # Red
            'gloves': (0, 255, 0),      # Green
            'helmet': (0, 0, 255),      # Blue
            'human': (255, 255, 0),     # Cyan
            'no helmet': (255, 0, 255), # Magenta
            'no vest': (0, 255, 255),   # Yellow
            'vest': (128, 0, 128)       # Purple
        }
        
        # Setup GUI
        self.setup_gui()
        
        # Update status
        self.update_status("Ready - Select dataset folder to begin")

    def setup_gui(self):
        """Setup the Tkinter GUI interface"""
        
        # Create main container with padding
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(3, weight=1)
        
        # ==================== DATASET SECTION ====================
        dataset_frame = ttk.LabelFrame(main_container, text="📁 Dataset Configuration", padding="10")
        dataset_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Dataset path selector
        ttk.Label(dataset_frame, text="Dataset Folder:").grid(row=0, column=0, sticky=tk.W)
        self.dataset_entry = ttk.Entry(dataset_frame, width=50)
        self.dataset_entry.grid(row=0, column=1, padx=(5, 5), sticky=(tk.W, tk.E))
        ttk.Button(dataset_frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(dataset_frame, text="Validate Dataset", command=self.validate_dataset).grid(row=0, column=3)
        
        # Dataset info display
        self.dataset_info = ttk.Label(dataset_frame, text="No dataset selected")
        self.dataset_info.grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(5, 0))
        
        # ==================== TRAINING SECTION ====================
        training_frame = ttk.LabelFrame(main_container, text="🧠 Model Training", padding="10")
        training_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Epoch configuration
        ttk.Label(training_frame, text="Training Epochs:").grid(row=0, column=0, sticky=tk.W)
        self.epoch_entry = ttk.Entry(training_frame, width=15)
        self.epoch_entry.insert(0, "50")  # Default value
        self.epoch_entry.grid(row=0, column=1, sticky=tk.W, padx=(5, 20))
        
        # Training buttons
        self.train_button = ttk.Button(training_frame, text="Start Training", 
                                      command=self.start_training, state=tk.DISABLED)
        self.train_button.grid(row=0, column=2, padx=(0, 10))
        
        ttk.Button(training_frame, text="Stop Training", command=self.stop_training).grid(row=0, column=3)
        
        # Training logs
        ttk.Label(training_frame, text="Training Logs:").grid(row=1, column=0, sticky=tk.NW, pady=(10, 0))
        self.log_text = scrolledtext.ScrolledText(training_frame, width=80, height=8, state=tk.DISABLED)
        self.log_text.grid(row=1, column=1, columnspan=3, pady=(10, 0), sticky=(tk.W, tk.E))
        
        # ==================== MODEL SECTION ====================
        model_frame = ttk.LabelFrame(main_container, text="📦 Model Management", padding="10")
        model_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Button(model_frame, text="Load Trained Model (.pt)", 
                  command=self.load_model).grid(row=0, column=0, padx=(0, 10))
        
        self.model_status = ttk.Label(model_frame, text="No model loaded")
        self.model_status.grid(row=0, column=1, sticky=tk.W)
        
        # ==================== IMAGE DETECTION SECTION ====================
        detection_frame = ttk.LabelFrame(main_container, text="🖼️ Image Detection", padding="10")
        detection_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        # Left panel for image selection and results
        left_panel = ttk.Frame(detection_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 10))
        
        ttk.Label(left_panel, text="Image Selection:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Button(left_panel, text="Select Image", 
                  command=self.select_image, width=20).grid(row=1, column=0, pady=(0, 10))
        
        ttk.Button(left_panel, text="Run Detection", 
                  command=self.run_detection, width=20, state=tk.DISABLED).grid(row=2, column=0, pady=(0, 10))
        
        ttk.Button(left_panel, text="Save Result", 
                  command=self.save_result, width=20, state=tk.DISABLED).grid(row=3, column=0, pady=(0, 10))
        
        # Image preview label
        self.image_label = ttk.Label(left_panel, text="No image selected", 
                                    relief=tk.SUNKEN, width=30)
        self.image_label.grid(row=4, column=0, pady=(10, 0))
        
        # Results display
        ttk.Label(left_panel, text="Detection Results:").grid(row=5, column=0, sticky=tk.W, pady=(10, 5))
        
        # Treeview for results
        columns = ('Class', 'Confidence')
        self.results_tree = ttk.Treeview(left_panel, columns=columns, show='headings', height=8)
        self.results_tree.heading('Class', text='Class')
        self.results_tree.heading('Confidence', text='Confidence')
        self.results_tree.column('Class', width=100)
        self.results_tree.column('Confidence', width=80)
        self.results_tree.grid(row=6, column=0, sticky=(tk.W, tk.E))
        
        # Scrollbar for results
        scrollbar = ttk.Scrollbar(left_panel, orient=tk.VERTICAL, 
                                 command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=6, column=1, sticky=(tk.N, tk.S))
        
        # Right panel for image display
        right_panel = ttk.Frame(detection_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        detection_frame.columnconfigure(1, weight=1)
        detection_frame.rowconfigure(0, weight=1)
        
        # Image display area
        self.image_display = tk.Canvas(right_panel, bg='white', width=640, height=480)
        self.image_display.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Image placeholder text on canvas
        self.image_display.create_text(320, 240, text="Original image will appear here", 
                                      fill="gray", font=("Arial", 12))
        
        # Detection image display
        self.detection_display = tk.Canvas(right_panel, bg='white', width=640, height=480)
        self.detection_display.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), pady=(10, 0))
        
        self.detection_display.create_text(320, 240, text="Detection results will appear here", 
                                          fill="gray", font=("Arial", 12))
        
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        
        # ==================== STATUS BAR ====================
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X)

    def browse_dataset(self):
        """Browse and select dataset folder"""
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            self.dataset_path = folder
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, folder)
            self.dataset_info.config(text=f"Dataset: {os.path.basename(folder)}")

    def validate_dataset(self):
        """Validate the selected dataset structure"""
        if not self.dataset_path or not os.path.exists(self.dataset_path):
            messagebox.showerror("Error", "Please select a valid dataset folder first")
            return
        
        required_folders = ['train/images', 'train/labels', 
                           'valid/images', 'valid/labels',
                           'test/images', 'test/labels']
        
        missing_folders = []
        for folder in required_folders:
            full_path = os.path.join(self.dataset_path, folder)
            if not os.path.exists(full_path):
                missing_folders.append(folder)
        
        # Check for data.yaml
        yaml_path = os.path.join(self.dataset_path, 'data.yaml')
        if not os.path.exists(yaml_path):
            missing_folders.append('data.yaml')
        
        if missing_folders:
            messagebox.showerror("Validation Failed", 
                                f"Missing folders/files:\n" + "\n".join(missing_folders))
            self.update_status("Dataset validation failed")
            self.train_button.config(state=tk.DISABLED)
        else:
            # Read and validate data.yaml
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Fix relative paths in data.yaml
                self.fix_yaml_paths(yaml_path)
                
                messagebox.showinfo("Success", "Dataset structure is valid!\n"
                                             f"Classes: {data.get('nc', 'N/A')}\n"
                                             f"Training images: {data.get('train', 'N/A')}")
                self.update_status("Dataset validated successfully")
                self.train_button.config(state=tk.NORMAL)
                
            except Exception as e:
                messagebox.showerror("YAML Error", f"Error reading data.yaml: {str(e)}")
                self.update_status("Dataset validation failed")

    def fix_yaml_paths(self, yaml_path):
        """Fix relative paths in data.yaml to absolute paths"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Convert relative paths to absolute paths
        if 'train' in data and data['train'].startswith('../'):
            rel_path = data['train'][3:]  # Remove '../'
            data['train'] = os.path.join(self.dataset_path, rel_path)
        
        if 'val' in data and data['val'].startswith('../'):
            rel_path = data['val'][3:]  # Remove '../'
            data['val'] = os.path.join(self.dataset_path, rel_path)
        
        if 'test' in data and data['test'].startswith('../'):
            rel_path = data['test'][3:]  # Remove '../'
            data['test'] = os.path.join(self.dataset_path, rel_path)
        
        # Save the fixed yaml
        fixed_yaml_path = os.path.join(self.dataset_path, 'data_fixed.yaml')
        with open(fixed_yaml_path, 'w') as f:
            yaml.dump(data, f)
        
        return fixed_yaml_path

    def start_training(self):
        """Start training the YOLOv8 model from scratch"""
        if not self.dataset_path:
            messagebox.showerror("Error", "Please select and validate dataset first")
            return
        
        try:
            epochs = int(self.epoch_entry.get())
            if epochs <= 0:
                messagebox.showerror("Error", "Epochs must be a positive integer")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for epochs")
            return
        
        # Disable training button during training
        self.train_button.config(state=tk.DISABLED)
        self.update_status("Training in progress...")
        
        # Clear previous logs
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "Initializing YOLOv8 model from scratch...\n")
        self.log_text.insert(tk.END, f"Training for {epochs} epochs\n")
        self.log_text.insert(tk.END, "="*50 + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Start training in a separate thread
        self.training_active = True
        self.training_thread = threading.Thread(
            target=self.train_model_thread,
            args=(epochs,),
            daemon=True
        )
        self.training_thread.start()

    def train_model_thread(self, epochs):
        """Training thread function"""
        try:
            # Initialize YOLOv8 model from scratch (no pretrained weights)
            model = YOLO("yolov8n.yaml")
            
            # Get the fixed yaml path
            yaml_path = os.path.join(self.dataset_path, 'data_fixed.yaml')
            
            # Train the model
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=640,
                batch=16,
                name='ppe_training',
                pretrained=False,
                patience=10,
                save=True,
                exist_ok=True
            )
            
            # Save the trained model
            model.save('trained_model.pt')
            
            # Update GUI on main thread
            self.root.after(0, self.training_completed, "Training completed successfully!")
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.root.after(0, self.training_completed, error_msg)

    def training_completed(self, message):
        """Handle training completion"""
        self.training_active = False
        self.train_button.config(state=tk.NORMAL)
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"\n{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        self.update_status(message)
        
        if "successfully" in message.lower():
            messagebox.showinfo("Training Complete", 
                              "Model training completed successfully!\n"
                              "Model saved as 'trained_model.pt'")

    def stop_training(self):
        """Stop the training process"""
        if self.training_active:
            self.training_active = False
            self.update_status("Training stopped by user")
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, "\nTraining stopped by user\n")
            self.log_text.config(state=tk.DISABLED)

    def load_model(self):
        """Load a trained .pt model file"""
        file_path = filedialog.askopenfilename(
            title="Select Trained Model",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.model = YOLO(file_path)
                self.model_status.config(text=f"Model loaded: {os.path.basename(file_path)}")
                self.update_status("Model loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.update_status("Model loading failed")

    def select_image(self):
        """Select an image for detection"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                # Load and display image
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise ValueError("Could not read image")
                
                # Convert BGR to RGB for display
                image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                
                # Resize if too large
                max_size = (640, 480)
                if image_pil.size[0] > max_size[0] or image_pil.size[1] > max_size[1]:
                    image_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Display on canvas
                photo = ImageTk.PhotoImage(image_pil)
                self.image_display.delete("all")
                self.image_display.create_image(max_size[0]//2, max_size[1]//2, 
                                              image=photo, anchor=tk.CENTER)
                self.image_display.image = photo  # Keep reference
                
                # Update status and enable detection button
                self.image_label.config(text=os.path.basename(file_path))
                self.root.nametowidget('.!ttkframe.!ttklabelframe.!ttkframe.!ttkbutton[3]').config(state=tk.NORMAL)
                self.update_status(f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def run_detection(self):
        """Run detection on the selected image"""
        if self.model is None:
            messagebox.showerror("Error", "Please load a trained model first")
            return
        
        if self.current_image is None:
            messagebox.showerror("Error", "Please select an image first")
            return
        
        try:
            self.update_status("Running detection...")
            
            # Run inference
            results = self.model(self.current_image, conf=0.25)
            
            # Clear previous results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Process results
            if len(results) > 0:
                result = results[0]
                detected_objects = []
                
                # Create a copy for drawing
                self.detection_image = self.current_image.copy()
                
                # Draw bounding boxes and collect results
                for box in result.boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"Class {cls_id}"
                    
                    # Add to results tree
                    self.results_tree.insert('', tk.END, values=(class_name, f"{conf:.2f}"))
                    
                    # Draw bounding box
                    color = self.class_colors.get(class_name, (255, 255, 255))
                    cv2.rectangle(self.detection_image, (int(x1), int(y1)), 
                                 (int(x2), int(y2)), color, 2)
                    
                    # Add label with confidence
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(self.detection_image, label, (int(x1), int(y1)-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Display detection image
                image_rgb = cv2.cvtColor(self.detection_image, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(image_rgb)
                
                # Resize if too large
                max_size = (640, 480)
                if image_pil.size[0] > max_size[0] or image_pil.size[1] > max_size[1]:
                    image_pil.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image_pil)
                self.detection_display.delete("all")
                self.detection_display.create_image(max_size[0]//2, max_size[1]//2, 
                                                  image=photo, anchor=tk.CENTER)
                self.detection_display.image = photo  # Keep reference
                
                # Enable save button
                self.root.nametowidget('.!ttkframe.!ttklabelframe.!ttkframe.!ttkbutton[4]').config(state=tk.NORMAL)
                
                self.update_status(f"Detection complete: {len(result.boxes)} objects detected")
            else:
                self.update_status("No objects detected")
                
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.update_status("Detection failed")

    def save_result(self):
        """Save the detection result image"""
        if self.detection_image is None:
            messagebox.showerror("Error", "No detection result to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Detection Result",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                cv2.imwrite(file_path, self.detection_image)
                self.update_status(f"Result saved: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", "Detection result saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")

    def update_status(self, message):
        """Update the status bar"""
        self.status_label.config(text=f"Status: {message}")

def main():
    """Main entry point"""
    root = tk.Tk()
    app = PPEDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()