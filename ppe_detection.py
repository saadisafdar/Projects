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
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
from ultralytics import YOLO

class PPEDetectionApp:
    """Main application class for PPE Detection System"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PPE Detection System - YOLOv8 from Scratch")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Set application icon and style
        self.setup_styles()
        
        # Initialize variables
        self.dataset_path = None
        self.model = None
        self.current_image = None
        self.detection_image = None
        self.original_image_display = None
        self.detected_image_display = None
        self.class_names = ['boots', 'gloves', 'helmet', 'human', 'no helmet', 'no vest', 'vest']
        self.training_thread = None
        self.training_active = False
        self.detect_button = None
        self.save_button = None
        
        # Color scheme for different PPE classes
        self.class_colors = {
            'boots': (220, 60, 60),     # Red
            'gloves': (60, 220, 60),    # Green
            'helmet': (60, 60, 220),    # Blue
            'human': (220, 220, 60),    # Yellow
            'no helmet': (220, 60, 220),# Magenta
            'no vest': (60, 220, 220),  # Cyan
            'vest': (160, 60, 160)      # Purple
        }
        
        # Setup GUI
        self.setup_gui()
        
        # Update status
        self.update_status("Ready - Select dataset folder to begin")

    def setup_styles(self):
        """Setup ttk styles for professional look"""
        style = ttk.Style()
        
        # Try to use a modern theme if available
        try:
            style.theme_use('clam')
        except:
            pass
        
        # Configure styles
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Section.TLabelframe.Label', font=('Arial', 11, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 9))
        
        # Configure button styles
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'))
        style.map('Accent.TButton',
                 background=[('active', '#4CAF50'), ('pressed', '#45a049')])

    def setup_gui(self):
        """Setup the Tkinter GUI interface"""
        
        # Create main container with padding
        main_container = ttk.Frame(self.root, padding="15")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsive design
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(3, weight=1)
        
        # ==================== HEADER ====================
        header_frame = ttk.Frame(main_container)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        title_label = ttk.Label(header_frame, 
                               text="🚧 PPE Detection System", 
                               font=('Arial', 18, 'bold'),
                               foreground='#2c3e50')
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        subtitle_label = ttk.Label(header_frame,
                                  text="YOLOv8 Training from Scratch | Image Detection Only",
                                  font=('Arial', 10),
                                  foreground='#7f8c8d')
        subtitle_label.grid(row=1, column=0, sticky=tk.W, pady=(2, 0))
        
        # ==================== CONFIGURATION PANEL ====================
        config_frame = ttk.Frame(main_container)
        config_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # DATASET SECTION
        dataset_frame = ttk.LabelFrame(config_frame, text="📁 Dataset Configuration", padding="10")
        dataset_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Dataset path selector with improved layout
        path_frame = ttk.Frame(dataset_frame)
        path_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        ttk.Label(path_frame, text="Dataset Folder:").grid(row=0, column=0, sticky=tk.W)
        self.dataset_entry = ttk.Entry(path_frame, width=45)
        self.dataset_entry.grid(row=0, column=1, padx=(10, 10), sticky=(tk.W, tk.E))
        ttk.Button(path_frame, text="Browse", 
                  command=self.browse_dataset, width=10).grid(row=0, column=2, padx=(0, 10))
        ttk.Button(path_frame, text="Validate", 
                  command=self.validate_dataset, style='Accent.TButton').grid(row=0, column=3)
        
        # Dataset info display
        self.dataset_info = ttk.Label(dataset_frame, 
                                     text="No dataset selected",
                                     foreground='#666666')
        self.dataset_info.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # TRAINING SECTION
        training_frame = ttk.LabelFrame(config_frame, text="🧠 Model Training", padding="10")
        training_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        
        # Epoch configuration
        epoch_frame = ttk.Frame(training_frame)
        epoch_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(epoch_frame, text="Training Epochs:").grid(row=0, column=0, sticky=tk.W)
        self.epoch_entry = ttk.Entry(epoch_frame, width=15)
        self.epoch_entry.insert(0, "50")
        self.epoch_entry.grid(row=0, column=1, padx=(10, 20), sticky=tk.W)
        
        # Training buttons
        button_frame = ttk.Frame(training_frame)
        button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.train_button = ttk.Button(button_frame, text="▶ Start Training", 
                                      command=self.start_training, 
                                      style='Accent.TButton',
                                      state=tk.DISABLED)
        self.train_button.grid(row=0, column=0, padx=(0, 10))
        
        ttk.Button(button_frame, text="⏹ Stop Training", 
                  command=self.stop_training).grid(row=0, column=1)
        
        # Training logs
        logs_frame = ttk.LabelFrame(training_frame, text="Training Logs", padding="5")
        logs_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.log_text = scrolledtext.ScrolledText(logs_frame, height=6, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        logs_frame.columnconfigure(0, weight=1)
        
        # Configure column weights
        config_frame.columnconfigure(0, weight=1)
        config_frame.columnconfigure(1, weight=1)
        
        # ==================== MODEL SECTION ====================
        model_frame = ttk.LabelFrame(main_container, text="📦 Model Management", padding="10")
        model_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        model_controls = ttk.Frame(model_frame)
        model_controls.grid(row=0, column=0, sticky=tk.W)
        
        ttk.Button(model_controls, text="📂 Load Trained Model", 
                  command=self.load_model).grid(row=0, column=0, padx=(0, 10))
        
        self.model_status = ttk.Label(model_controls, 
                                     text="No model loaded",
                                     foreground='#666666')
        self.model_status.grid(row=0, column=1, sticky=tk.W)
        
        # ==================== MAIN DETECTION PANEL ====================
        detection_panel = ttk.Frame(main_container)
        detection_panel.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure weights for responsive layout
        detection_panel.columnconfigure(1, weight=2)  # Image display gets more space
        detection_panel.rowconfigure(0, weight=1)
        
        # LEFT PANEL - Controls and Results
        left_panel = ttk.LabelFrame(detection_panel, text="🖼️ Detection Controls", padding="15")
        left_panel.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 15))
        
        # Control buttons
        control_frame = ttk.Frame(left_panel)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        ttk.Button(control_frame, text="📷 Select Image", 
                  command=self.select_image, width=20).grid(row=0, column=0, pady=(0, 10))
        
        self.detect_button = ttk.Button(control_frame, text="🔍 Run Detection", 
                                       command=self.run_detection, 
                                       width=20, state=tk.DISABLED)
        self.detect_button.grid(row=1, column=0, pady=(0, 10))
        
        self.save_button = ttk.Button(control_frame, text="💾 Save Result", 
                                     command=self.save_result, 
                                     width=20, state=tk.DISABLED)
        self.save_button.grid(row=2, column=0, pady=(0, 10))
        
        # Image info display
        image_info_frame = ttk.LabelFrame(left_panel, text="Image Info", padding="10")
        image_info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.image_label = ttk.Label(image_info_frame, 
                                    text="No image selected",
                                    wraplength=250,
                                    justify=tk.LEFT)
        self.image_label.grid(row=0, column=0, sticky=tk.W)
        
        # Results display
        results_frame = ttk.LabelFrame(left_panel, text="📊 Detection Results", padding="10")
        results_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Treeview for results with better styling
        columns = ('Class', 'Confidence')
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=10)
        
        # Configure columns
        self.results_tree.heading('Class', text='Class Name')
        self.results_tree.heading('Confidence', text='Confidence')
        self.results_tree.column('Class', width=120, anchor=tk.W)
        self.results_tree.column('Confidence', width=80, anchor=tk.CENTER)
        
        # Add scrollbar
        tree_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=tree_scroll.set)
        
        # Grid layout
        self.results_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tree_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure weights for results frame
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        left_panel.rowconfigure(2, weight=1)
        
        # RIGHT PANEL - Image Display
        right_panel = ttk.LabelFrame(detection_panel, text="Image Preview", padding="15")
        right_panel.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Configure grid for image displays
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)
        
        # Original Image Display
        original_frame = ttk.LabelFrame(right_panel, text="Original Image", padding="5")
        original_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W), pady=(0, 10))
        
        # Create canvas with border
        self.original_canvas = tk.Canvas(original_frame, bg='#f5f5f5', highlightthickness=1, 
                                        highlightbackground='#cccccc')
        self.original_canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Add placeholder text
        self.original_canvas.create_text(320, 200, 
                                        text="No image selected\n\nClick 'Select Image' to load an image",
                                        fill="#888888", 
                                        font=("Arial", 11),
                                        justify=tk.CENTER)
        
        # Detected Image Display
        detected_frame = ttk.LabelFrame(right_panel, text="Detection Results", padding="5")
        detected_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Create canvas for detection results
        self.detected_canvas = tk.Canvas(detected_frame, bg='#f5f5f5', highlightthickness=1,
                                        highlightbackground='#cccccc')
        self.detected_canvas.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.E, tk.W))
        
        # Add placeholder text
        self.detected_canvas.create_text(320, 200,
                                        text="Detection results will appear here\n\n"
                                             "Load an image and click 'Run Detection'",
                                        fill="#888888",
                                        font=("Arial", 11),
                                        justify=tk.CENTER)
        
        # Configure frame weights for proper expansion
        original_frame.columnconfigure(0, weight=1)
        original_frame.rowconfigure(0, weight=1)
        detected_frame.columnconfigure(0, weight=1)
        detected_frame.rowconfigure(0, weight=1)
        
        # ==================== STATUS BAR ====================
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN)
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, 
                                     text="Ready",
                                     relief=tk.FLAT,
                                     anchor=tk.W,
                                     padding=(10, 5),
                                     style='Status.TLabel')
        self.status_label.pack(fill=tk.X)
        
        # Configure root weights for final row
        self.root.rowconfigure(1, weight=0)  # Status bar fixed height

    def browse_dataset(self):
        """Browse and select dataset folder"""
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            self.dataset_path = folder
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, folder)
            self.dataset_info.config(text=f"Dataset: {os.path.basename(folder)}", foreground='#2c3e50')

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
            self.dataset_info.config(text="Dataset: Invalid structure", foreground='#e74c3c')
            self.train_button.config(state=tk.DISABLED)
        else:
            # Read and validate data.yaml
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Fix relative paths in data.yaml
                fixed_yaml = self.fix_yaml_paths(yaml_path)
                
                messagebox.showinfo("Success", 
                                   "Dataset structure is valid!\n"
                                   f"Classes: {data.get('nc', 'N/A')}\n"
                                   f"Training images: {data.get('train', 'N/A')}")
                self.update_status("Dataset validated successfully")
                self.dataset_info.config(text=f"Dataset: Valid ({data.get('nc', 0)} classes)", 
                                        foreground='#27ae60')
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
            if epochs > 500:
                if not messagebox.askyesno("Warning", 
                                          f"Training for {epochs} epochs may take a long time. Continue?"):
                    return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for epochs")
            return
        
        # Disable training button during training
        self.train_button.config(state=tk.DISABLED, text="⏳ Training...")
        self.update_status(f"Training started for {epochs} epochs...")
        
        # Clear previous logs
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.insert(tk.END, "=" * 60 + "\n")
        self.log_text.insert(tk.END, "🚀 INITIALIZING YOLOv8 TRAINING FROM SCRATCH\n")
        self.log_text.insert(tk.END, "=" * 60 + "\n")
        self.log_text.insert(tk.END, f"📊 Dataset: {self.dataset_path}\n")
        self.log_text.insert(tk.END, f"⏱️ Epochs: {epochs}\n")
        self.log_text.insert(tk.END, f"🖼️ Image size: 640x640\n")
        self.log_text.insert(tk.END, f"🔬 Pretrained weights: No (Training from scratch)\n")
        self.log_text.insert(tk.END, "-" * 60 + "\n")
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
            self.root.after(0, self.update_training_log, "🔄 Starting training process...")
            
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=640,
                batch=16,
                name='ppe_training',
                pretrained=False,
                patience=10,
                save=True,
                exist_ok=True,
                verbose=False  # We'll handle logging ourselves
            )
            
            # Save the trained model
            model.save('trained_model.pt')
            
            # Update GUI on main thread
            self.root.after(0, self.training_completed, "✅ Training completed successfully!")
            
        except Exception as e:
            error_msg = f"❌ Training error: {str(e)}"
            self.root.after(0, self.training_completed, error_msg)

    def update_training_log(self, message):
        """Update training log from thread"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def training_completed(self, message):
        """Handle training completion"""
        self.training_active = False
        self.train_button.config(state=tk.NORMAL, text="▶ Start Training")
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, "=" * 60 + "\n")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.insert(tk.END, "=" * 60 + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        self.update_status(message)
        
        if "successfully" in message.lower():
            messagebox.showinfo("Training Complete", 
                              "✅ Model training completed successfully!\n"
                              "📁 Model saved as 'trained_model.pt'\n\n"
                              "You can now load this model for detection.")

    def stop_training(self):
        """Stop the training process"""
        if self.training_active:
            self.training_active = False
            self.update_status("Training stopped by user")
            self.train_button.config(state=tk.NORMAL, text="▶ Start Training")
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, "\n⏹️ Training stopped by user\n")
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
                filename = os.path.basename(file_path)
                self.model_status.config(text=f"Model loaded: {filename}", foreground='#27ae60')
                self.update_status(f"Model '{filename}' loaded successfully")
                
                # Enable detection if image is already loaded
                if self.current_image is not None:
                    self.detect_button.config(state=tk.NORMAL)
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.update_status("Model loading failed")

    def select_image(self):
        """Select an image for detection"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load image using OpenCV
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    raise ValueError(f"Could not read image file: {file_path}")
                
                # Clear previous detection
                self.detection_image = None
                self.detected_canvas.delete("all")
                self.detected_canvas.create_text(320, 200,
                                                text="Detection results will appear here\n\n"
                                                     "Click 'Run Detection' to analyze this image",
                                                fill="#888888",
                                                font=("Arial", 11),
                                                justify=tk.CENTER)
                
                # Clear previous results
                for item in self.results_tree.get_children():
                    self.results_tree.delete(item)
                
                # Update image info
                filename = os.path.basename(file_path)
                img_size = f"{self.current_image.shape[1]}x{self.current_image.shape[0]}"
                self.image_label.config(
                    text=f"📄 {filename}\n"
                         f"📏 Size: {img_size}\n"
                         f"🎨 Channels: {self.current_image.shape[2] if len(self.current_image.shape) > 2 else 1}",
                    foreground='#2c3e50'
                )
                
                # Display original image
                self.display_image_on_canvas(self.current_image, self.original_canvas, "Original Image")
                
                # Update status and enable detection button if model is loaded
                self.update_status(f"Image loaded: {filename}")
                self.detect_button.config(state=tk.NORMAL if self.model else tk.DISABLED)
                self.save_button.config(state=tk.DISABLED)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.update_status("Image loading failed")

    def display_image_on_canvas(self, image, canvas, title):
        """Display an image on a canvas with proper scaling"""
        # Convert BGR to RGB for display
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Convert to PIL Image
        image_pil = Image.fromarray(image_rgb)
        
        # Get canvas dimensions
        canvas_width = canvas.winfo_width() or 640
        canvas_height = canvas.winfo_height() or 480
        
        # Calculate scaling while maintaining aspect ratio
        img_width, img_height = image_pil.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        image_pil = image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image_pil)
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, 
                          image=photo, anchor=tk.CENTER)
        
        # Add title
        canvas.create_text(canvas_width//2, 20, 
                         text=title, 
                         fill="#2c3e50", 
                         font=("Arial", 10, "bold"),
                         anchor=tk.N)
        
        # Keep reference
        if canvas == self.original_canvas:
            self.original_image_display = photo
        else:
            self.detected_image_display = photo

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
            self.detect_button.config(state=tk.DISABLED, text="🔍 Processing...")
            
            # Run inference
            results = self.model(self.current_image, conf=0.25, verbose=False)
            
            # Clear previous results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)
            
            # Process results
            if len(results) > 0:
                result = results[0]
                
                # Create a copy for drawing
                self.detection_image = self.current_image.copy()
                
                # Count objects by class
                class_counts = {}
                
                # Draw bounding boxes and collect results
                for box in result.boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    if cls_id < len(self.class_names):
                        class_name = self.class_names[cls_id]
                    else:
                        class_name = f"Class_{cls_id}"
                    
                    # Update count
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    # Add to results tree with color-coded rows
                    item_id = self.results_tree.insert('', tk.END, values=(class_name, f"{conf:.3f}"))
                    
                    # Add color coding based on confidence
                    if conf > 0.7:
                        self.results_tree.tag_configure('high', background='#d4edda')
                        self.results_tree.item(item_id, tags=('high',))
                    elif conf > 0.4:
                        self.results_tree.tag_configure('medium', background='#fff3cd')
                        self.results_tree.item(item_id, tags=('medium',))
                    else:
                        self.results_tree.tag_configure('low', background='#f8d7da')
                        self.results_tree.item(item_id, tags=('low',))
                    
                    # Draw bounding box with thicker lines for better visibility
                    color = self.class_colors.get(class_name, (255, 255, 255))
                    cv2.rectangle(self.detection_image, (int(x1), int(y1)), 
                                 (int(x2), int(y2)), color, 3)
                    
                    # Add label with confidence
                    label = f"{class_name}: {conf:.2f}"
                    
                    # Calculate text background
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(self.detection_image, 
                                 (int(x1), int(y1) - text_size[1] - 10),
                                 (int(x1) + text_size[0] + 10, int(y1)),
                                 color, -1)
                    
                    # Add text
                    cv2.putText(self.detection_image, label, (int(x1) + 5, int(y1) - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display detection image
                self.display_image_on_canvas(self.detection_image, self.detected_canvas, "Detection Results")
                
                # Add statistics to the canvas
                canvas_width = self.detected_canvas.winfo_width() or 640
                stats_text = f"Detected: {len(result.boxes)} objects"
                self.detected_canvas.create_text(canvas_width//2, 40, 
                                               text=stats_text,
                                               fill="#2c3e50",
                                               font=("Arial", 9, "bold"),
                                               anchor=tk.N)
                
                # Enable save button
                self.save_button.config(state=tk.NORMAL)
                
                # Update status with detailed information
                status_msg = f"Detection complete: {len(result.boxes)} objects detected"
                if class_counts:
                    status_msg += f" ({', '.join([f'{k}: {v}' for k, v in class_counts.items()])})"
                self.update_status(status_msg)
                
            else:
                self.update_status("No objects detected")
                self.detected_canvas.delete("all")
                self.detected_canvas.create_text(320, 200,
                                                text="❌ No objects detected\n\n"
                                                     "Try adjusting the confidence threshold\n"
                                                     "or use a different image",
                                                fill="#888888",
                                                font=("Arial", 11),
                                                justify=tk.CENTER)
            
            self.detect_button.config(state=tk.NORMAL, text="🔍 Run Detection")
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
            self.update_status("Detection failed")
            self.detect_button.config(state=tk.NORMAL, text="🔍 Run Detection")

    def save_result(self):
        """Save the detection result image"""
        if self.detection_image is None:
            messagebox.showerror("Error", "No detection result to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Detection Result",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Determine file format from extension
                if file_path.lower().endswith('.png'):
                    cv2.imwrite(file_path, self.detection_image)
                else:
                    # Default to JPEG with good quality
                    cv2.imwrite(file_path, self.detection_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                self.update_status(f"Result saved: {os.path.basename(file_path)}")
                messagebox.showinfo("Success", 
                                  f"✅ Detection result saved successfully!\n"
                                  f"📁 Location: {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")

    def update_status(self, message):
        """Update the status bar"""
        self.status_label.config(text=f"Status: {message}")
        
        # Color code based on message type
        if "Error" in message or "failed" in message.lower():
            self.status_label.config(foreground='#e74c3c')
        elif "success" in message.lower() or "complete" in message.lower():
            self.status_label.config(foreground='#27ae60')
        elif "Training" in message:
            self.status_label.config(foreground='#3498db')
        else:
            self.status_label.config(foreground='#2c3e50')

def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Set window icon if available
    try:
        root.iconbitmap(default='icon.ico')
    except:
        pass
    
    app = PPEDetectionApp(root)
    
    # Handle window resize events
    def on_resize(event):
        # Redisplay images if they exist
        if app.current_image is not None:
            app.display_image_on_canvas(app.current_image, app.original_canvas, "Original Image")
        if app.detection_image is not None:
            app.display_image_on_canvas(app.detection_image, app.detected_canvas, "Detection Results")
    
    root.bind('<Configure>', on_resize)
    
    root.mainloop()

if __name__ == "__main__":
    main()