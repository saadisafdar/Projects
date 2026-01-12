#!/usr/bin/env python3
"""
ENHANCED PPE DETECTION SYSTEM
Single-file implementation with YOLOv8 and Tkinter GUI
Improved version with better features - No auto-install
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import threading
import warnings
warnings.filterwarnings('ignore')

# Check required packages without auto-installing
def check_requirements():
    """Check if all required packages are installed"""
    required_packages = {
        'ultralytics': 'ultralytics',
        'opencv-python': 'cv2',
        'pillow': 'PIL',
        'pyyaml': 'yaml',
        'numpy': 'numpy',
        'torch': 'torch'
    }
    
    missing = []
    for package, import_name in required_packages.items():
        try:
            if import_name == 'PIL':
                __import__('PIL')
            else:
                __import__(import_name)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("\n" + "="*60)
        print("MISSING REQUIRED PACKAGES:")
        print("="*60)
        for package in missing:
            print(f"  ✗ {package}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing)}")
        print("\nOr install all requirements:")
        print("pip install ultralytics opencv-python pillow pyyaml numpy torch")
        print("="*60 + "\n")
        sys.exit(1)
    
    print("✓ All required packages are installed")

# Check requirements on startup
check_requirements()

# Now import all libraries
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
import torch

class Enhanced_PPE_Detection_System:
    """Enhanced PPE Detection System Class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced PPE Detection System")
        self.root.geometry("1300x900")
        
        # Configure root
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.dataset_path = tk.StringVar()
        self.model_path = tk.StringVar()
        self.image_path = tk.StringVar()
        self.epochs = tk.StringVar(value="50")
        self.confidence_threshold = tk.DoubleVar(value=0.25)
        self.iou_threshold = tk.DoubleVar(value=0.45)
        self.train_thread = None
        self.detection_thread = None
        self.model = None
        self.class_names = []
        self.safety_status = ""
        self.safety_color = "black"
        self.class_colors = {}
        self.train_metrics = {}
        self.detection_history = []
        
        # Default safety rules (will be updated from dataset)
        self.required_ppe = ["helmet", "vest", "boots"]
        self.violation_classes = ["no helmet", "no vest"]
        
        # Create GUI
        self.setup_gui()
        
        # Generate colors for classes
        self.generate_class_colors()
        
        # Log initial message
        self.log_message("Enhanced PPE Detection System Initialized", "info")
        if torch.cuda.is_available():
            self.log_message(f"GPU Available: {torch.cuda.get_device_name(0)}", "success")
        else:
            self.log_message("Running on CPU", "warning")
        
    def generate_class_colors(self):
        """Generate distinct colors for different classes"""
        # Bright, distinct colors
        self.class_colors = {
            "boots": (0, 200, 255),        # Cyan
            "gloves": (255, 191, 0),       # Gold
            "helmet": (50, 205, 50),       # Lime Green
            "human": (255, 105, 180),      # Pink
            "no helmet": (255, 69, 0),     # Red-Orange
            "no vest": (220, 20, 60),      # Crimson
            "vest": (0, 128, 0),           # Green
        }
        
    def setup_gui(self):
        """Setup the enhanced Tkinter GUI interface"""
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Custom colors
        style.configure('Custom.TFrame', background='#f0f0f0')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'), background='#f0f0f0')
        style.configure('Success.TLabel', foreground='green')
        style.configure('Warning.TLabel', foreground='orange')
        style.configure('Error.TLabel', foreground='red')
        
        # Create main container with notebook (tabs)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tab 1: Main Dashboard
        self.dashboard_tab = ttk.Frame(notebook)
        notebook.add(self.dashboard_tab, text="Dashboard")
        self.setup_dashboard_tab()
        
        # Tab 2: Detection
        self.detection_tab = ttk.Frame(notebook)
        notebook.add(self.detection_tab, text="Detection")
        self.setup_detection_tab()
        
        # Tab 3: Training
        self.training_tab = ttk.Frame(notebook)
        notebook.add(self.training_tab, text="Training")
        self.setup_training_tab()
        
        # Tab 4: Statistics
        self.stats_tab = ttk.Frame(notebook)
        notebook.add(self.stats_tab, text="Statistics")
        self.setup_stats_tab()
        
    def setup_dashboard_tab(self):
        """Setup dashboard tab"""
        # Main container
        main_container = ttk.Frame(self.dashboard_tab)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Quick Actions
        left_panel = ttk.LabelFrame(main_container, text="Quick Actions", padding="15")
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        # Action buttons
        actions = [
            ("📁 Load Dataset", self.browse_dataset),
            ("🔧 Validate Dataset", self.validate_dataset),
            ("🧠 Train from Scratch", self.train_from_scratch),
            ("📊 Load Model", self.browse_model),
            ("🔍 Detect Image", self.browse_image),
        ]
        
        for text, command in actions:
            btn = ttk.Button(left_panel, text=text, command=command, width=20)
            btn.pack(pady=5, fill="x")
        
        # System Info
        info_frame = ttk.LabelFrame(left_panel, text="System Info", padding="10")
        info_frame.pack(fill="x", pady=10)
        
        self.system_info_text = tk.Text(info_frame, height=8, width=25, 
                                       font=("Consolas", 9), wrap="word")
        self.system_info_text.pack()
        self.update_system_info()
        
        # Right panel - Status Display
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Status cards
        status_frame = ttk.Frame(right_panel)
        status_frame.pack(fill="x", pady=(0, 10))
        
        self.status_cards = {}
        status_data = [
            ("Model Status", "Not Loaded", "#FF6B6B"),
            ("Dataset", "Not Set", "#4ECDC4"),
            ("Last Detection", "None", "#45B7D1"),
            ("Safety Score", "0%", "#96CEB4"),
        ]
        
        for i, (title, value, color) in enumerate(status_data):
            card = self.create_status_card(status_frame, title, value, color)
            card.grid(row=i//2, column=i%2, padx=5, pady=5, sticky="nsew")
            self.status_cards[title] = card
        
        # Image preview
        preview_frame = ttk.LabelFrame(right_panel, text="Image Preview", padding="10")
        preview_frame.pack(fill="both", expand=True)
        
        self.image_label = ttk.Label(preview_frame, text="No image loaded", 
                                     background="gray90", anchor="center")
        self.image_label.pack(fill="both", expand=True)
        
    def setup_detection_tab(self):
        """Setup detection tab"""
        # Main container
        main_container = ttk.Frame(self.detection_tab)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        # Detection settings
        settings_frame = ttk.LabelFrame(left_panel, text="Detection Settings", padding="15")
        settings_frame.pack(fill="x", pady=(0, 10))
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence:").pack(anchor="w", pady=(0, 5))
        confidence_scale = ttk.Scale(settings_frame, from_=0.1, to=0.9, 
                                    variable=self.confidence_threshold, 
                                    orient="horizontal")
        confidence_scale.pack(fill="x", pady=(0, 10))
        ttk.Label(settings_frame, textvariable=self.confidence_threshold).pack()
        
        # IOU threshold
        ttk.Label(settings_frame, text="IOU Threshold:").pack(anchor="w", pady=(0, 5))
        iou_scale = ttk.Scale(settings_frame, from_=0.1, to=0.9, 
                             variable=self.iou_threshold, 
                             orient="horizontal")
        iou_scale.pack(fill="x", pady=(0, 10))
        ttk.Label(settings_frame, textvariable=self.iou_threshold).pack()
        
        # Detection buttons
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(btn_frame, text="Load Image", command=self.browse_image).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Run Detection", command=self.run_detection, 
                  style="Accent.TButton").pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Batch Process", command=self.batch_detect).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Load Model", command=self.browse_model).pack(fill="x", pady=2)
        
        # Results display
        results_frame = ttk.LabelFrame(left_panel, text="Detection Results", padding="10")
        results_frame.pack(fill="both", expand=True)
        
        self.results_text = tk.Text(results_frame, height=15, font=("Consolas", 10), wrap="word")
        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Safety status
        self.safety_label = ttk.Label(results_frame, text="SAFETY: N/A", 
                                     font=("Arial", 14, "bold"))
        self.safety_label.pack(pady=5)
        
        # Right panel - Image display
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side="right", fill="both", expand=True)
        
        self.detection_image_label = ttk.Label(right_panel, text="Detection Results", 
                                              background="gray90", anchor="center")
        self.detection_image_label.pack(fill="both", expand=True)
        
    def setup_training_tab(self):
        """Setup training tab"""
        # Main container
        main_container = ttk.Frame(self.training_tab)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Training controls
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        
        # Training parameters
        params_frame = ttk.LabelFrame(left_panel, text="Training Parameters", padding="15")
        params_frame.pack(fill="x", pady=(0, 10))
        
        # Epochs
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky="w", pady=5)
        epochs_entry = ttk.Entry(params_frame, textvariable=self.epochs, width=10)
        epochs_entry.grid(row=0, column=1, padx=5, pady=5)
        
        # Batch size
        ttk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, sticky="w", pady=5)
        self.batch_size = tk.StringVar(value="16")
        batch_entry = ttk.Entry(params_frame, textvariable=self.batch_size, width=10)
        batch_entry.grid(row=1, column=1, padx=5, pady=5)
        
        # Image size
        ttk.Label(params_frame, text="Image Size:").grid(row=2, column=0, sticky="w", pady=5)
        self.img_size = tk.StringVar(value="640")
        size_entry = ttk.Entry(params_frame, textvariable=self.img_size, width=10)
        size_entry.grid(row=2, column=1, padx=5, pady=5)
        
        # Use GPU checkbox
        self.use_gpu = tk.BooleanVar(value=torch.cuda.is_available())
        gpu_check = ttk.Checkbutton(params_frame, text="Use GPU", variable=self.use_gpu)
        gpu_check.grid(row=3, column=0, columnspan=2, pady=5)
        
        # Training buttons
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill="x", pady=5)
        
        ttk.Button(btn_frame, text="Train from Scratch", 
                  command=self.train_from_scratch).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Resume Training", 
                  command=self.resume_training).pack(fill="x", pady=2)
        ttk.Button(btn_frame, text="Validate Model", 
                  command=self.validate_model).pack(fill="x", pady=2)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(left_panel, text="Training Progress", padding="10")
        progress_frame.pack(fill="both", expand=True)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress_bar.pack(fill="x", pady=5)
        
        self.epoch_label = ttk.Label(progress_frame, text="Epoch: 0/0")
        self.epoch_label.pack(pady=5)
        
        # Right panel - Training logs
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side="right", fill="both", expand=True)
        
        log_frame = ttk.LabelFrame(right_panel, text="Training Logs", padding="10")
        log_frame.pack(fill="both", expand=True)
        
        self.training_log = scrolledtext.ScrolledText(log_frame, height=25, 
                                                     font=("Consolas", 9))
        self.training_log.pack(fill="both", expand=True)
        
    def setup_stats_tab(self):
        """Setup statistics tab"""
        # Main container
        main_container = ttk.Frame(self.stats_tab)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Stats display
        stats_frame = ttk.LabelFrame(main_container, text="Model Statistics", padding="15")
        stats_frame.pack(fill="both", expand=True)
        
        self.stats_text = tk.Text(stats_frame, height=20, font=("Consolas", 10), wrap="word")
        scrollbar = ttk.Scrollbar(stats_frame, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        
        self.stats_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Update with initial message
        self.stats_text.insert(tk.END, "No statistics available yet.\n")
        self.stats_text.insert(tk.END, "Train a model or load a trained model to see statistics.")
        
    def create_status_card(self, parent, title, value, color):
        """Create a status card widget"""
        card = tk.Frame(parent, bg=color, relief=tk.RAISED, borderwidth=2)
        
        title_label = tk.Label(card, text=title, bg=color, fg="white", 
                              font=("Arial", 10, "bold"))
        title_label.pack(pady=(5, 0))
        
        value_label = tk.Label(card, text=value, bg=color, fg="white", 
                              font=("Arial", 14, "bold"))
        value_label.pack(pady=(0, 5))
        
        card.value_label = value_label
        return card
    
    def update_system_info(self):
        """Update system information display"""
        info = f"Python: {sys.version.split()[0]}\n"
        info += f"PyTorch: {torch.__version__}\n"
        info += f"CUDA: {torch.cuda.is_available()}\n"
        if torch.cuda.is_available():
            info += f"GPU: {torch.cuda.get_device_name(0)}\n"
        info += f"OpenCV: {cv2.__version__}\n"
        info += f"NumPy: {np.__version__}\n"
        
        self.system_info_text.delete(1.0, tk.END)
        self.system_info_text.insert(tk.END, info)
        
    # ===== DATASET METHODS =====
    def browse_dataset(self):
        """Browse for dataset directory"""
        path = filedialog.askdirectory(title="Select Dataset Directory")
        if path:
            self.dataset_path.set(path)
            self.log_message(f"Dataset path set: {path}", "info")
            self.update_status_card("Dataset", os.path.basename(path))
            self.validate_dataset(show_dialog=False)
            
    def validate_dataset(self, show_dialog=True):
        """Validate dataset structure"""
        dataset_path = self.dataset_path.get()
        if not dataset_path:
            if show_dialog:
                messagebox.showerror("Error", "Please select a dataset path")
            return
            
        # Check for data.yaml
        yaml_path = os.path.join(dataset_path, "data.yaml")
        if not os.path.exists(yaml_path):
            if show_dialog:
                messagebox.showerror("Error", "data.yaml not found in dataset directory")
            return
            
        try:
            # Read data.yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            
            # Extract paths
            train_path = data.get('train', '')
            val_path = data.get('val', '')
            test_path = data.get('test', '')
            
            # Convert relative paths to absolute
            def get_absolute_path(rel_path):
                if rel_path.startswith('../'):
                    # Go one level up from dataset directory
                    return os.path.join(os.path.dirname(dataset_path), rel_path[3:])
                else:
                    return os.path.join(dataset_path, rel_path)
            
            train_abs = get_absolute_path(train_path) if train_path else ''
            val_abs = get_absolute_path(val_path) if val_path else ''
            test_abs = get_absolute_path(test_path) if test_path else ''
            
            # Check class names
            self.class_names = data.get('names', [])
            nc = data.get('nc', 0)
            
            # Update safety rules based on classes
            self.update_safety_rules()
            
            # Validate paths
            errors = []
            if train_abs and not os.path.exists(train_abs):
                errors.append(f"Train images not found: {train_abs}")
            if val_abs and not os.path.exists(val_abs):
                errors.append(f"Validation images not found: {val_abs}")
            
            # Check for images
            def count_images(path):
                if not os.path.exists(path):
                    return 0
                extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                return len([f for f in os.listdir(path) 
                          if os.path.splitext(f)[1].lower() in extensions])
            
            train_count = count_images(train_abs) if train_abs else 0
            val_count = count_images(val_abs) if val_abs else 0
            
            if errors and show_dialog:
                messagebox.showerror("Validation Failed", "\n".join(errors))
                self.log_message("Dataset validation failed", "error")
                return
            
            # Log success
            self.log_message(f"Dataset validated successfully", "success")
            self.log_message(f"Classes: {self.class_names}", "info")
            self.log_message(f"Training images: {train_count}", "info")
            self.log_message(f"Validation images: {val_count}", "info")
            
            if show_dialog:
                messagebox.showinfo("Validation Successful", 
                                  f"Dataset validated successfully!\n"
                                  f"Classes: {nc}\n"
                                  f"Training images: {train_count}\n"
                                  f"Validation images: {val_count}")
            
            # Update status
            self.update_status_card("Dataset", f"{nc} classes")
            
        except Exception as e:
            error_msg = f"Error reading data.yaml: {str(e)}"
            self.log_message(error_msg, "error")
            if show_dialog:
                messagebox.showerror("Validation Error", error_msg)
                
    def update_safety_rules(self):
        """Update safety rules based on available classes"""
        # Look for helmet, vest, boots in class names
        self.required_ppe = []
        self.violation_classes = []
        
        for cls in self.class_names:
            cls_lower = cls.lower()
            if cls_lower in ['helmet', 'vest', 'boots', 'gloves']:
                self.required_ppe.append(cls)
            elif 'no' in cls_lower:
                self.violation_classes.append(cls)
        
        self.log_message(f"Safety rules updated: Required={self.required_ppe}, Violations={self.violation_classes}", "info")
        
    # ===== TRAINING METHODS =====
    def train_from_scratch(self):
        """Train YOLOv8 model from scratch"""
        if not self.dataset_path.get():
            messagebox.showerror("Error", "Please select dataset path first")
            return
            
        try:
            epochs = int(self.epochs.get())
            batch_size = int(self.batch_size.get())
            img_size = int(self.img_size.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter: {str(e)}")
            return
            
        # Check if training is already running
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showwarning("Warning", "Training is already in progress")
            return
            
        # Confirm training
        if not messagebox.askyesno("Confirm Training", 
                                  f"Start training with:\n"
                                  f"Epochs: {epochs}\n"
                                  f"Batch size: {batch_size}\n"
                                  f"Image size: {img_size}\n"
                                  f"GPU: {'Yes' if self.use_gpu.get() else 'No'}"):
            return
            
        # Start training in separate thread
        self.train_thread = threading.Thread(
            target=self._train_model_thread,
            args=(epochs, batch_size, img_size, True),
            daemon=True
        )
        self.train_thread.start()
        
        # Start progress bar
        self.progress_bar.start(10)
        self.log_message(f"Starting training from scratch with {epochs} epochs...", "info")
        
    def resume_training(self):
        """Resume training from existing model"""
        model_path = self.model_path.get()
        if not model_path:
            messagebox.showerror("Error", "Please load a model first")
            return
            
        try:
            epochs = int(self.epochs.get())
            batch_size = int(self.batch_size.get())
            img_size = int(self.img_size.get())
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid parameter: {str(e)}")
            return
            
        # Check if training is already running
        if self.train_thread and self.train_thread.is_alive():
            messagebox.showwarning("Warning", "Training is already in progress")
            return
            
        # Start training in separate thread
        self.train_thread = threading.Thread(
            target=self._train_model_thread,
            args=(epochs, batch_size, img_size, False),
            daemon=True
        )
        self.train_thread.start()
        
        # Start progress bar
        self.progress_bar.start(10)
        self.log_message(f"Resuming training with {epochs} epochs...", "info")
        
    def _train_model_thread(self, epochs, batch_size, img_size, from_scratch):
        """Thread function for model training"""
        try:
            dataset_path = self.dataset_path.get()
            yaml_path = os.path.join(dataset_path, "data.yaml")
            
            if from_scratch:
                # Create from scratch
                model = YOLO('yolov8n.yaml')
            else:
                # Resume from existing model
                model = YOLO(self.model_path.get())
            
            # Determine device
            device = 'cuda' if (self.use_gpu.get() and torch.cuda.is_available()) else 'cpu'
            
            # Train the model
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=img_size,
                batch=batch_size,
                device=device,
                save=True,
                exist_ok=True,
                verbose=False,
                workers=0  # Avoid multiprocessing issues
            )
            
            # Save model path
            model_path = results.save_dir or 'runs/detect/train'
            best_model_path = os.path.join(model_path, 'weights', 'best.pt')
            self.model_path.set(best_model_path)
            
            # Store metrics
            self.train_metrics = {
                'epochs': epochs,
                'batch_size': batch_size,
                'img_size': img_size,
                'results': results,
                'save_dir': model_path
            }
            
            self.log_message("Training completed successfully!", "success")
            self.log_message(f"Model saved to: {best_model_path}", "info")
            
            # Update stats
            self.update_training_stats(results)
            
            # Load the trained model
            self.root.after(0, self.load_trained_model, best_model_path)
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.log_message(f"ERROR: {error_msg}", "error")
            self.root.after(0, messagebox.showerror, "Training Error", error_msg)
        finally:
            # Stop progress bar
            self.root.after(0, self.progress_bar.stop)
            
    def load_trained_model(self, model_path):
        """Load the trained model after training completes"""
        self.model_path.set(model_path)
        self.load_model()
        
    def validate_model(self):
        """Validate model performance"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return
            
        try:
            dataset_path = self.dataset_path.get()
            if not dataset_path:
                messagebox.showerror("Error", "Please select dataset path")
                return
                
            yaml_path = os.path.join(dataset_path, "data.yaml")
            
            # Run validation
            results = self.model.val(data=yaml_path)
            
            # Display results
            self.display_validation_results(results)
            
        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            self.log_message(error_msg, "error")
            messagebox.showerror("Validation Error", error_msg)
            
    def display_validation_results(self, results):
        """Display validation results"""
        if not hasattr(results, 'results_dict'):
            return
            
        metrics = results.results_dict
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "=== VALIDATION RESULTS ===\n\n")
        
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.stats_text.insert(tk.END, f"{key}: {value:.4f}\n")
            else:
                self.stats_text.insert(tk.END, f"{key}: {value}\n")
                
        self.log_message("Model validation completed", "info")
        
    def update_training_stats(self, results):
        """Update training statistics"""
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "=== TRAINING STATISTICS ===\n\n")
        
        if hasattr(results, 'results_dict'):
            metrics = results.results_dict
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.stats_text.insert(tk.END, f"{key}: {value:.4f}\n")
                else:
                    self.stats_text.insert(tk.END, f"{key}: {value}\n")
        else:
            self.stats_text.insert(tk.END, "Training completed successfully!\n")
            self.stats_text.insert(tk.END, f"Model saved to: {self.model_path.get()}\n")
            
    # ===== MODEL METHODS =====
    def browse_model(self):
        """Browse for trained model file"""
        path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.model_path.set(path)
            self.log_message(f"Model path set: {path}", "info")
            
    def load_model(self):
        """Load YOLOv8 model"""
        model_path = self.model_path.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file")
            return
            
        try:
            self.model = YOLO(model_path)
            
            # Try to get class names
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
                self.update_safety_rules()
            else:
                self.class_names = []
                
            # Update class colors for all classes
            for cls in self.class_names:
                if cls not in self.class_colors:
                    # Generate random color for new classes
                    import random
                    self.class_colors[cls] = (random.randint(0, 255), 
                                             random.randint(0, 255), 
                                             random.randint(0, 255))
            
            self.log_message(f"Model loaded successfully: {os.path.basename(model_path)}", "success")
            self.log_message(f"Model classes: {self.class_names}", "info")
            
            # Update status
            self.update_status_card("Model Status", "Loaded")
            
            # Update stats
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "=== MODEL INFORMATION ===\n\n")
            self.stats_text.insert(tk.END, f"Model: {os.path.basename(model_path)}\n")
            self.stats_text.insert(tk.END, f"Classes: {len(self.class_names)}\n")
            self.stats_text.insert(tk.END, f"Class list: {self.class_names}\n")
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.log_message(f"ERROR: {error_msg}", "error")
            messagebox.showerror("Model Error", error_msg)
            
    # ===== DETECTION METHODS =====
    def browse_image(self):
        """Browse for image file"""
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.image_path.set(path)
            self.display_image(path)
            self.log_message(f"Image loaded: {os.path.basename(path)}", "info")
            
    def batch_detect(self):
        """Batch process multiple images"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return
            
        paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("All files", "*.*")
            ]
        )
        
        if not paths:
            return
            
        self.log_message(f"Starting batch detection on {len(paths)} images...", "info")
        
        # Process in separate thread
        thread = threading.Thread(
            target=self._batch_detect_thread,
            args=(paths,),
            daemon=True
        )
        thread.start()
        
    def _batch_detect_thread(self, paths):
        """Thread function for batch detection"""
        try:
            results = []
            for i, path in enumerate(paths):
                try:
                    # Run detection
                    detections = self.model(path, conf=self.confidence_threshold.get(), 
                                          iou=self.iou_threshold.get(), verbose=False)[0]
                    
                    # Process results
                    if detections.boxes is not None:
                        safety_result = self.analyze_safety_from_results(detections)
                        results.append({
                            'image': os.path.basename(path),
                            'detections': len(detections.boxes),
                            'safety': safety_result['status'].split(' - ')[0]
                        })
                        
                    # Log progress
                    self.root.after(0, self.log_message, 
                                   f"Processed {i+1}/{len(paths)}: {os.path.basename(path)}", "info")
                    
                except Exception as e:
                    self.root.after(0, self.log_message, 
                                   f"Error processing {path}: {str(e)}", "error")
            
            # Display batch results
            self.root.after(0, self.display_batch_results, results)
            
        except Exception as e:
            error_msg = f"Batch detection error: {str(e)}"
            self.root.after(0, self.log_message, error_msg, "error")
            
    def display_batch_results(self, results):
        """Display batch detection results"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "=== BATCH DETECTION RESULTS ===\n\n")
        
        safe_count = 0
        total_count = len(results)
        
        for result in results:
            self.results_text.insert(tk.END, 
                                    f"Image: {result['image']}\n"
                                    f"Detections: {result['detections']}\n"
                                    f"Safety: {result['safety']}\n\n")
            
            if result['safety'] == 'SAFE':
                safe_count += 1
                
        safety_rate = (safe_count / total_count * 100) if total_count > 0 else 0
        
        self.results_text.insert(tk.END, f"Summary:\n")
        self.results_text.insert(tk.END, f"Total images: {total_count}\n")
        self.results_text.insert(tk.END, f"Safe images: {safe_count}\n")
        self.results_text.insert(tk.END, f"Safety rate: {safety_rate:.1f}%\n")
        
        self.log_message(f"Batch detection completed: {safe_count}/{total_count} safe ({safety_rate:.1f}%)", 
                        "success" if safety_rate > 80 else "warning")
        
    def display_image(self, image_path):
        """Display image in the preview panel"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Resize to fit panel while maintaining aspect ratio
            max_size = (400, 300)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Keep reference
            
        except Exception as e:
            self.log_message(f"Error loading image: {str(e)}", "error")
            
    def run_detection(self):
        """Run detection on selected image"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return
            
        if not self.image_path.get():
            messagebox.showerror("Error", "Please select an image first")
            return
            
        # Check if detection is already running
        if self.detection_thread and self.detection_thread.is_alive():
            messagebox.showwarning("Warning", "Detection is already in progress")
            return
            
        # Start detection in separate thread
        self.detection_thread = threading.Thread(
            target=self._detection_thread,
            daemon=True
        )
        self.detection_thread.start()
        
        self.log_message("Starting detection...", "info")
        
    def _detection_thread(self):
        """Thread function for detection"""
        try:
            image_path = self.image_path.get()
            
            # Run detection
            results = self.model(image_path, conf=self.confidence_threshold.get(), 
                               iou=self.iou_threshold.get(), verbose=False)
            
            # Process results
            self.root.after(0, self.process_detection_results, results, image_path)
            
        except Exception as e:
            error_msg = f"Detection error: {str(e)}"
            self.log_message(f"ERROR: {error_msg}", "error")
            self.root.after(0, messagebox.showerror, "Detection Error", error_msg)
            
    def process_detection_results(self, results, image_path):
        """Process and display detection results"""
        try:
            # Get first result (single image)
            result = results[0]
            
            # Get detections
            boxes = result.boxes
            detections = []
            
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].tolist()
                    
                    if class_id < len(self.class_names):
                        class_name = self.class_names[class_id]
                    else:
                        class_name = f"Class_{class_id}"
                        
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox
                    })
            
            # Run safety analysis
            safety_result = self.analyze_safety_from_results(result)
            
            # Draw results on image
            annotated_image = self.draw_detections(image_path, detections, safety_result)
            
            # Update GUI
            self.display_annotated_image(annotated_image)
            self.display_detection_results(detections, safety_result)
            
            # Update history
            self.detection_history.append({
                'timestamp': datetime.now(),
                'image': os.path.basename(image_path),
                'detections': len(detections),
                'safety': safety_result['status'].split(' - ')[0]
            })
            
            # Update status cards
            self.update_status_card("Last Detection", f"{len(detections)} objects")
            self.update_status_card("Safety Score", f"{safety_result['safety_score']}%")
            
            self.log_message(f"Detection complete: {len(detections)} objects found", "success")
            
        except Exception as e:
            self.log_message(f"Error processing results: {str(e)}", "error")
            
    def analyze_safety_from_results(self, result):
        """Analyze safety compliance based on detection results"""
        # Get detected classes and their counts
        if result.boxes is None or result.boxes.cls is None:
            return {
                'status': 'SAFE - No objects detected',
                'color': 'green',
                'violations': [],
                'missing_ppe': [],
                'persons_detected': False,
                'safety_score': 100,
                'detected_classes': []
            }
            
        detected_class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []
        detected_classes = [self.class_names[i] for i in detected_class_ids if i < len(self.class_names)]
        
        # Check for violations
        violations = []
        for violation_class in self.violation_classes:
            if violation_class in detected_classes:
                violations.append(violation_class)
        
        # Check for persons
        persons_detected = any(cls.lower() in ['human', 'person'] for cls in detected_classes)
        
        # Check for required PPE if person is detected
        missing_ppe = []
        if persons_detected:
            for required in self.required_ppe:
                if required not in detected_classes:
                    missing_ppe.append(required)
        
        # Determine safety status
        if violations:
            status = f"UNSAFE - Violations: {', '.join(violations)}"
            color = "red"
            safety_score = 0
        elif missing_ppe:
            status = f"UNSAFE - Missing PPE: {', '.join(missing_ppe)}"
            color = "orange"
            safety_score = 30
        elif not persons_detected:
            status = "SAFE - No persons detected"
            color = "blue"
            safety_score = 100
        else:
            status = "SAFE"
            color = "green"
            safety_score = 100
            
        return {
            'status': status,
            'color': color,
            'violations': violations,
            'missing_ppe': missing_ppe,
            'persons_detected': persons_detected,
            'safety_score': safety_score,
            'detected_classes': detected_classes
        }
        
    def draw_detections(self, image_path, detections, safety_result):
        """Draw bounding boxes on image with enhanced visualization"""
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create PIL image for drawing
        pil_image = Image.fromarray(image_rgb)
        draw = ImageDraw.Draw(pil_image)
        
        # Try to load a font, fall back to default
        try:
            import tkinter.font as tkFont
            font = ImageDraw.ImageFont.truetype("arial.ttf", 20)
        except:
            # Use default font
            font = ImageDraw.ImageFont.load_default()
        
        # Draw each detection
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            bbox = detection['bbox']
            
            # Get color for this class
            if class_name in self.class_colors:
                color = self.class_colors[class_name]
            elif 'no' in class_name.lower():
                color = (255, 69, 0)  # Red-Orange for violations
            else:
                color = (0, 200, 0)  # Green for PPE
            
            # Convert coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label background
            label = f"{class_name} {confidence:.2f}"
            
            # Draw text
            draw.text((x1 + 5, y1 + 5), label, fill=color, font=font)
        
        # Add safety status overlay
        status = safety_result['status']
        color_name = safety_result['color']
        
        # Convert color name to RGB
        if color_name == "red":
            status_color = (255, 0, 0)
        elif color_name == "green":
            status_color = (0, 255, 0)
        elif color_name == "orange":
            status_color = (255, 165, 0)
        else:
            status_color = (0, 0, 255)  # Blue
            
        # Draw status bar
        draw.rectangle([0, 0, pil_image.width, 50], fill=(40, 40, 40))
        draw.text((10, 10), f"SAFETY: {status}", fill=status_color, font=font)
        
        return np.array(pil_image)
        
    def display_annotated_image(self, image_array):
        """Display annotated image in GUI"""
        try:
            # Convert numpy array to PIL Image
            image = Image.fromarray(image_array)
            
            # Resize to fit panel
            max_size = (600, 400)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.detection_image_label.configure(image=photo)
            self.detection_image_label.image = photo  # Keep reference
            
        except Exception as e:
            self.log_message(f"Error displaying annotated image: {str(e)}", "error")
            
    def display_detection_results(self, detections, safety_result):
        """Display detection results in text box"""
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Add detection summary
        self.results_text.insert(tk.END, "=== DETECTION RESULTS ===\n\n")
        self.results_text.insert(tk.END, f"Total detections: {len(detections)}\n\n")
        
        # Add class counts
        class_counts = {}
        for detection in detections:
            class_name = detection['class']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        for class_name, count in class_counts.items():
            self.results_text.insert(tk.END, f"{class_name}: {count}\n")
            
        self.results_text.insert(tk.END, "\n" + "="*40 + "\n\n")
        
        # Add safety analysis
        self.results_text.insert(tk.END, "=== SAFETY ANALYSIS ===\n\n")
        
        if safety_result['persons_detected']:
            self.results_text.insert(tk.END, "✓ Persons detected\n")
        else:
            self.results_text.insert(tk.END, "⚠ No persons detected\n")
            
        if safety_result['violations']:
            self.results_text.insert(tk.END, f"✗ Violations: {', '.join(safety_result['violations'])}\n")
        else:
            self.results_text.insert(tk.END, "✓ No violations detected\n")
            
        if safety_result['missing_ppe']:
            self.results_text.insert(tk.END, f"✗ Missing PPE: {', '.join(safety_result['missing_ppe'])}\n")
        else:
            self.results_text.insert(tk.END, "✓ All required PPE present\n")
            
        self.results_text.insert(tk.END, "\n" + "="*40 + "\n\n")
        
        # Add final safety status
        status_text = safety_result['status']
        status_color = safety_result['color']
        safety_score = safety_result['safety_score']
        
        self.results_text.insert(tk.END, "FINAL STATUS:\n")
        self.results_text.insert(tk.END, f"Status: {status_text}\n")
        self.results_text.insert(tk.END, f"Safety Score: {safety_score}%\n")
        
        # Update safety label
        self.safety_label.config(text=f"SAFETY: {status_text.split(' - ')[0]}", 
                                foreground=status_color)
        
    # ===== UTILITY METHODS =====
    def log_message(self, message, level="info"):
        """Add message to log window with color coding"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding based on level
        if level == "error":
            color = "red"
            prefix = "[ERROR]"
        elif level == "warning":
            color = "orange"
            prefix = "[WARN] "
        elif level == "success":
            color = "green"
            prefix = "[OK]   "
        else:
            color = "black"
            prefix = "[INFO] "
            
        log_entry = f"[{timestamp}] {prefix} {message}\n"
        
        # Insert with color tag
        self.training_log.insert(tk.END, log_entry)
        
        # Configure tag for color
        self.training_log.tag_configure(color, foreground=color)
        
        # Apply tag to the line
        start_idx = self.training_log.index("end-1c linestart")
        end_idx = self.training_log.index("end-1c lineend")
        self.training_log.tag_add(color, start_idx, end_idx)
        
        self.training_log.see(tk.END)  # Auto-scroll to bottom
        
        # Also print to console
        print(log_entry.strip())
        
    def update_status_card(self, title, value):
        """Update a status card value"""
        if title in self.status_cards:
            self.status_cards[title].value_label.config(text=value)
            
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.root.destroy()
            sys.exit(0)

def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Create application
    app = Enhanced_PPE_Detection_System(root)
    
    # Set closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()