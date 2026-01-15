"""
PPE Image Detection Desktop Application
Complete YOLOv8-based PPE detection system with training and inference
Author: AI Lab Project
Date: 2026-01-14
"""

# Import required libraries
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk, ImageOps
import cv2
import numpy as np
import yaml
import os
import shutil
from datetime import datetime
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ultralytics import YOLO

class PPE_Detection_App:
    """Main application class for PPE Detection System"""
    
    def __init__(self, root):
        """Initialize the main application window"""
        self.root = root
        self.root.title("PPE Detection System - HeadGear Sentinel")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Application state variables
        self.dataset_path = ""
        self.model = None
        self.trained_model_path = "trained_model.pt"
        self.class_names = []
        self.detection_results = []
        self.training_thread = None
        self.is_training = False
        self.ppe_requirements = {}  # Store PPE requirements for each class
        
        # Create the main notebook (tab system)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create the three main sections
        self.create_training_section()
        self.create_detection_section()
        self.create_results_section()
        
        # Set initial status
        self.update_status("Idle")
    
    def create_training_section(self):
        """Create the training section (Tab 1)"""
        # (Training section code remains the same as before)
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🟦 TRAINING")
        
        # Title
        title_label = tk.Label(training_frame, text="PPE Model Training", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=(10, 20))
        
        # Main content frame
        content_frame = tk.Frame(training_frame, bg='#f0f0f0')
        content_frame.pack(fill='both', expand=True, padx=20)
        
        # Left panel - Dataset configuration
        left_panel = tk.Frame(content_frame, bg='#ffffff', relief='solid', borderwidth=1)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        tk.Label(left_panel, text="Dataset Configuration", font=('Arial', 12, 'bold'), 
                bg='#ffffff').pack(pady=10)
        
        # Dataset selection
        dataset_frame = tk.Frame(left_panel, bg='#ffffff')
        dataset_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(dataset_frame, text="Dataset Folder:", bg='#ffffff').pack(anchor='w')
        
        self.dataset_entry = tk.Entry(dataset_frame, width=50)
        self.dataset_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        browse_btn = tk.Button(dataset_frame, text="Browse", 
                              command=self.browse_dataset, bg='#4CAF50', fg='white')
        browse_btn.pack(side='right')
        
        # Dataset format information
        format_frame = tk.Frame(left_panel, bg='#ffffff')
        format_frame.pack(fill='x', padx=20, pady=10)
        
        format_text = """Required Dataset Structure:
dataset/
├── data.yaml
├── train/images/
├── train/labels/
├── valid/images/
├── valid/labels/
└── test/images/ (optional)"""
        
        format_label = tk.Label(format_frame, text=format_text, justify='left', 
                               bg='#f8f9fa', relief='solid', borderwidth=1)
        format_label.pack(fill='x', pady=5)
        
        # Validation button
        validate_btn = tk.Button(left_panel, text="Validate Dataset Structure", 
                                command=self.validate_dataset, bg='#2196F3', fg='white')
        validate_btn.pack(pady=10)
        
        self.validation_label = tk.Label(left_panel, text="", bg='#ffffff')
        self.validation_label.pack()
        
        # Training parameters
        params_frame = tk.Frame(left_panel, bg='#ffffff')
        params_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(params_frame, text="Training Parameters", font=('Arial', 11, 'bold'), 
                bg='#ffffff').pack(anchor='w', pady=(0, 10))
        
        # Epochs input
        epoch_frame = tk.Frame(params_frame, bg='#ffffff')
        epoch_frame.pack(fill='x', pady=5)
        
        tk.Label(epoch_frame, text="Epochs:", bg='#ffffff').pack(side='left')
        self.epoch_entry = tk.Entry(epoch_frame, width=15)
        self.epoch_entry.insert(0, "100")
        self.epoch_entry.pack(side='left', padx=(10, 0))
        
        # Image size selector
        size_frame = tk.Frame(params_frame, bg='#ffffff')
        size_frame.pack(fill='x', pady=5)
        
        tk.Label(size_frame, text="Image Size:", bg='#ffffff').pack(side='left')
        self.size_var = tk.StringVar(value="640")
        size_options = ["320", "416", "512", "640", "768"]
        size_menu = tk.OptionMenu(size_frame, self.size_var, *size_options)
        size_menu.pack(side='left', padx=(10, 0))
        
        # Right panel - Training controls and logs
        right_panel = tk.Frame(content_frame, bg='#ffffff', relief='solid', borderwidth=1)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Training controls
        controls_frame = tk.Frame(right_panel, bg='#ffffff')
        controls_frame.pack(fill='x', padx=20, pady=20)
        
        # Status indicator
        self.status_frame = tk.Frame(controls_frame, bg='#e0e0e0', height=30)
        self.status_frame.pack(fill='x', pady=(0, 20))
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(self.status_frame, text="Status: Idle", 
                                    bg='#e0e0e0', font=('Arial', 10))
        self.status_label.pack(side='left', padx=10)
        
        self.status_indicator = tk.Label(self.status_frame, text="●", 
                                        fg='gray', font=('Arial', 14))
        self.status_indicator.pack(side='right', padx=10)
        
        # Start training button
        self.train_btn = tk.Button(controls_frame, text="Start Training", 
                                  command=self.start_training, 
                                  bg='#4CAF50', fg='white', font=('Arial', 11))
        self.train_btn.pack(pady=10)
        
        # Stop training button
        self.stop_btn = tk.Button(controls_frame, text="Stop Training", 
                                 command=self.stop_training, 
                                 bg='#f44336', fg='white', font=('Arial', 11),
                                 state='disabled')
        self.stop_btn.pack(pady=10)
        
        # Training logs
        log_frame = tk.Frame(right_panel, bg='#ffffff')
        log_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        tk.Label(log_frame, text="Training Logs", font=('Arial', 11, 'bold'), 
                bg='#ffffff').pack(anchor='w', pady=(0, 10))
        
        # Create scrolled text widget for logs
        self.log_text = scrolledtext.ScrolledText(log_frame, height=20, 
                                                 bg='#000000', fg='#00ff00')
        self.log_text.pack(fill='both', expand=True)
    
    def create_detection_section(self):
        """Create the detection section (Tab 2) - FIXED IMAGE DISPLAY"""
        detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(detection_frame, text="🟦 DETECTION")
        
        # Title
        title_label = tk.Label(detection_frame, text="PPE Detection", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=(10, 20))
        
        # Main content frame with scrollbars
        main_container = tk.Frame(detection_frame, bg='#f0f0f0')
        main_container.pack(fill='both', expand=True)
        
        # Create a canvas with scrollbars
        canvas = tk.Canvas(main_container, bg='#f0f0f0')
        scrollbar_y = ttk.Scrollbar(main_container, orient='vertical', command=canvas.yview)
        scrollbar_x = ttk.Scrollbar(detection_frame, orient='horizontal', command=canvas.xview)
        
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        # Pack everything
        canvas.pack(side="left", fill="both", expand=True, padx=20)
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")
        
        content_frame = tk.Frame(scrollable_frame, bg='#f0f0f0')
        content_frame.pack(fill='both', expand=True)
        
        # Left panel - Controls
        left_panel = tk.Frame(content_frame, bg='#ffffff', relief='solid', borderwidth=1)
        left_panel.pack(side='left', fill='both', expand=False, padx=(0, 10), pady=10)
        
        tk.Label(left_panel, text="Detection Controls", font=('Arial', 12, 'bold'), 
                bg='#ffffff').pack(pady=20)
        
        # Load model button
        load_frame = tk.Frame(left_panel, bg='#ffffff')
        load_frame.pack(fill='x', padx=20, pady=10)
        
        self.load_model_btn = tk.Button(load_frame, text="Load Trained Model", 
                                       command=self.load_model, 
                                       bg='#2196F3', fg='white', width=20)
        self.load_model_btn.pack()
        
        self.model_status_label = tk.Label(load_frame, text="No model loaded", 
                                          bg='#ffffff', fg='red')
        self.model_status_label.pack(pady=5)
        
        # Upload options
        upload_frame = tk.Frame(left_panel, bg='#ffffff')
        upload_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(upload_frame, text="Upload Images:", font=('Arial', 11), 
                bg='#ffffff').pack(anchor='w', pady=(0, 10))
        
        # Single image upload
        single_img_btn = tk.Button(upload_frame, text="Upload Single Image", 
                                  command=self.upload_single_image, 
                                  bg='#607D8B', fg='white', width=20)
        single_img_btn.pack(pady=5)
        
        # Folder upload
        folder_btn = tk.Button(upload_frame, text="Upload Image Folder", 
                              command=self.upload_image_folder, 
                              bg='#607D8B', fg='white', width=20)
        folder_btn.pack(pady=5)
        
        # Image info display
        self.image_info_label = tk.Label(upload_frame, text="", bg='#ffffff')
        self.image_info_label.pack(pady=10)
        
        # Detection button
        self.detect_btn = tk.Button(left_panel, text="Run Detection", 
                                   command=self.run_detection, 
                                   bg='#4CAF50', fg='white', width=20,
                                   state='disabled')
        self.detect_btn.pack(pady=20)
        
        # Save output option
        self.save_var = tk.BooleanVar(value=True)
        save_check = tk.Checkbutton(left_panel, text="Save detected output", 
                                   variable=self.save_var, bg='#ffffff')
        save_check.pack(pady=10)
        
        # Right panel - Image preview with better layout
        right_panel = tk.Frame(content_frame, bg='#ffffff', relief='solid', borderwidth=1)
        right_panel.pack(side='right', fill='both', expand=True, pady=10)
        
        tk.Label(right_panel, text="Image Preview", font=('Arial', 12, 'bold'), 
                bg='#ffffff').pack(pady=20)
        
        # Create a frame to hold both image displays with scrollbars
        images_container = tk.Frame(right_panel, bg='#ffffff')
        images_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Original image display with label
        orig_frame = tk.Frame(images_container, bg='#f0f0f0')
        orig_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        tk.Label(orig_frame, text="Original Image", font=('Arial', 11, 'bold'), 
                bg='#f0f0f0').pack(pady=(0, 10))
        
        # Create scrollable canvas for original image
        orig_canvas_frame = tk.Frame(orig_frame, bg='white')
        orig_canvas_frame.pack(fill='both', expand=True)
        
        self.original_canvas = tk.Canvas(orig_canvas_frame, bg='white', 
                                        width=500, height=400,
                                        scrollregion=(0, 0, 500, 400))
        orig_v_scrollbar = ttk.Scrollbar(orig_canvas_frame, orient='vertical', 
                                        command=self.original_canvas.yview)
        orig_h_scrollbar = ttk.Scrollbar(orig_frame, orient='horizontal',
                                        command=self.original_canvas.xview)
        
        self.original_canvas.configure(yscrollcommand=orig_v_scrollbar.set,
                                      xscrollcommand=orig_h_scrollbar.set)
        
        self.original_canvas.pack(side='left', fill='both', expand=True)
        orig_v_scrollbar.pack(side='right', fill='y')
        orig_h_scrollbar.pack(side='bottom', fill='x')
        
        # Detected image display with label
        det_frame = tk.Frame(images_container, bg='#f0f0f0')
        det_frame.pack(side='right', fill='both', expand=True)
        
        tk.Label(det_frame, text="Detected Image", font=('Arial', 11, 'bold'), 
                bg='#f0f0f0').pack(pady=(0, 10))
        
        # Create scrollable canvas for detected image
        det_canvas_frame = tk.Frame(det_frame, bg='white')
        det_canvas_frame.pack(fill='both', expand=True)
        
        self.detected_canvas = tk.Canvas(det_canvas_frame, bg='white',
                                        width=500, height=400,
                                        scrollregion=(0, 0, 500, 400))
        det_v_scrollbar = ttk.Scrollbar(det_canvas_frame, orient='vertical',
                                       command=self.detected_canvas.yview)
        det_h_scrollbar = ttk.Scrollbar(det_frame, orient='horizontal',
                                       command=self.detected_canvas.xview)
        
        self.detected_canvas.configure(yscrollcommand=det_v_scrollbar.set,
                                      xscrollcommand=det_h_scrollbar.set)
        
        self.detected_canvas.pack(side='left', fill='both', expand=True)
        det_v_scrollbar.pack(side='right', fill='y')
        det_h_scrollbar.pack(side='bottom', fill='x')
        
        # Navigation buttons for multiple images
        nav_frame = tk.Frame(right_panel, bg='#ffffff')
        nav_frame.pack(pady=10)
        
        self.prev_btn = tk.Button(nav_frame, text="← Previous",
                                 command=self.show_previous_image,
                                 state='disabled')
        self.prev_btn.pack(side='left', padx=5)
        
        self.next_btn = tk.Button(nav_frame, text="Next →",
                                 command=self.show_next_image,
                                 state='disabled')
        self.next_btn.pack(side='left', padx=5)
        
        self.image_counter_label = tk.Label(nav_frame, text="Image 0/0",
                                           bg='#ffffff')
        self.image_counter_label.pack(side='left', padx=10)
        
        # Current images
        self.current_original_image = None
        self.current_detected_image = None
        self.image_paths = []
        self.current_image_index = 0
        self.detection_results_list = []
    
    def show_previous_image(self):
        """Show previous image in the list"""
        if self.image_paths and self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_original_image(self.image_paths[self.current_image_index])
            self.update_image_navigation()
            
            # Show detection results if available
            if self.current_image_index < len(self.detection_results_list):
                self.display_detected_image(
                    self.image_paths[self.current_image_index],
                    self.detection_results_list[self.current_image_index]
                )
    
    def show_next_image(self):
        """Show next image in the list"""
        if self.image_paths and self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.display_original_image(self.image_paths[self.current_image_index])
            self.update_image_navigation()
            
            # Show detection results if available
            if self.current_image_index < len(self.detection_results_list):
                self.display_detected_image(
                    self.image_paths[self.current_image_index],
                    self.detection_results_list[self.current_image_index]
                )
    
    def update_image_navigation(self):
        """Update navigation buttons and counter"""
        if len(self.image_paths) > 1:
            self.prev_btn.config(state='normal' if self.current_image_index > 0 else 'disabled')
            self.next_btn.config(state='normal' if self.current_image_index < len(self.image_paths) - 1 else 'disabled')
            self.image_counter_label.config(
                text=f"Image {self.current_image_index + 1}/{len(self.image_paths)}"
            )
        else:
            self.prev_btn.config(state='disabled')
            self.next_btn.config(state='disabled')
            self.image_counter_label.config(text="")
    
    def create_results_section(self):
        """Create the results and analysis section (Tab 3)"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="🟦 RESULTS & ANALYSIS")
        
        # Title
        title_label = tk.Label(results_frame, text="Detection Results & Analysis", 
                              font=('Arial', 16, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=(10, 20))
        
        # Main content frame
        content_frame = tk.Frame(results_frame, bg='#f0f0f0')
        content_frame.pack(fill='both', expand=True, padx=20)
        
        # Left panel - Detailed results
        left_panel = tk.Frame(content_frame, bg='#ffffff', relief='solid', borderwidth=1)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10), pady=10)
        
        tk.Label(left_panel, text="Detection Results", font=('Arial', 12, 'bold'), 
                bg='#ffffff').pack(pady=20)
        
        # Results text box
        results_text_frame = tk.Frame(left_panel, bg='#ffffff')
        results_text_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.results_text = scrolledtext.ScrolledText(results_text_frame, height=20,
                                                     bg='#f8f9fa')
        self.results_text.pack(fill='both', expand=True)
        
        # Safety compliance calculation
        safety_frame = tk.Frame(left_panel, bg='#ffffff')
        safety_frame.pack(fill='x', padx=20, pady=20)
        
        self.safety_label = tk.Label(safety_frame, text="Safety Compliance: Not Calculated", 
                                    font=('Arial', 12, 'bold'), bg='#ffffff')
        self.safety_label.pack()
        
        # Explanation text
        explanation_frame = tk.Frame(left_panel, bg='#ffffff')
        explanation_frame.pack(fill='x', padx=20, pady=10)
        
        self.explanation_text = tk.Text(explanation_frame, height=6, bg='#f8f9fa')
        self.explanation_text.pack(fill='x')
        
        # Right panel - Visualization
        right_panel = tk.Frame(content_frame, bg='#ffffff', relief='solid', borderwidth=1)
        right_panel.pack(side='right', fill='both', expand=True, pady=10)
        
        tk.Label(right_panel, text="Visualization", font=('Arial', 12, 'bold'), 
                bg='#ffffff').pack(pady=20)
        
        # Chart display area
        chart_frame = tk.Frame(right_panel, bg='#f0f0f0')
        chart_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.chart_canvas = tk.Canvas(chart_frame, bg='white')
        self.chart_canvas.pack(fill='both', expand=True)
        
        # Default message in chart area
        self.chart_canvas.create_text(250, 150, 
                                     text="Detection results will appear here", 
                                     fill='gray', font=('Arial', 12))
    
    # ============================================================================
    # DATASET AND TRAINING FUNCTIONS
    # ============================================================================
    
    def browse_dataset(self):
        """Open dialog to select dataset folder"""
        folder_path = filedialog.askdirectory(title="Select Dataset Folder")
        if folder_path:
            self.dataset_entry.delete(0, tk.END)
            self.dataset_entry.insert(0, folder_path)
            self.dataset_path = folder_path
    
    def validate_dataset(self):
        """Validate the dataset structure and data.yaml file"""
        dataset_path = self.dataset_entry.get()
        
        if not dataset_path:
            self.validation_label.config(text="Please select a dataset folder", fg='red')
            return False
        
        # Check required folders
        required_folders = [
            ('train/images', 'Train images'),
            ('train/labels', 'Train labels'),
            ('valid/images', 'Validation images'),
            ('valid/labels', 'Validation labels')
        ]
        
        all_valid = True
        missing_items = []
        
        for folder, name in required_folders:
            full_path = os.path.join(dataset_path, folder)
            if not os.path.exists(full_path):
                all_valid = False
                missing_items.append(name)
        
        # Check data.yaml file
        yaml_path = os.path.join(dataset_path, 'data.yaml')
        if not os.path.exists(yaml_path):
            all_valid = False
            missing_items.append('data.yaml file')
        
        if all_valid:
            # Try to read and parse data.yaml
            try:
                with open(yaml_path, 'r') as f:
                    data_yaml = yaml.safe_load(f)
                
                if 'names' in data_yaml:
                    self.class_names = data_yaml['names']
                    class_count = len(self.class_names)
                    validation_msg = f"✓ Dataset valid! Found {class_count} classes"
                    
                    # Initialize PPE requirements based on class names
                    self.initialize_ppe_requirements()
                else:
                    validation_msg = "⚠ data.yaml missing 'names' key"
                    all_valid = False
                
                self.validation_label.config(text=validation_msg, fg='green')
                
            except Exception as e:
                self.validation_label.config(text=f"Error reading data.yaml: {str(e)}", fg='red')
                all_valid = False
        else:
            missing_text = ", ".join(missing_items)
            self.validation_label.config(text=f"Missing: {missing_text}", fg='red')
        
        return all_valid
    
    def initialize_ppe_requirements(self):
        """Initialize PPE requirements for each class based on typical safety standards"""
        # Define typical PPE requirements
        # Minimum instances needed for safety compliance
        self.ppe_requirements = {}
        
        for i, class_name in enumerate(self.class_names):
            class_name_lower = class_name.lower()
            
            # Set minimum required instances for each PPE item
            if 'glove' in class_name_lower or 'hand' in class_name_lower:
                self.ppe_requirements[i] = {'min_instances': 2, 'critical': True, 'weight': 1.5}
            elif 'boot' in class_name_lower or 'shoe' in class_name_lower or 'foot' in class_name_lower:
                self.ppe_requirements[i] = {'min_instances': 2, 'critical': True, 'weight': 1.5}
            elif 'helmet' in class_name_lower or 'hardhat' in class_name_lower:
                self.ppe_requirements[i] = {'min_instances': 1, 'critical': True, 'weight': 2.0}
            elif 'vest' in class_name_lower or 'high-vis' in class_name_lower:
                self.ppe_requirements[i] = {'min_instances': 1, 'critical': True, 'weight': 1.8}
            elif 'goggle' in class_name_lower or 'glass' in class_name_lower:
                self.ppe_requirements[i] = {'min_instances': 1, 'critical': True, 'weight': 1.7}
            elif 'mask' in class_name_lower or 'respirator' in class_name_lower:
                self.ppe_requirements[i] = {'min_instances': 1, 'critical': True, 'weight': 1.9}
            elif 'ear' in class_name_lower or 'hearing' in class_name_lower:
                self.ppe_requirements[i] = {'min_instances': 2, 'critical': False, 'weight': 1.3}
            else:
                self.ppe_requirements[i] = {'min_instances': 1, 'critical': False, 'weight': 1.0}
    
    def update_status(self, status, color='gray'):
        """Update the training status indicator"""
        status_text = f"Status: {status}"
        self.status_label.config(text=status_text)
        self.status_indicator.config(fg=color)
        
        if status == "Training...":
            self.train_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.is_training = True
        elif status in ["Completed", "Error", "Idle"]:
            self.train_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.is_training = False
    
    def log_message(self, message):
        """Add a message to the training log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def start_training(self):
        """Start the YOLOv8 training process in a separate thread"""
        if not self.validate_dataset():
            messagebox.showerror("Error", "Please fix dataset validation errors before training")
            return
        
        try:
            epochs = int(self.epoch_entry.get())
            img_size = int(self.size_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for epochs and image size")
            return
        
        if self.is_training:
            messagebox.showwarning("Warning", "Training is already in progress")
            return
        
        self.log_text.delete(1.0, tk.END)
        
        self.training_thread = threading.Thread(
            target=self.run_training,
            args=(epochs, img_size),
            daemon=True
        )
        self.training_thread.start()
    
    def run_training(self, epochs, img_size):
        """Run YOLOv8 training (to be called in a thread)"""
        try:
            self.update_status("Training...", color='orange')
            self.log_message("Starting YOLOv8 training...")
            self.log_message(f"Parameters: Epochs={epochs}, Image Size={img_size}")
            
            self.log_message("Initializing YOLOv8 model...")
            model = YOLO("yolov8n.yaml")
            
            resume = False
            if os.path.exists(self.trained_model_path):
                resume = True
                self.log_message("Found existing model, resuming training...")
                model = YOLO(self.trained_model_path)
            
            yaml_path = os.path.join(self.dataset_path, 'data.yaml')
            
            self.log_message("Training started. This may take several minutes...")
            
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=img_size,
                pretrained=False,
                resume=resume,
                project=".",
                name="runs",
                save=True,
                exist_ok=True
            )
            
            model.save(self.trained_model_path)
            
            self.log_message(f"Training completed successfully!")
            self.log_message(f"Model saved as: {self.trained_model_path}")
            self.update_status("Completed", color='green')
            
            if os.path.exists(self.trained_model_path):
                self.load_model_btn.config(state='normal')
                self.log_message("Model is ready for detection.")
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.log_message(error_msg)
            self.update_status("Error", color='red')
            messagebox.showerror("Training Error", error_msg)
    
    def stop_training(self):
        """Stop the training process"""
        if self.is_training:
            self.update_status("Stopping...", color='orange')
            self.log_message("Training stopped by user")
            self.update_status("Idle", color='gray')
    
    # ============================================================================
    # DETECTION FUNCTIONS (WITH IMPROVED IMAGE DISPLAY)
    # ============================================================================
    
    def load_model(self):
        """Load the trained model for inference"""
        try:
            if not os.path.exists(self.trained_model_path):
                messagebox.showerror("Error", "No trained model found. Please train a model first.")
                return
            
            self.log_message("Loading trained model for detection...")
            
            self.model = YOLO(self.trained_model_path)
            
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
            else:
                yaml_path = os.path.join(self.dataset_path, 'data.yaml')
                if os.path.exists(yaml_path):
                    with open(yaml_path, 'r') as f:
                        data_yaml = yaml.safe_load(f)
                    self.class_names = data_yaml.get('names', [])
            
            # Initialize PPE requirements
            self.initialize_ppe_requirements()
            
            self.model_status_label.config(
                text=f"Model loaded: {len(self.class_names)} classes", 
                fg='green'
            )
            self.detect_btn.config(state='normal')
            
            messagebox.showinfo("Success", "Model loaded successfully!")
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.model_status_label.config(text=error_msg, fg='red')
            messagebox.showerror("Error", error_msg)
    
    def upload_single_image(self):
        """Upload a single image for detection"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.image_paths = [file_path]
            self.current_image_index = 0
            self.display_original_image(file_path)
            self.image_info_label.config(text=f"Loaded: {os.path.basename(file_path)}")
            self.update_image_navigation()
    
    def upload_image_folder(self):
        """Upload a folder of images for detection"""
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        
        if folder_path:
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            self.image_paths = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
            
            if self.image_paths:
                self.current_image_index = 0
                self.display_original_image(self.image_paths[0])
                self.image_info_label.config(
                    text=f"Loaded {len(self.image_paths)} images from folder"
                )
                self.update_image_navigation()
            else:
                messagebox.showwarning("Warning", "No valid images found in folder")
    
    def display_original_image(self, image_path):
        """Display the original image in the preview panel - FIXED"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Get canvas dimensions
            canvas_width = self.original_canvas.winfo_width() or 500
            canvas_height = self.original_canvas.winfo_height() or 400
            
            # Calculate scaling factor to fit image in canvas
            img_width, img_height = image.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            if scale < 1:  # Only scale down, not up
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Update canvas scroll region
            self.original_canvas.config(scrollregion=(0, 0, image.width, image.height))
            
            # Convert to PhotoImage
            self.current_original_image = ImageTk.PhotoImage(image)
            
            # Display on canvas
            self.original_canvas.delete("all")
            self.original_canvas.create_image(
                0, 0,
                image=self.current_original_image,
                anchor='nw'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load image: {str(e)}")
    
    def display_detected_image(self, image_path, results):
        """Display the detected image with bounding boxes - FIXED"""
        try:
            # Load image with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot read image: {image_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_height, original_width = image.shape[:2]
            
            # Draw bounding boxes from results
            if hasattr(results[0], 'boxes'):
                boxes = results[0].boxes
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    # Get class name and color
                    class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class{cls}"
                    color = self.get_class_color(cls)
                    
                    # Scale coordinates back to original image size if needed
                    scale_x = original_width / 640  # Assuming model input size is 640
                    scale_y = original_height / 640
                    
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    
                    # Draw rectangle
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    
                    # Create label
                    label = f"{class_name}: {conf:.2f}"
                    
                    # Get text size
                    font_scale = max(0.5, min(1.0, original_width / 1000))
                    thickness = max(1, int(original_width / 500))
                    
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        image,
                        (int(x1), int(y1) - text_height - 10),
                        (int(x1) + text_width, int(y1)),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        image,
                        label,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (255, 255, 255),
                        thickness
                    )
            
            # Convert to PIL Image
            detected_image = Image.fromarray(image)
            
            # Get canvas dimensions
            canvas_width = self.detected_canvas.winfo_width() or 500
            canvas_height = self.detected_canvas.winfo_height() or 400
            
            # Calculate scaling factor
            img_width, img_height = detected_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            if scale < 1:  # Only scale down, not up
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                detected_image = detected_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Update canvas scroll region
            self.detected_canvas.config(scrollregion=(0, 0, detected_image.width, detected_image.height))
            
            # Convert to PhotoImage
            self.current_detected_image = ImageTk.PhotoImage(detected_image)
            
            # Display on canvas
            self.detected_canvas.delete("all")
            self.detected_canvas.create_image(
                0, 0,
                image=self.current_detected_image,
                anchor='nw'
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot display detected image: {str(e)}")
    
    def get_class_color(self, class_id):
        """Get a unique color for each class"""
        colors = [
            (255, 0, 0),    # Red - Critical items
            (0, 255, 0),    # Green - Non-critical items
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (128, 0, 0),    # Maroon
            (0, 128, 0),    # Dark Green
            (0, 0, 128),    # Navy
            (128, 128, 0)   # Olive
        ]
        return colors[class_id % len(colors)]
    
    def run_detection(self):
        """Run detection on the loaded image(s)"""
        if not self.model:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        if not self.image_paths:
            messagebox.showwarning("Warning", "Please upload an image or folder first")
            return
        
        try:
            # Clear previous results
            self.detection_results = []
            self.detection_results_list = []
            
            # Run detection on all images
            for i, image_path in enumerate(self.image_paths):
                results = self.model(image_path, conf=0.25)  # Lower confidence threshold for better detection
                self.detection_results_list.append(results)
                
                # Display first image results
                if i == 0:
                    self.display_detected_image(image_path, results)
                    self.process_detection_results(results, image_path)
                    
                    if self.save_var.get():
                        self.save_detection_output(results, image_path)
            
            # Enable navigation for multiple images
            self.update_image_navigation()
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
    
    def process_detection_results(self, results, image_path):
        """Process and display detection results - IMPROVED CALCULATION"""
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        
        # Initialize counters
        detected_classes = {}
        class_counts = {}
        all_classes = set(range(len(self.class_names))) if self.class_names else set()
        detected_class_ids = set()
        
        # Analyze results
        if hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            
            # Count detections by class
            for box in boxes:
                cls = int(box.cls[0].item())
                conf = box.conf[0].item()
                
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class{cls}"
                
                if class_name not in detected_classes:
                    detected_classes[class_name] = []
                    class_counts[cls] = 0
                
                detected_classes[class_name].append(conf)
                class_counts[cls] = class_counts.get(cls, 0) + 1
                detected_class_ids.add(cls)
            
            # Build results text
            results_text = f"📊 DETECTION RESULTS ANALYSIS\n"
            results_text += "=" * 50 + "\n\n"
            results_text += f"📁 Image: {os.path.basename(image_path)}\n"
            results_text += f"📅 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            results_text += "=" * 50 + "\n\n"
            
            # Detected objects with detailed information
            if detected_classes:
                results_text += "✅ DETECTED PPE ITEMS:\n"
                results_text += "-" * 30 + "\n"
                
                for class_name, confidences in detected_classes.items():
                    count = len(confidences)
                    avg_conf = sum(confidences) / count if count > 0 else 0
                    max_conf = max(confidences) if confidences else 0
                    
                    # Get class ID for requirements
                    cls_id = next((i for i, name in enumerate(self.class_names) if name == class_name), None)
                    
                    if cls_id is not None and cls_id in self.ppe_requirements:
                        req_min = self.ppe_requirements[cls_id]['min_instances']
                        is_critical = self.ppe_requirements[cls_id]['critical']
                        status = "✓ SATISFIED" if count >= req_min else "⚠ INSUFFICIENT"
                        critical_mark = "🔴" if is_critical else "🟡"
                        
                        results_text += f"{critical_mark} {class_name}:\n"
                        results_text += f"   • Instances: {count}/{req_min} {status}\n"
                        results_text += f"   • Avg Confidence: {avg_conf:.1%}\n"
                        results_text += f"   • Max Confidence: {max_conf:.1%}\n"
                    else:
                        results_text += f"🟢 {class_name}:\n"
                        results_text += f"   • Instances: {count}\n"
                        results_text += f"   • Avg Confidence: {avg_conf:.1%}\n"
            else:
                results_text += "❌ NO PPE ITEMS DETECTED\n"
            
            results_text += "\n" + "=" * 50 + "\n\n"
            
            # Missing PPE items analysis
            missing_classes = all_classes - detected_class_ids
            if missing_classes:
                results_text += "⚠ MISSING PPE ITEMS:\n"
                results_text += "-" * 30 + "\n"
                
                critical_missing = 0
                for cls in missing_classes:
                    if cls < len(self.class_names):
                        class_name = self.class_names[cls]
                        if cls in self.ppe_requirements and self.ppe_requirements[cls]['critical']:
                            results_text += f"🔴 {class_name} - CRITICAL MISSING\n"
                            critical_missing += 1
                        else:
                            results_text += f"🟡 {class_name} - Recommended\n"
            else:
                results_text += "✅ ALL PPE ITEMS PRESENT\n"
            
            results_text += "\n" + "=" * 50 + "\n\n"
            
            # Calculate comprehensive safety compliance
            safety_percentage, details = self.calculate_safety_compliance(class_counts, all_classes)
            
            results_text += "📈 SAFETY COMPLIANCE ANALYSIS\n"
            results_text += "-" * 30 + "\n"
            results_text += f"Overall Safety Score: {safety_percentage:.1f}%\n\n"
            
            # Add breakdown
            results_text += "Breakdown:\n"
            for detail in details:
                results_text += f"• {detail}\n"
            
            # Store results for visualization
            self.detection_results = {
                'detected': detected_classes,
                'missing': missing_classes,
                'class_counts': class_counts,
                'safety_percentage': safety_percentage,
                'total_classes': len(all_classes),
                'critical_missing': critical_missing
            }
            
        else:
            results_text = "❌ No detections found in the image.\n"
            results_text += "=" * 50 + "\n\n"
            results_text += "Possible reasons:\n"
            results_text += "1. Image quality is poor\n"
            results_text += "2. PPE items are not visible\n"
            results_text += "3. Confidence threshold is too high\n"
            
            self.detection_results = {
                'detected': {},
                'missing': all_classes,
                'class_counts': {},
                'safety_percentage': 0.0,
                'total_classes': len(all_classes),
                'critical_missing': len(all_classes)
            }
        
        # Display results
        self.results_text.insert(tk.END, results_text)
        
        # Update safety label
        safety_level = "GOOD" if safety_percentage >= 80 else "MODERATE" if safety_percentage >= 60 else "POOR"
        safety_color = 'green' if safety_percentage >= 80 else 'orange' if safety_percentage >= 60 else 'red'
        
        safety_text = f"Safety Compliance: {safety_percentage:.1f}% ({safety_level})"
        self.safety_label.config(text=safety_text, fg=safety_color)
        
        # Update explanation
        self.update_explanation()
        
        # Update visualization
        self.update_visualization()
    
    def calculate_safety_compliance(self, class_counts, all_classes):
        """Calculate comprehensive safety compliance percentage"""
        if not all_classes:
            return 0.0, ["No PPE classes defined"]
        
        total_score = 0
        max_score = 0
        details = []
        
        for cls in all_classes:
            if cls in self.ppe_requirements:
                req = self.ppe_requirements[cls]
                detected = class_counts.get(cls, 0)
                required = req['min_instances']
                weight = req['weight']
                
                # Calculate score for this class
                if detected >= required:
                    class_score = weight * 100  # Full points
                    status = "✓ Complete"
                elif detected > 0:
                    class_score = weight * 50 * (detected / required)  # Partial points
                    status = f"⚠ Partial ({detected}/{required})"
                else:
                    class_score = 0
                    status = "❌ Missing"
                
                total_score += class_score
                max_score += weight * 100
                
                # Add to details
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class{cls}"
                details.append(f"{class_name}: {status} ({class_score/weight:.0f} pts)")
            else:
                # Default calculation for classes without specific requirements
                detected = class_counts.get(cls, 0)
                if detected > 0:
                    total_score += 100
                    status = "✓ Present"
                else:
                    status = "❌ Missing"
                max_score += 100
                
                class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class{cls}"
                details.append(f"{class_name}: {status}")
        
        # Calculate overall percentage
        if max_score > 0:
            safety_percentage = (total_score / max_score) * 100
        else:
            safety_percentage = 0.0
        
        # Round to 1 decimal place
        safety_percentage = round(safety_percentage, 1)
        
        return safety_percentage, details
    
    def update_explanation(self):
        """Update the explanation text based on detection results"""
        explanation = ""
        
        if not self.detection_results:
            explanation = "No detection results available. Please run detection first."
        else:
            safety_percentage = self.detection_results['safety_percentage']
            critical_missing = self.detection_results.get('critical_missing', 0)
            
            if safety_percentage >= 90:
                explanation = "✅ EXCELLENT SAFETY COMPLIANCE\n\n"
                explanation += "All critical PPE items are properly detected and meet minimum requirements. "
                explanation += "The work environment is considered SAFE according to OSHA standards."
                explanation += "\n\nRecommendations: Continue current safety practices."
                
            elif safety_percentage >= 75:
                explanation = "⚠ MODERATE SAFETY COMPLIANCE\n\n"
                explanation += "Most PPE items are detected, but some may be missing or insufficient. "
                explanation += "Review the missing items and ensure all required PPE is properly worn."
                
                if critical_missing > 0:
                    explanation += f"\n\n⚠ CRITICAL ISSUE: {critical_missing} critical PPE item(s) missing. "
                    explanation += "Immediate action required."
                    
                explanation += "\n\nRecommendations:\n"
                explanation += "1. Verify all critical PPE items are present\n"
                explanation += "2. Check PPE fit and proper usage\n"
                explanation += "3. Conduct safety brief before proceeding"
                
            elif safety_percentage >= 50:
                explanation = "⚠ LOW SAFETY COMPLIANCE\n\n"
                explanation += "Multiple PPE items are missing or insufficient. "
                explanation += "Significant safety risks present."
                
                if critical_missing > 0:
                    explanation += f"\n\n🔴 CRITICAL ALERT: {critical_missing} critical PPE item(s) missing. "
                    explanation += "WORK SHOULD NOT PROCEED."
                    
                explanation += "\n\nImmediate Actions Required:\n"
                explanation += "1. STOP all work immediately\n"
                explanation += "2. Ensure all missing PPE is obtained and worn\n"
                explanation += "3. Conduct safety training if needed\n"
                explanation += "4. Supervisory approval required to resume"
                
            else:
                explanation = "❌ POOR SAFETY COMPLIANCE\n\n"
                explanation += "Inadequate PPE detection indicates serious safety violations. "
                explanation += "High risk of injury or accident."
                
                if critical_missing > 0:
                    explanation += f"\n\n🔴 EMERGENCY ALERT: {critical_missing} critical PPE item(s) missing. "
                    explanation += "IMMEDIATE EVACUATION MAY BE NECESSARY."
                    
                explanation += "\n\nEMERGENCY PROCEDURES:\n"
                explanation += "1. EVACUATE area if immediate danger exists\n"
                explanation += "2. Report to safety officer immediately\n"
                explanation += "3. Full safety audit required\n"
                explanation += "4. All personnel must complete safety retraining"
        
        self.explanation_text.delete(1.0, tk.END)
        self.explanation_text.insert(tk.END, explanation)
    
    def update_visualization(self):
        """Update the visualization chart with detection results"""
        if not self.detection_results:
            return
        
        # Clear previous chart
        self.chart_canvas.delete("all")
        
        # Create figure for matplotlib
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Get data for visualization
        detected_count = len(self.detection_results['detected'])
        missing_count = len(self.detection_results['missing'])
        safety_percentage = self.detection_results['safety_percentage']
        class_counts = self.detection_results.get('class_counts', {})
        
        # Chart 1: Detected vs Missing (Pie Chart)
        if detected_count + missing_count > 0:
            sizes = [detected_count, missing_count]
            labels = [f'Detected\n{detected_count}', f'Missing\n{missing_count}']
            colors = ['#4CAF50', '#f44336']
            explode = (0.1, 0)  # explode the detected slice
            
            ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
            ax1.set_title('PPE Items Detection Status', fontweight='bold')
            ax1.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle
        else:
            ax1.text(0.5, 0.5, 'No Data Available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=12)
            ax1.set_title('PPE Items Detection Status', fontweight='bold')
        
        # Chart 2: Safety Compliance Gauge
        ax2 = plt.subplot(1, 2, 2, projection='polar')
        
        # Create gauge chart
        categories = ['Excellent', 'Good', 'Moderate', 'Poor', 'Critical']
        N = len(categories)
        
        # What is the angle of each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Initialise the spider plot
        ax2.set_theta_offset(np.pi / 2)
        ax2.set_theta_direction(-1)
        
        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], categories)
        
        # Draw ylabels
        ax2.set_rlabel_position(0)
        plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=7)
        plt.ylim(0, 100)
        
        # Plot safety percentage
        safety_angle = (safety_percentage / 100) * 2 * np.pi
        ax2.plot([0, safety_angle], [0, safety_percentage], color='red', linewidth=3)
        ax2.fill_between([0, safety_angle], 0, safety_percentage, color='red', alpha=0.3)
        
        # Add value text
        ax2.text(0, 0, f'{safety_percentage:.1f}%', 
                horizontalalignment='center', verticalalignment='center',
                fontsize=20, fontweight='bold')
        
        ax2.set_title('Safety Compliance Gauge', fontweight='bold', pad=20)
        
        # Adjust layout
        plt.tight_layout()
        
        # Convert matplotlib figure to tkinter canvas
        canvas = FigureCanvasTkAgg(fig, master=self.chart_canvas)
        canvas.draw()
        
        # Get the tkinter widget and pack it
        chart_widget = canvas.get_tk_widget()
        chart_widget.pack(fill='both', expand=True)
    
    def save_detection_output(self, results, image_path):
        """Save the detection output to a file"""
        try:
            # Create output directory
            output_dir = "detection_output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save the image with detections
            original_image = cv2.imread(image_path)
            
            if hasattr(results[0], 'boxes'):
                boxes = results[0].boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class{cls}"
                    color = self.get_class_color(cls)
                    color_bgr = (color[2], color[1], color[0])
                    
                    # Draw rectangle
                    cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 3)
                    
                    # Create label
                    label = f"{class_name}: {conf:.2f}"
                    
                    # Draw label
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                    )
                    
                    cv2.rectangle(
                        original_image,
                        (int(x1), int(y1) - text_height - 10),
                        (int(x1) + text_width, int(y1)),
                        color_bgr,
                        -1
                    )
                    
                    cv2.putText(
                        original_image,
                        label,
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
            
            # Save the image
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
            cv2.imwrite(output_path, original_image)
            
            # Save results to text file
            results_path = os.path.join(output_dir, f"{base_name}_results.txt")
            with open(results_path, 'w') as f:
                f.write(self.results_text.get(1.0, tk.END))
            
            # Log the save action
            self.log_message(f"Saved detection output to: {output_path}")
            
        except Exception as e:
            self.log_message(f"Error saving output: {str(e)}")


# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main function to run the application"""
    # Create the main window
    root = tk.Tk()
    
    # Create the application
    app = PPE_Detection_App(root)
    
    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()