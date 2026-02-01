"""
PPE Image Detection Desktop Application
Complete YOLOv8-based PPE detection system with training and inference
Enhanced GUI with professional results section
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
        self.root.title("PPE Detection System - HeadGear Sentinel v2.0")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2c3e50')
        
        # Create main frame with scrollbars
        self.main_frame = tk.Frame(root, bg='#2c3e50')
        self.main_frame.pack(fill='both', expand=True)
        
        # Create canvas for scrolling
        self.canvas = tk.Canvas(self.main_frame, bg='#2c3e50', highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self.main_frame, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Create scrollable frame
        self.scrollable_frame = tk.Frame(self.canvas, bg='#2c3e50')
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Bind configure event to update scroll region
        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        
        # Bind mouse wheel for scrolling
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self._on_shift_mousewheel)
        
        # Application state variables
        self.dataset_path = ""
        self.model = None
        self.trained_model_path = "trained_model.pt"
        self.class_names = []
        self.detection_results = []
        self.training_thread = None
        self.is_training = False
        self.ppe_requirements = {}  # Store PPE requirements for each class
        self.current_model_path = ""
        
        # Image display variables
        self.original_image_pil = None
        self.detected_image_pil = None
        self.current_original_photo = None
        self.current_detected_photo = None
        
        # Create the main notebook (tab system)
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Style configuration
        self.setup_styles()
        
        # Create the three main sections
        self.create_training_section()
        self.create_detection_section()
        self.create_results_section()
        
        # Set initial status
        self.update_status("Idle")
        
        # Schedule canvas initialization after GUI is fully loaded
        self.root.after(100, self.initialize_canvas_placeholders)
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _on_shift_mousewheel(self, event):
        """Handle shift+mouse wheel horizontal scrolling"""
        self.canvas.xview_scroll(int(-1*(event.delta/120)), "units")
    
    def setup_styles(self):
        """Configure custom styles for the application"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Custom.TNotebook', background='#34495e')
        style.configure('Custom.TNotebook.Tab', 
                       background='#2c3e50', 
                       foreground='white',
                       padding=[20, 5])
        style.map('Custom.TNotebook.Tab', 
                 background=[('selected', '#3498db')],
                 foreground=[('selected', 'white')])
        
        # Apply style to notebook
        self.notebook.configure(style='Custom.TNotebook')
    
    def create_training_section(self):
        """Create the training section with enhanced GUI"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🏋️ TRAINING")
        
        # Main container with gradient background
        main_container = tk.Frame(training_frame, bg='#ecf0f1')
        main_container.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Header
        header_frame = tk.Frame(main_container, bg='#3498db', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="🧠 PPE Model Training", 
                font=('Arial', 20, 'bold'), bg='#3498db', fg='white').pack(pady=20)
        
        # Content area
        content_frame = tk.Frame(main_container, bg='white')
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left panel - Configuration
        config_frame = tk.Frame(content_frame, bg='white', relief='flat')
        config_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Dataset configuration card
        dataset_card = tk.Frame(config_frame, bg='#f8f9fa', relief='groove', borderwidth=2)
        dataset_card.pack(fill='x', pady=(0, 15))
        
        tk.Label(dataset_card, text="📂 Dataset Configuration", 
                font=('Arial', 14, 'bold'), bg='#f8f9fa').pack(pady=15, padx=20, anchor='w')
        
        # Dataset selection
        dataset_row = tk.Frame(dataset_card, bg='#f8f9fa')
        dataset_row.pack(fill='x', padx=20, pady=(0, 10))
        
        tk.Label(dataset_row, text="Dataset Folder:", bg='#f8f9fa', 
                font=('Arial', 11)).pack(side='left', padx=(0, 10))
        
        self.dataset_entry = tk.Entry(dataset_row, width=40, font=('Arial', 10),
                                     relief='solid', borderwidth=1)
        self.dataset_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        browse_btn = tk.Button(dataset_row, text="Browse", 
                              command=self.browse_dataset, 
                              bg='#3498db', fg='white', font=('Arial', 10),
                              relief='raised', borderwidth=2, padx=15)
        browse_btn.pack(side='right')
        
        # Format information
        format_frame = tk.Frame(dataset_card, bg='#e8f4f8', relief='solid', borderwidth=1)
        format_frame.pack(fill='x', padx=20, pady=10)
        
        format_text = """📁 Required Dataset Structure:

dataset/
├── data.yaml
├── train/images/
├── train/labels/
├── valid/images/
├── valid/labels/
└── test/images/ (optional)"""
        
        format_label = tk.Label(format_frame, text=format_text, justify='left', 
                               bg='#e8f4f8', font=('Courier', 9), padx=10, pady=10)
        format_label.pack()
        
        # Validation button
        validate_btn = tk.Button(dataset_card, text="🔍 Validate Dataset", 
                                command=self.validate_dataset, 
                                bg='#2ecc71', fg='white', font=('Arial', 11, 'bold'),
                                relief='raised', borderwidth=2, padx=20, pady=8)
        validate_btn.pack(pady=15)
        
        self.validation_label = tk.Label(dataset_card, text="", bg='#f8f9fa',
                                        font=('Arial', 10))
        self.validation_label.pack(pady=(0, 15))
        
        # Training parameters card
        params_card = tk.Frame(config_frame, bg='#f8f9fa', relief='groove', borderwidth=2)
        params_card.pack(fill='x', pady=15)
        
        tk.Label(params_card, text="⚙️ Training Parameters", 
                font=('Arial', 14, 'bold'), bg='#f8f9fa').pack(pady=15, padx=20, anchor='w')
        
        # Epochs input
        epoch_frame = tk.Frame(params_card, bg='#f8f9fa')
        epoch_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(epoch_frame, text="Epochs:", bg='#f8f9fa', 
                font=('Arial', 11)).pack(side='left', padx=(0, 20))
        
        self.epoch_entry = tk.Entry(epoch_frame, width=15, font=('Arial', 10),
                                   relief='solid', borderwidth=1)
        self.epoch_entry.insert(0, "100")
        self.epoch_entry.pack(side='left')
        
        # Image size selector
        size_frame = tk.Frame(params_card, bg='#f8f9fa')
        size_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(size_frame, text="Image Size:", bg='#f8f9fa', 
                font=('Arial', 11)).pack(side='left', padx=(0, 20))
        
        self.size_var = tk.StringVar(value="640")
        size_options = ["320", "416", "512", "640", "768"]
        size_menu = ttk.Combobox(size_frame, textvariable=self.size_var, 
                                values=size_options, state='readonly',
                                width=15, font=('Arial', 10))
        size_menu.pack(side='left')
        
        # Right panel - Training controls
        control_frame = tk.Frame(content_frame, bg='white', relief='flat')
        control_frame.pack(side='right', fill='both', expand=True)
        
        # Status card
        status_card = tk.Frame(control_frame, bg='#f8f9fa', relief='groove', borderwidth=2)
        status_card.pack(fill='x', pady=(0, 15))
        
        tk.Label(status_card, text="📊 Training Status", 
                font=('Arial', 14, 'bold'), bg='#f8f9fa').pack(pady=15, padx=20, anchor='w')
        
        # Status indicator
        self.status_frame = tk.Frame(status_card, bg='#e0e0e0', height=50)
        self.status_frame.pack(fill='x', padx=20, pady=(0, 20))
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(self.status_frame, text="Status: Idle", 
                                    bg='#e0e0e0', font=('Arial', 12, 'bold'))
        self.status_label.pack(side='left', padx=20)
        
        self.status_indicator = tk.Label(self.status_frame, text="●", 
                                        fg='gray', font=('Arial', 24))
        self.status_indicator.pack(side='right', padx=20)
        
        # Training buttons
        button_frame = tk.Frame(status_card, bg='#f8f9fa')
        button_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        self.train_btn = tk.Button(button_frame, text="▶ Start Training", 
                                  command=self.start_training, 
                                  bg='#2ecc71', fg='white', font=('Arial', 12, 'bold'),
                                  relief='raised', borderwidth=2, padx=30, pady=10)
        self.train_btn.pack(side='left', padx=(0, 10))
        
        self.stop_btn = tk.Button(button_frame, text="⏹ Stop Training", 
                                 command=self.stop_training, 
                                 bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
                                 relief='raised', borderwidth=2, padx=30, pady=10,
                                 state='disabled')
        self.stop_btn.pack(side='left')
        
        # Training logs card
        log_card = tk.Frame(control_frame, bg='#f8f9fa', relief='groove', borderwidth=2)
        log_card.pack(fill='both', expand=True)
        
        tk.Label(log_card, text="📝 Training Logs", 
                font=('Arial', 14, 'bold'), bg='#f8f9fa').pack(pady=15, padx=20, anchor='w')
        
        # Create scrolled text widget for logs
        self.log_text = scrolledtext.ScrolledText(log_card, height=20, 
                                                 bg='#2c3e50', fg='#ecf0f1',
                                                 font=('Courier', 9))
        self.log_text.pack(fill='both', expand=True, padx=20, pady=(0, 20))
    
    def create_detection_section(self):
        """Create the detection section with enhanced GUI and model selection"""
        detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(detection_frame, text="🔍 DETECTION")
        
        # Main container
        main_container = tk.Frame(detection_frame, bg='#ecf0f1')
        main_container.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Header
        header_frame = tk.Frame(main_container, bg='#9b59b6', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="👁️ PPE Detection", 
                font=('Arial', 20, 'bold'), bg='#9b59b6', fg='white').pack(pady=20)
        
        # Content area
        content_frame = tk.Frame(main_container, bg='white')
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left panel - Controls
        control_panel = tk.Frame(content_frame, bg='white', relief='flat')
        control_panel.pack(side='left', fill='both', expand=False, padx=(0, 20))
        
        # Model loading card
        model_card = tk.Frame(control_panel, bg='#f8f9fa', relief='groove', borderwidth=2)
        model_card.pack(fill='x', pady=(0, 15))
        
        tk.Label(model_card, text="🤖 Model Selection", 
                font=('Arial', 14, 'bold'), bg='#f8f9fa').pack(pady=15, padx=20, anchor='w')
        
        # Model loading buttons
        load_frame = tk.Frame(model_card, bg='#f8f9fa')
        load_frame.pack(fill='x', padx=20, pady=10)
        
        # Button 1: Load default trained model
        default_btn = tk.Button(load_frame, text="📁 Load Trained Model", 
                               command=self.load_default_model, 
                               bg='#3498db', fg='white', font=('Arial', 11),
                               relief='raised', borderwidth=2, padx=20, pady=10,
                               width=20)
        default_btn.pack(pady=5)
        
        # Button 2: Select custom model
        custom_btn = tk.Button(load_frame, text="📂 Select Custom Model", 
                              command=self.select_custom_model, 
                              bg='#3498db', fg='white', font=('Arial', 11),
                              relief='raised', borderwidth=2, padx=20, pady=10,
                              width=20)
        custom_btn.pack(pady=5)
        
        # Model status display
        status_frame = tk.Frame(model_card, bg='#e8f4f8', relief='solid', borderwidth=1)
        status_frame.pack(fill='x', padx=20, pady=15)
        
        self.model_status_label = tk.Label(status_frame, text="⏳ No model loaded", 
                                          bg='#e8f4f8', fg='#7f8c8d', 
                                          font=('Arial', 11, 'bold'))
        self.model_status_label.pack(pady=10, padx=10)
        
        self.model_details_label = tk.Label(status_frame, text="", 
                                           bg='#e8f4f8', fg='#95a5a6',
                                           font=('Arial', 9))
        self.model_details_label.pack(pady=(0, 10), padx=10)
        
        # Image upload card
        upload_card = tk.Frame(control_panel, bg='#f8f9fa', relief='groove', borderwidth=2)
        upload_card.pack(fill='x', pady=15)
        
        tk.Label(upload_card, text="📤 Image Upload", 
                font=('Arial', 14, 'bold'), bg='#f8f9fa').pack(pady=15, padx=20, anchor='w')
        
        upload_frame = tk.Frame(upload_card, bg='#f8f9fa')
        upload_frame.pack(fill='x', padx=20, pady=10)
        
        # Single image upload
        single_btn = tk.Button(upload_frame, text="🖼️ Single Image", 
                              command=self.upload_single_image, 
                              bg='#2ecc71', fg='white', font=('Arial', 11),
                              relief='raised', borderwidth=2, padx=20, pady=10,
                              width=15)
        single_btn.pack(pady=5)
        
        # Folder upload
        folder_btn = tk.Button(upload_frame, text="📁 Image Folder", 
                              command=self.upload_image_folder, 
                              bg='#2ecc71', fg='white', font=('Arial', 11),
                              relief='raised', borderwidth=2, padx=20, pady=10,
                              width=15)
        folder_btn.pack(pady=5)
        
        # Image info display
        self.image_info_label = tk.Label(upload_card, text="", bg='#f8f9fa',
                                        font=('Arial', 10), fg='#2c3e50')
        self.image_info_label.pack(pady=(0, 15))
        
        # Detection button
        self.detect_btn = tk.Button(control_panel, text="🚀 Run Detection", 
                                   command=self.run_detection, 
                                   bg='#e74c3c', fg='white', font=('Arial', 14, 'bold'),
                                   relief='raised', borderwidth=3, padx=30, pady=15,
                                   state='disabled')
        self.detect_btn.pack(pady=20)
        
        # Save option
        self.save_var = tk.BooleanVar(value=True)
        save_check = tk.Checkbutton(control_panel, text="💾 Save detected output", 
                                   variable=self.save_var, bg='white',
                                   font=('Arial', 10), selectcolor='#3498db')
        save_check.pack(pady=10)
        
        # Right panel - Image preview
        preview_panel = tk.Frame(content_frame, bg='white', relief='flat')
        preview_panel.pack(side='right', fill='both', expand=True)
        
        # Image preview card
        preview_card = tk.Frame(preview_panel, bg='#f8f9fa', relief='groove', borderwidth=2)
        preview_card.pack(fill='both', expand=True)
        
        tk.Label(preview_card, text="👁️‍🗨️ Image Preview", 
                font=('Arial', 14, 'bold'), bg='#f8f9fa').pack(pady=15, padx=20, anchor='w')
        
        # Image container with two columns
        image_container = tk.Frame(preview_card, bg='#f8f9fa')
        image_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Original image section
        orig_frame = tk.LabelFrame(image_container, text="Original Image", 
                                  font=('Arial', 12, 'bold'),
                                  bg='#2c3e50', fg='white', 
                                  relief='solid', borderwidth=1)
        orig_frame.pack(side='left', fill='both', expand=True, padx=(0, 10), pady=5)
        
        # Frame for original image with fixed size (640x640)
        self.orig_image_frame = tk.Frame(orig_frame, bg='#34495e', width=640, height=640)
        self.orig_image_frame.pack(fill='both', expand=True, padx=5, pady=5)
        self.orig_image_frame.pack_propagate(False)
        
        # Canvas for original image (for better control)
        self.original_canvas = tk.Canvas(self.orig_image_frame, bg='#34495e', highlightthickness=0,
                                        width=630, height=630)
        self.original_canvas.pack(fill='both', expand=True)
        
        # Detected image section
        det_frame = tk.LabelFrame(image_container, text="Detected Image", 
                                 font=('Arial', 12, 'bold'),
                                 bg='#2c3e50', fg='white', 
                                 relief='solid', borderwidth=1)
        det_frame.pack(side='right', fill='both', expand=True, pady=5)
        
        # Frame for detected image with fixed size (640x640)
        self.det_image_frame = tk.Frame(det_frame, bg='#34495e', width=640, height=640)
        self.det_image_frame.pack(fill='both', expand=True, padx=5, pady=5)
        self.det_image_frame.pack_propagate(False)
        
        # Canvas for detected image
        self.detected_canvas = tk.Canvas(self.det_image_frame, bg='#34495e', highlightthickness=0,
                                        width=630, height=630)
        self.detected_canvas.pack(fill='both', expand=True)
        
        # Navigation buttons
        nav_frame = tk.Frame(preview_card, bg='#f8f9fa')
        nav_frame.pack(pady=10)
        
        self.prev_btn = tk.Button(nav_frame, text="◀ Previous",
                                 command=self.show_previous_image,
                                 bg='#3498db', fg='white', font=('Arial', 11, 'bold'),
                                 relief='raised', borderwidth=2, state='disabled',
                                 padx=15, pady=8)
        self.prev_btn.pack(side='left', padx=5)
        
        self.next_btn = tk.Button(nav_frame, text="Next ▶",
                                 command=self.show_next_image,
                                 bg='#3498db', fg='white', font=('Arial', 11, 'bold'),
                                 relief='raised', borderwidth=2, state='disabled',
                                 padx=15, pady=8)
        self.next_btn.pack(side='left', padx=5)
        
        self.image_counter_label = tk.Label(nav_frame, text="Image 0/0",
                                           bg='#f8f9fa', font=('Arial', 12, 'bold'),
                                           fg='#2c3e50')
        self.image_counter_label.pack(side='left', padx=20)
        
        # Current images
        self.current_original_image = None
        self.current_detected_image = None
        self.image_paths = []
        self.current_image_index = 0
        self.detection_results_list = []
        
        # Add default placeholders (will be called after GUI loads)
    
    def initialize_canvas_placeholders(self):
        """Initialize canvas placeholders after GUI is fully loaded"""
        self.show_placeholder_images()
    
    def show_placeholder_images(self):
        """Show placeholder text when no images are loaded"""
        # Clear both canvases
        self.original_canvas.delete("all")
        self.detected_canvas.delete("all")
        
        # Get canvas dimensions
        orig_width = 630
        orig_height = 630
        
        det_width = 630
        det_height = 630
        
        # Draw placeholder on original canvas
        self.original_canvas.create_text(
            orig_width // 2, orig_height // 2,
            text="Upload an image to start\n(Click 'Single Image' or 'Image Folder')",
            fill='#95a5a6', font=('Arial', 14),
            justify='center'
        )
        
        # Draw placeholder on detected canvas
        self.detected_canvas.create_text(
            det_width // 2, det_height // 2,
            text="Detection results will appear here\n(After uploading, click 'Run Detection')",
            fill='#95a5a6', font=('Arial', 14),
            justify='center'
        )
    
    def create_results_section(self):
        """Create the enhanced results and analysis section"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📊 RESULTS")
        
        # Main container
        main_container = tk.Frame(results_frame, bg='#ecf0f1')
        main_container.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Header
        header_frame = tk.Frame(main_container, bg='#e67e22', height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="📈 Detection Results & Analysis", 
                font=('Arial', 20, 'bold'), bg='#e67e22', fg='white').pack(pady=20)
        
        # Content area with two columns
        content_frame = tk.Frame(main_container, bg='white')
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left column - Text results
        left_column = tk.Frame(content_frame, bg='white')
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Results summary card
        summary_card = tk.Frame(left_column, bg='#f8f9fa', relief='groove', borderwidth=2)
        summary_card.pack(fill='both', expand=True, pady=(0, 15))
        
        tk.Label(summary_card, text="📋 Detection Summary", 
                font=('Arial', 14, 'bold'), bg='#f8f9fa').pack(pady=15, padx=20, anchor='w')
        
        # Two-column summary
        summary_container = tk.Frame(summary_card, bg='#f8f9fa')
        summary_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Detected items
        detected_frame = tk.Frame(summary_container, bg='#e8f6f3', relief='solid', borderwidth=1)
        detected_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5), pady=5)
        
        tk.Label(detected_frame, text="✅ DETECTED PPE ITEMS", font=('Arial', 12, 'bold'), 
                bg='#27ae60', fg='white').pack(fill='x', pady=5)
        
        self.detected_listbox = tk.Listbox(detected_frame, height=8, bg='#e8f6f3',
                                          font=('Arial', 10), selectmode=tk.SINGLE,
                                          relief='flat')
        detected_scrollbar = tk.Scrollbar(detected_frame)
        detected_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.detected_listbox.config(yscrollcommand=detected_scrollbar.set)
        detected_scrollbar.config(command=self.detected_listbox.yview)
        self.detected_listbox.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Missing items
        missing_frame = tk.Frame(summary_container, bg='#fdebd0', relief='solid', borderwidth=1)
        missing_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0), pady=5)
        
        tk.Label(missing_frame, text="⚠ MISSING PPE ITEMS", font=('Arial', 12, 'bold'), 
                bg='#e67e22', fg='white').pack(fill='x', pady=5)
        
        self.missing_listbox = tk.Listbox(missing_frame, height=8, bg='#fdebd0',
                                         font=('Arial', 10), selectmode=tk.SINGLE,
                                         relief='flat')
        missing_scrollbar = tk.Scrollbar(missing_frame)
        missing_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.missing_listbox.config(yscrollcommand=missing_scrollbar.set)
        missing_scrollbar.config(command=self.missing_listbox.yview)
        self.missing_listbox.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Configure grid weights
        summary_container.columnconfigure(0, weight=1)
        summary_container.columnconfigure(1, weight=1)
        summary_container.rowconfigure(0, weight=1)
        
        # Summary stats
        self.summary_stats_label = tk.Label(summary_card, text="📊 No detection results yet", 
                                           bg='#f8f9fa', font=('Arial', 11, 'bold'),
                                           fg='#2c3e50')
        self.summary_stats_label.pack(pady=10)
        
        # Safety analysis card
        safety_card = tk.Frame(left_column, bg='#f8f9fa', relief='groove', borderwidth=2)
        safety_card.pack(fill='both', expand=True)
        
        tk.Label(safety_card, text="🛡️ Safety Compliance Analysis", 
                font=('Arial', 14, 'bold'), bg='#f8f9fa').pack(pady=15, padx=20, anchor='w')
        
        # Safety percentage display
        safety_display = tk.Frame(safety_card, bg='#2c3e50', relief='solid', borderwidth=1)
        safety_display.pack(fill='x', padx=20, pady=10)
        
        self.safety_percentage_label = tk.Label(safety_display, 
                                               text="0.0%",
                                               font=('Arial', 36, 'bold'),
                                               bg='#2c3e50', fg='#ecf0f1')
        self.safety_percentage_label.pack(pady=10)
        
        self.safety_status_label = tk.Label(safety_display,
                                          text="NOT CALCULATED",
                                          font=('Arial', 16, 'bold'),
                                          bg='#2c3e50', fg='#bdc3c7')
        self.safety_status_label.pack(pady=(0, 10))
        
        # Safety breakdown
        breakdown_frame = tk.Frame(safety_card, bg='#f8f9fa')
        breakdown_frame.pack(fill='x', padx=20, pady=10)
        
        self.safety_breakdown_label = tk.Label(breakdown_frame, text="",
                                              bg='#f8f9fa', font=('Arial', 10),
                                              justify='left')
        self.safety_breakdown_label.pack()
        
        # Right column - Visualizations
        right_column = tk.Frame(content_frame, bg='white')
        right_column.pack(side='right', fill='both', expand=True)
        
        # Charts card
        charts_card = tk.Frame(right_column, bg='#f8f9fa', relief='groove', borderwidth=2)
        charts_card.pack(fill='both', expand=True)
        
        tk.Label(charts_card, text="📊 Visual Analytics", 
                font=('Arial', 14, 'bold'), bg='#f8f9fa').pack(pady=15, padx=20, anchor='w')
        
        # Chart container
        chart_container = tk.Frame(charts_card, bg='#f8f9fa')
        chart_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Bar chart frame
        bar_chart_frame = tk.Frame(chart_container, bg='white', relief='solid', borderwidth=1)
        bar_chart_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        tk.Label(bar_chart_frame, text="📈 Detection Count by Class", 
                font=('Arial', 12, 'bold'), bg='white').pack(pady=10)
        
        self.bar_chart_canvas = tk.Canvas(bar_chart_frame, bg='white', height=200)
        self.bar_chart_canvas.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Default message
        self.bar_chart_canvas.create_text(250, 100, 
                                         text="Detection chart will appear here", 
                                         fill='#95a5a6', font=('Arial', 12))
        
        # Pie chart frame
        pie_chart_frame = tk.Frame(chart_container, bg='white', relief='solid', borderwidth=1)
        pie_chart_frame.pack(fill='both', expand=True)
        
        tk.Label(pie_chart_frame, text="🥧 Safety Distribution", 
                font=('Arial', 12, 'bold'), bg='white').pack(pady=10)
        
        self.pie_chart_canvas = tk.Canvas(pie_chart_frame, bg='white', height=200)
        self.pie_chart_canvas.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Default message
        self.pie_chart_canvas.create_text(250, 100, 
                                         text="Safety distribution will appear here", 
                                         fill='#95a5a6', font=('Arial', 12))
        
        # Detailed logs card (below charts)
        logs_card = tk.Frame(right_column, bg='#f8f9fa', relief='groove', borderwidth=2)
        logs_card.pack(fill='both', expand=True, pady=15)
        
        tk.Label(logs_card, text="📝 Detailed Analysis", 
                font=('Arial', 14, 'bold'), bg='#f8f9fa').pack(pady=15, padx=20, anchor='w')
        
        # Create scrolled text widget for logs
        self.results_text = scrolledtext.ScrolledText(logs_card, height=8,
                                                     bg='#2c3e50', fg='#ecf0f1',
                                                     font=('Arial', 10))
        self.results_text.pack(fill='both', expand=True, padx=20, pady=(0, 20))

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
            self.validation_label.config(text="⚠ Please select a dataset folder", fg='#e67e22')
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
                    
                    # Initialize PPE requirements
                    self.initialize_ppe_requirements()
                else:
                    validation_msg = "⚠ data.yaml missing 'names' key"
                    all_valid = False
                
                self.validation_label.config(text=validation_msg, fg='#27ae60')
                
            except Exception as e:
                self.validation_label.config(text=f"✗ Error reading data.yaml: {str(e)}", fg='#e74c3c')
                all_valid = False
        else:
            missing_text = ", ".join(missing_items)
            self.validation_label.config(text=f"✗ Missing: {missing_text}", fg='#e74c3c')
        
        return all_valid
    
    def initialize_ppe_requirements(self):
        """Initialize PPE requirements for each class based on typical safety standards"""
        self.ppe_requirements = {}
        
        # Define critical PPE items and their requirements
        critical_ppe = {
            'helmet': {'min_instances': 1, 'critical': True, 'weight': 2.0},
            'hardhat': {'min_instances': 1, 'critical': True, 'weight': 2.0},
            'vest': {'min_instances': 1, 'critical': True, 'weight': 1.8},
            'high-vis': {'min_instances': 1, 'critical': True, 'weight': 1.8},
            'glove': {'min_instances': 2, 'critical': True, 'weight': 1.5},
            'hand': {'min_instances': 2, 'critical': True, 'weight': 1.5},
            'boot': {'min_instances': 2, 'critical': True, 'weight': 1.5},
            'shoe': {'min_instances': 2, 'critical': True, 'weight': 1.5},
            'goggle': {'min_instances': 1, 'critical': True, 'weight': 1.7},
            'glass': {'min_instances': 1, 'critical': True, 'weight': 1.7},
            'mask': {'min_instances': 1, 'critical': True, 'weight': 1.9},
            'respirator': {'min_instances': 1, 'critical': True, 'weight': 1.9},
            'ear': {'min_instances': 2, 'critical': False, 'weight': 1.3},
            'hearing': {'min_instances': 2, 'critical': False, 'weight': 1.3}
        }
        
        for i, class_name in enumerate(self.class_names):
            class_name_lower = class_name.lower()
            matched = False
            
            # Check for critical PPE items
            for key, requirements in critical_ppe.items():
                if key in class_name_lower:
                    self.ppe_requirements[i] = requirements.copy()
                    matched = True
                    break
            
            # Default for non-critical items
            if not matched:
                self.ppe_requirements[i] = {'min_instances': 1, 'critical': False, 'weight': 1.0}
    
    def update_status(self, status, color='#95a5a6'):
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
            self.update_status("Training...", color='#f39c12')
            self.log_message("=" * 60)
            self.log_message("Starting YOLOv8 training...")
            self.log_message(f"Parameters: Epochs={epochs}, Image Size={img_size}")
            self.log_message("=" * 60)
            
            self.log_message("Initializing YOLOv8 model...")
            model = YOLO("yolov8n.yaml")
            
            resume = False
            if os.path.exists(self.trained_model_path):
                resume = True
                self.log_message("Found existing model, resuming training...")
                model = YOLO(self.trained_model_path)
            
            yaml_path = os.path.join(self.dataset_path, 'data.yaml')
            
            self.log_message("Training started. This may take several minutes...")
            self.log_message("-" * 60)
            
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
            
            self.log_message("=" * 60)
            self.log_message("✓ Training completed successfully!")
            self.log_message(f"✓ Model saved as: {self.trained_model_path}")
            self.log_message("=" * 60)
            self.update_status("Completed", color='#27ae60')
            
            if os.path.exists(self.trained_model_path):
                self.log_message("Model is ready for detection.")
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.log_message("✗ " + error_msg)
            self.update_status("Error", color='#e74c3c')
            messagebox.showerror("Training Error", error_msg)
    
    def stop_training(self):
        """Stop the training process"""
        if self.is_training:
            self.update_status("Stopping...", color='#e67e22')
            self.log_message("Training stopped by user")
            self.update_status("Idle", color='#95a5a6')
    
    # ============================================================================
    # MODEL LOADING FUNCTIONS
    # ============================================================================
    
    def load_default_model(self):
        """Load the trained model from default location"""
        try:
            if not os.path.exists(self.trained_model_path):
                messagebox.showerror("Error", "No trained model found. Please train a model first.")
                return
            
            self.model = YOLO(self.trained_model_path)
            self.current_model_path = self.trained_model_path
            
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
            else:
                yaml_path = os.path.join(self.dataset_path, 'data.yaml')
                if os.path.exists(yaml_path):
                    with open(yaml_path, 'r') as f:
                        data_yaml = yaml.safe_load(f)
                    self.class_names = data_yaml.get('names', [])
            
            self.initialize_ppe_requirements()
            
            self.model_status_label.config(
                text=f"✓ Model loaded: {len(self.class_names)} classes", 
                fg='#27ae60'
            )
            self.model_details_label.config(text=f"Path: {self.trained_model_path}")
            self.detect_btn.config(state='normal')
            
            self.log_message(f"Default model loaded: {len(self.class_names)} classes")
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.model_status_label.config(text=f"✗ {error_msg}", fg='#e74c3c')
            messagebox.showerror("Error", error_msg)
    
    def select_custom_model(self):
        """Allow user to select a custom model file"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if file_path:
            self.load_custom_model(file_path)
    
    def load_custom_model(self, model_path):
        """Load a custom model from specified path"""
        try:
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return
            
            self.model = YOLO(model_path)
            self.current_model_path = model_path
            
            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
            else:
                self.class_names = [f"Class{i}" for i in range(100)]
            
            self.initialize_ppe_requirements()
            
            # Truncate path for display
            display_path = model_path
            if len(display_path) > 40:
                display_path = "..." + display_path[-40:]
            
            self.model_status_label.config(
                text=f"✓ Custom model loaded: {len(self.class_names)} classes", 
                fg='#27ae60'
            )
            self.model_details_label.config(text=f"Path: {display_path}")
            self.detect_btn.config(state='normal')
            
            self.log_message(f"Custom model loaded: {model_path}")
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.model_status_label.config(text=f"✗ {error_msg}", fg='#e74c3c')
            messagebox.showerror("Error", error_msg)
    
    # ============================================================================
    # DETECTION FUNCTIONS - FIXED IMAGE DISPLAY
    # ============================================================================
    
    def upload_single_image(self):
        """Upload a single image for detection"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif")]
        )
        
        if file_path:
            self.image_paths = [file_path]
            self.current_image_index = 0
            self.display_original_image(file_path)
            self.image_info_label.config(text=f"📁 Loaded: {os.path.basename(file_path)}")
            self.update_image_navigation()
    
    def upload_image_folder(self):
        """Upload a folder of images for detection"""
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        
        if folder_path:
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
            self.image_paths = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in valid_extensions
            ]
            
            if self.image_paths:
                self.current_image_index = 0
                self.display_original_image(self.image_paths[0])
                self.image_info_label.config(
                    text=f"📂 Loaded {len(self.image_paths)} images"
                )
                self.update_image_navigation()
            else:
                messagebox.showwarning("Warning", "No valid images found in folder")
    
    def display_original_image(self, image_path):
        """Display the original image in the preview panel (640x640)"""
        try:
            # Clear canvas
            self.original_canvas.delete("all")
            
            # Load image
            img = Image.open(image_path)
            self.original_image_pil = img.copy()
            
            # Calculate dimensions to fit 640x640 while maintaining aspect ratio
            img_width, img_height = img.size
            canvas_width, canvas_height = 630, 630
            
            # Calculate scaling factor
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            # Calculate new dimensions
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with black background
            new_img = Image.new('RGB', (canvas_width, canvas_height), color='#34495e')
            
            # Calculate position to center the image
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            
            # Paste the resized image onto the center of the black background
            new_img.paste(img_resized, (x_offset, y_offset))
            
            # Convert to PhotoImage
            self.current_original_photo = ImageTk.PhotoImage(new_img)
            
            # Display image on canvas
            self.original_canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.current_original_photo,
                anchor='center'
            )
            
            # Store reference
            self.original_canvas.image = self.current_original_photo
            
            # Enable detect button if model is loaded
            if self.model:
                self.detect_btn.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load image: {str(e)}")
            self.show_placeholder_images()
    
    def display_detected_image(self, image_path, results):
        """Display the detected image with bounding boxes (640x640)"""
        try:
            # Clear canvas
            self.detected_canvas.delete("all")
            
            # Load image with OpenCV
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            # Draw bounding boxes if results exist
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                boxes = results[0].boxes
                
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    # Get class name
                    if cls < len(self.class_names):
                        class_name = self.class_names[cls]
                    else:
                        class_name = f"Class_{cls}"
                    
                    # Get color for this class
                    color = self.get_class_color(cls)
                    
                    # Draw rectangle
                    cv2.rectangle(img_rgb, 
                                (int(x1), int(y1)), 
                                (int(x2), int(y2)), 
                                color, 
                                3)
                    
                    # Create label
                    label = f"{class_name} {conf:.2f}"
                    
                    # Get text size
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, font_scale, thickness
                    )
                    
                    # Draw label background
                    cv2.rectangle(img_rgb,
                                (int(x1), int(y1) - text_height - 10),
                                (int(x1) + text_width, int(y1)),
                                color,
                                -1)
                    
                    # Draw label text
                    cv2.putText(img_rgb,
                              label,
                              (int(x1), int(y1) - 5),
                              font,
                              font_scale,
                              (255, 255, 255),
                              thickness)
            
            # Convert to PIL Image
            img_pil = Image.fromarray(img_rgb)
            self.detected_image_pil = img_pil.copy()
            
            # Calculate dimensions to fit 640x640 while maintaining aspect ratio
            img_width, img_height = img_pil.size
            canvas_width, canvas_height = 630, 630
            
            # Calculate scaling factor
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            # Calculate new dimensions
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image
            img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with black background
            new_img = Image.new('RGB', (canvas_width, canvas_height), color='#34495e')
            
            # Calculate position to center the image
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            
            # Paste the resized image onto the center of the black background
            new_img.paste(img_resized, (x_offset, y_offset))
            
            # Convert to PhotoImage
            self.current_detected_photo = ImageTk.PhotoImage(new_img)
            
            # Display image on canvas
            self.detected_canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.current_detected_photo,
                anchor='center'
            )
            
            # Store reference
            self.detected_canvas.image = self.current_detected_photo
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot display detected image: {str(e)}")
            self.show_detected_placeholder()

    def show_detected_placeholder(self):
        """Show placeholder for detected image"""
        self.detected_canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width, canvas_height = 630, 630
        
        # Draw placeholder
        self.detected_canvas.create_text(
            canvas_width // 2,
            canvas_height // 2,
            text="Run detection to see results",
            fill='#95a5a6',
            font=('Arial', 14),
            justify='center'
        )

    def get_class_color(self, class_id):
        """Get a unique color for each class"""
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 165, 0),    # Orange
            (128, 0, 128),    # Purple
            (0, 128, 128),    # Teal
            (128, 128, 0)     # Olive
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
            
            # Run detection on all images
            self.detection_results_list = []
            
            # Get current image path
            current_image_path = self.image_paths[self.current_image_index]
            
            # Run detection with confidence threshold of 0.25 (default)
            results = self.model(current_image_path, conf=0.25)
            self.detection_results_list.append(results)
            
            # Display detection results
            self.display_detected_image(current_image_path, results)
            self.process_detection_results(results, current_image_path)
            
            # Enable navigation for multiple images
            self.update_image_navigation()
            
            # Save output if requested
            if self.save_var.get():
                self.save_detection_output(results, current_image_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
    
    def show_previous_image(self):
        """Show previous image in the list"""
        if self.image_paths and self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_original_image(self.image_paths[self.current_image_index])
            self.update_image_navigation()
            
            # Show detection results if available
            if self.current_image_index < len(self.detection_results_list):
                results = self.detection_results_list[self.current_image_index]
                self.display_detected_image(self.image_paths[self.current_image_index], results)
                self.process_detection_results(results, self.image_paths[self.current_image_index])
            else:
                # Clear detection results if not available
                self.show_detected_placeholder()
                self.clear_results()
    
    def show_next_image(self):
        """Show next image in the list"""
        if self.image_paths and self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.display_original_image(self.image_paths[self.current_image_index])
            self.update_image_navigation()
            
            # Show detection results if available
            if self.current_image_index < len(self.detection_results_list):
                results = self.detection_results_list[self.current_image_index]
                self.display_detected_image(self.image_paths[self.current_image_index], results)
                self.process_detection_results(results, self.image_paths[self.current_image_index])
            else:
                # Clear detection results if not available
                self.show_detected_placeholder()
                self.clear_results()
    
    def clear_results(self):
        """Clear all results"""
        self.results_text.delete(1.0, tk.END)
        self.detected_listbox.delete(0, tk.END)
        self.missing_listbox.delete(0, tk.END)
        self.summary_stats_label.config(text="📊 No detection results yet")
        self.safety_percentage_label.config(text="0.0%")
        self.safety_status_label.config(text="NOT CALCULATED")
        self.safety_breakdown_label.config(text="")
        
        # Clear charts
        self.bar_chart_canvas.delete("all")
        self.pie_chart_canvas.delete("all")
        self.bar_chart_canvas.create_text(250, 100, 
                                         text="Detection chart will appear here", 
                                         fill='#95a5a6', font=('Arial', 12))
        self.pie_chart_canvas.create_text(250, 100, 
                                         text="Safety distribution will appear here", 
                                         fill='#95a5a6', font=('Arial', 12))
    
    def update_image_navigation(self):
        """Update navigation buttons and counter"""
        if len(self.image_paths) > 1:
            self.prev_btn.config(state='normal' if self.current_image_index > 0 else 'disabled')
            self.next_btn.config(state='normal' if self.current_image_index < len(self.image_paths) - 1 else 'disabled')
            self.image_counter_label.config(
                text=f"📄 Image {self.current_image_index + 1}/{len(self.image_paths)}"
            )
        else:
            self.prev_btn.config(state='disabled')
            self.next_btn.config(state='disabled')
            self.image_counter_label.config(text="")
    
    # ============================================================================
    # RESULTS PROCESSING FUNCTIONS
    # ============================================================================
    
    def process_detection_results(self, results, image_path):
        """Process and display enhanced detection results"""
        # Clear previous results
        self.results_text.delete(1.0, tk.END)
        self.detected_listbox.delete(0, tk.END)
        self.missing_listbox.delete(0, tk.END)
        
        # Initialize counters
        detected_classes = {}
        class_counts = {}
        all_classes = set(range(len(self.class_names))) if self.class_names else set()
        detected_class_ids = set()
        total_confidence = 0
        total_detections = 0
        
        # Analyze results
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
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
                total_confidence += conf
                total_detections += 1
            
            # Build results text
            results_text = "=" * 70 + "\n"
            results_text += "DETECTION RESULTS ANALYSIS\n"
            results_text += "=" * 70 + "\n\n"
            results_text += f"📁 Image: {os.path.basename(image_path)}\n"
            results_text += f"📅 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            results_text += f"🤖 Model: {os.path.basename(self.current_model_path)}\n"
            results_text += f"🎯 Confidence Threshold: 0.25 (default)\n"
            results_text += "-" * 70 + "\n\n"
            
            # Detected objects
            if detected_classes:
                results_text += "✅ DETECTED PPE ITEMS:\n"
                results_text += "-" * 40 + "\n"
                
                for class_name, confidences in detected_classes.items():
                    count = len(confidences)
                    avg_conf = sum(confidences) / count if count > 0 else 0
                    
                    results_text += f"• {class_name}: {count} instances (avg confidence: {avg_conf:.1%})\n"
                    
                    # Add to detected listbox
                    display_text = f"{class_name} ({count}×, {avg_conf:.0%} conf)"
                    self.detected_listbox.insert(tk.END, display_text)
            else:
                results_text += "❌ NO PPE ITEMS DETECTED\n"
            
            results_text += "\n" + "-" * 70 + "\n\n"
            
            # Missing PPE items - Only show actual PPE classes from the model
            if self.class_names:
                # Define which classes are actual PPE items (not including 'human' or other non-PPE)
                ppe_classes = []
                non_ppe_classes = ['person', 'human', 'people', 'man', 'woman']
                
                for i, class_name in enumerate(self.class_names):
                    class_lower = class_name.lower()
                    is_ppe = True
                    for non_ppe in non_ppe_classes:
                        if non_ppe in class_lower:
                            is_ppe = False
                            break
                    if is_ppe:
                        ppe_classes.append(i)
                
                missing_ppe_classes = set(ppe_classes) - detected_class_ids
                
                if missing_ppe_classes:
                    results_text += "⚠ MISSING PPE ITEMS:\n"
                    results_text += "-" * 40 + "\n"
                    
                    for cls in missing_ppe_classes:
                        if cls < len(self.class_names):
                            class_name = self.class_names[cls]
                            results_text += f"• {class_name}\n"
                            
                            # Add to missing listbox
                            self.missing_listbox.insert(tk.END, class_name)
                else:
                    results_text += "✅ ALL PPE ITEMS DETECTED\n"
            else:
                results_text += "⚠ Class names not available\n"
            
            results_text += "\n" + "-" * 70 + "\n\n"
            
            # Calculate average confidence
            avg_confidence = total_confidence / total_detections if total_detections > 0 else 0
            
            # Calculate safety compliance
            safety_percentage = self.calculate_safety_percentage(class_counts, all_classes)
            
            results_text += "📈 SAFETY COMPLIANCE ANALYSIS\n"
            results_text += "-" * 40 + "\n"
            results_text += f"Overall Safety Score: {safety_percentage:.1f}%\n"
            results_text += f"Safety Status: {self.get_safety_status(safety_percentage)}\n\n"
            
            results_text += f"📊 Statistics:\n"
            results_text += f"  • Detected Items: {len(detected_classes)}/{len(all_classes)}\n"
            results_text += f"  • Total Detections: {total_detections}\n"
            results_text += f"  • Average Confidence: {avg_confidence:.1%}\n"
            results_text += f"  • Missing PPE Items: {self.missing_listbox.size()}\n"
            
            # Store results for visualization
            self.detection_results = {
                'detected': detected_classes,
                'missing': missing_ppe_classes if 'missing_ppe_classes' in locals() else set(),
                'class_counts': class_counts,
                'safety_percentage': safety_percentage,
                'total_classes': len(all_classes),
                'detected_count': len(detected_classes),
                'missing_count': self.missing_listbox.size(),
                'total_detections': total_detections,
                'avg_confidence': avg_confidence
            }
            
        else:
            results_text = "❌ No detections found in the image.\n"
            results_text += "=" * 70 + "\n\n"
            results_text += "Possible reasons:\n"
            results_text += "1. Image quality is poor\n"
            results_text += "2. PPE items are not visible\n"
            results_text += "3. Model is not trained for these items\n"
            
            self.detection_results = {
                'detected': {},
                'missing': set(),
                'class_counts': {},
                'safety_percentage': 0.0,
                'total_classes': len(all_classes) if self.class_names else 0,
                'detected_count': 0,
                'missing_count': 0,
                'total_detections': 0,
                'avg_confidence': 0.0
            }
        
        # Display results
        self.results_text.insert(tk.END, results_text)
        
        # Update summary statistics
        self.update_summary_statistics()
        
        # Update safety display
        self.update_safety_display()
        
        # Update visualizations
        self.update_charts()
    
    def calculate_safety_percentage(self, class_counts, all_classes):
        """
        Calculate reasonable safety percentage based on:
        1. Critical PPE items have higher weight
        2. Partial credit for some items (like 1 glove instead of 2)
        3. Overall detection confidence
        """
        if not all_classes:
            return 0.0
        
        total_score = 0
        max_score = 0
        
        for cls in all_classes:
            if cls in self.ppe_requirements:
                req = self.ppe_requirements[cls]
                detected = class_counts.get(cls, 0)
                required = req['min_instances']
                weight = req['weight']
                
                # Calculate score for this class
                if detected >= required:
                    # Full points for meeting requirements
                    class_score = weight * 100
                elif detected > 0:
                    # Partial points (e.g., 1 glove instead of 2 gives 50% credit)
                    class_score = weight * 100 * (detected / required)
                else:
                    # No points for missing items
                    class_score = 0
                
                total_score += class_score
                max_score += weight * 100
        
        # Calculate overall percentage
        if max_score > 0:
            safety_percentage = (total_score / max_score) * 100
        else:
            safety_percentage = 0.0
        
        # Round to 1 decimal place
        return round(safety_percentage, 1)
    
    def get_safety_status(self, percentage):
        """Get safety status based on percentage"""
        if percentage >= 85:
            return "EXCELLENT ✅"
        elif percentage >= 70:
            return "GOOD ✓"
        elif percentage >= 50:
            return "MODERATE ⚠"
        elif percentage >= 30:
            return "POOR ⚠"
        else:
            return "CRITICAL ❌"
    
    def update_summary_statistics(self):
        """Update the summary statistics display"""
        if not self.detection_results:
            self.summary_stats_label.config(text="📊 No detection results yet")
            return
        
        detected = self.detection_results['detected_count']
        missing = self.detection_results['missing_count']
        total = self.detection_results['total_classes']
        
        stats_text = f"📊 Detected: {detected}/{total} | Missing: {missing}/{total}"
        self.summary_stats_label.config(text=stats_text)
    
    def update_safety_display(self):
        """Update the safety percentage and status display"""
        if not self.detection_results:
            self.safety_percentage_label.config(text="0.0%")
            self.safety_status_label.config(text="NOT CALCULATED")
            self.safety_breakdown_label.config(text="")
            return
        
        safety_percentage = self.detection_results['safety_percentage']
        safety_status = self.get_safety_status(safety_percentage)
        
        # Update percentage display
        self.safety_percentage_label.config(text=f"{safety_percentage:.1f}%")
        
        # Update status with color coding
        if "EXCELLENT" in safety_status:
            color = '#27ae60'  # Green
            bg_color = '#2c3e50'
        elif "GOOD" in safety_status:
            color = '#2ecc71'  # Light Green
            bg_color = '#2c3e50'
        elif "MODERATE" in safety_status:
            color = '#f39c12'  # Orange
            bg_color = '#2c3e50'
        elif "POOR" in safety_status:
            color = '#e67e22'  # Dark Orange
            bg_color = '#2c3e50'
        else:
            color = '#e74c3c'  # Red
            bg_color = '#2c3e50'
        
        self.safety_status_label.config(text=safety_status, fg=color)
        self.safety_percentage_label.config(fg=color)
        
        # Update breakdown
        detected = self.detection_results['detected_count']
        missing = self.detection_results['missing_count']
        total = self.detection_results['total_classes']
        avg_conf = self.detection_results.get('avg_confidence', 0)
        
        breakdown_text = f"Detected Items: {detected}/{total}\n"
        breakdown_text += f"Missing Items: {missing}/{total}\n"
        breakdown_text += f"Avg Confidence: {avg_conf:.1%}\n"
        breakdown_text += f"Safety Score: {safety_percentage:.1f}%"
        
        self.safety_breakdown_label.config(text=breakdown_text)
    
    def update_charts(self):
        """Update the bar chart and pie chart visualizations"""
        if not self.detection_results:
            return
        
        # Clear previous charts
        for widget in self.bar_chart_canvas.winfo_children():
            widget.destroy()
        for widget in self.pie_chart_canvas.winfo_children():
            widget.destroy()
        
        # Get data for visualization
        detected_classes = self.detection_results.get('detected', {})
        safety_percentage = self.detection_results['safety_percentage']
        
        # ================== BAR CHART: Detection Count by Class ==================
        fig1, ax1 = plt.subplots(figsize=(6, 3.5))
        fig1.patch.set_facecolor('#f8f9fa')
        ax1.set_facecolor('#ffffff')
        
        if detected_classes:
            class_names = list(detected_classes.keys())
            counts = [len(detected_classes[name]) for name in class_names]
            
            # Sort by count
            sorted_indices = np.argsort(counts)[::-1]  # Descending
            class_names = [class_names[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
            
            # Limit to top 8 for readability
            max_items = min(8, len(class_names))
            class_names = class_names[:max_items]
            counts = counts[:max_items]
            
            colors = plt.cm.Set3(np.arange(len(class_names)) / len(class_names))
            bars = ax1.bar(range(len(class_names)), counts, color=colors, alpha=0.8)
            
            ax1.set_xlabel('PPE Classes', fontweight='bold')
            ax1.set_ylabel('Detection Count', fontweight='bold')
            ax1.set_title('Top Detected Items', fontsize=12, fontweight='bold', pad=15)
            ax1.set_xticks(range(len(class_names)))
            ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No PPE Items Detected', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=12, fontweight='bold',
                    color='#7f8c8d')
            ax1.set_title('Top Detected Items', fontsize=12, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        # Embed bar chart in tkinter
        canvas1 = FigureCanvasTkAgg(fig1, master=self.bar_chart_canvas)
        canvas1.draw()
        chart1_widget = canvas1.get_tk_widget()
        chart1_widget.pack(fill='both', expand=True)
        
        # ================== PIE CHART: Safety Distribution ==================
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        fig2.patch.set_facecolor('#f8f9fa')
        ax2.set_facecolor('#ffffff')
        
        safe_percentage = safety_percentage
        unsafe_percentage = max(0, 100 - safe_percentage)
        
        sizes = [safe_percentage, unsafe_percentage]
        labels = [f'Safe\n{safe_percentage:.1f}%', f'Unsafe\n{unsafe_percentage:.1f}%']
        colors = ['#27ae60', '#e74c3c']
        
        # Create pie chart with shadow and explode
        explode = (0.05, 0)
        wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, 
                                          colors=colors, autopct='%1.1f%%',
                                          shadow=True, startangle=90,
                                          textprops={'fontweight': 'bold'})
        
        # Style the autopct text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax2.set_title('Safety Distribution', fontsize=12, fontweight='bold', pad=15)
        
        # Add a circle at the center for a donut chart effect
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax2.add_artist(centre_circle)
        
        # Add safety status in center
        safety_status = self.get_safety_status(safety_percentage).replace("✅", "").replace("❌", "").replace("⚠", "").strip()
        ax2.text(0, 0, safety_status, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='#2c3e50')
        
        plt.tight_layout()
        
        # Embed pie chart in tkinter
        canvas2 = FigureCanvasTkAgg(fig2, master=self.pie_chart_canvas)
        canvas2.draw()
        chart2_widget = canvas2.get_tk_widget()
        chart2_widget.pack(fill='both', expand=True)
    
    def save_detection_output(self, results, image_path):
        """Save the detection output to a file"""
        try:
            # Create output directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"detection_output_{timestamp}"
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
            self.log_message(f"✓ Saved detection output to: {output_dir}")
            
        except Exception as e:
            self.log_message(f"✗ Error saving output: {str(e)}")


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