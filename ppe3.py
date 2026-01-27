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
        self.root.title("PPE Detection System - Safety Sentinel v3.0")
        self.root.geometry("1400x900")
        
        # Set application icon and theme
        self.root.configure(bg='#1a1a2e')
        
        # Application information
        self.app_info = {
            "name": "Safety Sentinel",
            "version": "3.0",
            "author": "AI Lab Project",
            "date": "2026-01-14"
        }
        
        # Create main frame with scrollbars
        self.main_frame = tk.Frame(root, bg='#1a1a2e')
        self.main_frame.pack(fill='both', expand=True)
        
        # Create canvas for scrolling
        self.canvas = tk.Canvas(self.main_frame, bg='#1a1a2e', highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(self.main_frame, orient="horizontal", command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        
        self.v_scrollbar.pack(side="right", fill="y")
        self.h_scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Create scrollable frame
        self.scrollable_frame = tk.Frame(self.canvas, bg='#1a1a2e')
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
        
        # Create header section
        self.create_header()
        
        # Create the main notebook (tab system)
        self.notebook = ttk.Notebook(self.scrollable_frame)
        self.notebook.pack(fill='both', expand=True, padx=20, pady=(10, 20))
        
        # Style configuration
        self.setup_styles()
        
        # Create the three main sections
        self.create_training_section()
        self.create_detection_section()
        self.create_results_section()
        
        # Create footer section
        self.create_footer()
        
        # Set initial status
        self.update_status("Ready", color='#4ECDC4')
        
        # Schedule canvas initialization after GUI is fully loaded
        self.root.after(100, self.initialize_canvas_placeholders)
    
    def create_header(self):
        """Create application header with logo and title"""
        header_frame = tk.Frame(self.scrollable_frame, bg='#0f3460', height=100)
        header_frame.pack(fill='x', padx=20, pady=(20, 10))
        header_frame.pack_propagate(False)
        
        # Logo and title
        title_frame = tk.Frame(header_frame, bg='#0f3460')
        title_frame.pack(fill='both', expand=True, padx=30)
        
        # App logo/icon
        tk.Label(title_frame, text="🛡️", font=('Arial', 36), 
                bg='#0f3460', fg='#4ECDC4').pack(side='left', padx=(0, 20))
        
        # App title and info
        tk.Label(title_frame, text="SAFETY SENTINEL", 
                font=('Arial', 28, 'bold'), bg='#0f3460', fg='white').pack(side='left', padx=(0, 10))
        
        tk.Label(title_frame, text=f"v{self.app_info['version']} | PPE Detection System", 
                font=('Arial', 12), bg='#0f3460', fg='#a9b7c6').pack(side='left')
    
    def create_footer(self):
        """Create application footer"""
        footer_frame = tk.Frame(self.scrollable_frame, bg='#0f3460', height=60)
        footer_frame.pack(fill='x', padx=20, pady=(10, 20))
        footer_frame.pack_propagate(False)
        
        # Status bar
        status_bar = tk.Frame(footer_frame, bg='#16213e')
        status_bar.pack(fill='both', expand=True, padx=10, pady=10)
        
        # System status
        self.system_status_label = tk.Label(status_bar, text="⚡ System: Ready", 
                                          font=('Arial', 10), bg='#16213e', fg='#4ECDC4')
        self.system_status_label.pack(side='left', padx=20)
        
        # Model status
        self.footer_model_label = tk.Label(status_bar, text="🤖 Model: Not Loaded", 
                                         font=('Arial', 10), bg='#16213e', fg='#a9b7c6')
        self.footer_model_label.pack(side='left', padx=20)
        
        # Memory status (placeholder)
        self.memory_label = tk.Label(status_bar, text="💾 Memory: --", 
                                   font=('Arial', 10), bg='#16213e', fg='#a9b7c6')
        self.memory_label.pack(side='left', padx=20)
        
        # Copyright
        tk.Label(status_bar, text=f"© 2026 {self.app_info['author']} | Safety Sentinel v{self.app_info['version']}", 
                font=('Arial', 9), bg='#16213e', fg='#7a8ba6').pack(side='right', padx=20)
    
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
        
        # Configure notebook style
        style.configure('Custom.TNotebook', background='#16213e')
        style.configure('Custom.TNotebook.Tab', 
                       background='#1a1a2e', 
                       foreground='#a9b7c6',
                       padding=[20, 10],
                       font=('Arial', 11, 'bold'))
        style.map('Custom.TNotebook.Tab', 
                 background=[('selected', '#0f3460')],
                 foreground=[('selected', '#4ECDC4')])
        
        # Apply style to notebook
        self.notebook.configure(style='Custom.TNotebook')
    
    def create_training_section(self):
        """Create the training section with enhanced GUI"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🏋️ TRAINING")
        
        # Main container
        main_container = tk.Frame(training_frame, bg='#16213e')
        main_container.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Header
        header_frame = tk.Frame(main_container, bg='#0f3460', height=60)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="🧠 MODEL TRAINING WORKSPACE", 
                font=('Arial', 16, 'bold'), bg='#0f3460', fg='#4ECDC4').pack(pady=15)
        
        # Content area with two columns
        content_frame = tk.Frame(main_container, bg='#16213e')
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left panel - Configuration
        left_panel = tk.Frame(content_frame, bg='#16213e', relief='flat')
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # Dataset configuration card
        dataset_card = self.create_card(left_panel, "📂 DATASET CONFIGURATION")
        
        # Dataset selection
        tk.Label(dataset_card, text="Dataset Path:", bg='#1a1a2e', 
                font=('Arial', 11), fg='#a9b7c6').pack(anchor='w', padx=20, pady=(15, 5))
        
        dataset_row = tk.Frame(dataset_card, bg='#1a1a2e')
        dataset_row.pack(fill='x', padx=20, pady=(0, 10))
        
        self.dataset_entry = tk.Entry(dataset_row, width=40, font=('Arial', 10),
                                     bg='#0f3460', fg='white', insertbackground='white',
                                     relief='solid', borderwidth=1, selectbackground='#4ECDC4')
        self.dataset_entry.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        browse_btn = tk.Button(dataset_row, text="Browse", 
                              command=self.browse_dataset, 
                              bg='#4ECDC4', fg='#1a1a2e', font=('Arial', 10, 'bold'),
                              relief='raised', borderwidth=0, padx=20, pady=5,
                              cursor='hand2')
        browse_btn.pack(side='right')
        
        # Dataset structure info
        info_frame = tk.Frame(dataset_card, bg='#0f3460', relief='solid', borderwidth=1)
        info_frame.pack(fill='x', padx=20, pady=10)
        
        info_text = """Required YOLOv8 Dataset Structure:
        
📁 dataset/
├── 📄 data.yaml
├── 📁 train/
│   ├── 📁 images/
│   └── 📁 labels/
└── 📁 valid/
    ├── 📁 images/
    └── 📁 labels/"""
        
        tk.Label(info_frame, text=info_text, justify='left', 
                bg='#0f3460', fg='#a9b7c6', font=('Courier', 9), padx=10, pady=10).pack()
        
        # Validation button
        validate_btn = tk.Button(dataset_card, text="🔍 Validate Dataset Structure", 
                                command=self.validate_dataset, 
                                bg='#45B7D1', fg='white', font=('Arial', 11, 'bold'),
                                relief='raised', borderwidth=0, padx=25, pady=8,
                                cursor='hand2')
        validate_btn.pack(pady=15)
        
        self.validation_label = tk.Label(dataset_card, text="", bg='#1a1a2e',
                                        font=('Arial', 10))
        self.validation_label.pack(pady=(0, 15))
        
        # Training parameters card
        params_card = self.create_card(left_panel, "⚙️ TRAINING PARAMETERS")
        
        # Epochs input
        param_frame = tk.Frame(params_card, bg='#1a1a2e')
        param_frame.pack(fill='x', padx=20, pady=15)
        
        tk.Label(param_frame, text="Number of Epochs:", bg='#1a1a2e', 
                font=('Arial', 11), fg='#a9b7c6').grid(row=0, column=0, sticky='w', pady=5)
        
        self.epoch_entry = tk.Entry(param_frame, width=15, font=('Arial', 10),
                                   bg='#0f3460', fg='white', insertbackground='white')
        self.epoch_entry.insert(0, "100")
        self.epoch_entry.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        
        # Image size selector
        tk.Label(param_frame, text="Input Image Size:", bg='#1a1a2e', 
                font=('Arial', 11), fg='#a9b7c6').grid(row=1, column=0, sticky='w', pady=5)
        
        self.size_var = tk.StringVar(value="640")
        size_options = ["320", "416", "512", "640", "768"]
        size_menu = ttk.Combobox(param_frame, textvariable=self.size_var, 
                                values=size_options, state='readonly',
                                width=15, font=('Arial', 10))
        size_menu.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        
        # Right panel - Training controls
        right_panel = tk.Frame(content_frame, bg='#16213e', relief='flat')
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Status card
        status_card = self.create_card(right_panel, "📊 TRAINING STATUS")
        
        # Status indicator with color coding
        self.status_indicator = tk.Label(status_card, text="●", 
                                        font=('Arial', 48), bg='#1a1a2e')
        self.status_indicator.pack(pady=10)
        
        self.status_label = tk.Label(status_card, text="Status: Ready", 
                                    font=('Arial', 14, 'bold'), bg='#1a1a2e', fg='#4ECDC4')
        self.status_label.pack(pady=5)
        
        # Progress frame
        progress_frame = tk.Frame(status_card, bg='#1a1a2e')
        progress_frame.pack(fill='x', padx=20, pady=20)
        
        self.progress_label = tk.Label(progress_frame, text="Progress: --", 
                                      bg='#1a1a2e', fg='#a9b7c6', font=('Arial', 10))
        self.progress_label.pack(anchor='w')
        
        # Training buttons
        button_frame = tk.Frame(status_card, bg='#1a1a2e')
        button_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        self.train_btn = tk.Button(button_frame, text="▶ START TRAINING", 
                                  command=self.start_training, 
                                  bg='#4ECDC4', fg='#1a1a2e', font=('Arial', 12, 'bold'),
                                  relief='raised', borderwidth=0, padx=30, pady=12,
                                  cursor='hand2')
        self.train_btn.pack(side='left', padx=(0, 10))
        
        self.stop_btn = tk.Button(button_frame, text="⏹ STOP TRAINING", 
                                 command=self.stop_training, 
                                 bg='#FF6B6B', fg='white', font=('Arial', 12, 'bold'),
                                 relief='raised', borderwidth=0, padx=30, pady=12,
                                 cursor='hand2', state='disabled')
        self.stop_btn.pack(side='left')
        
        # Training logs card
        log_card = self.create_card(right_panel, "📝 TRAINING LOGS")
        
        # Create scrolled text widget for logs
        self.log_text = scrolledtext.ScrolledText(log_card, height=15, 
                                                 bg='#0f3460', fg='#a9b7c6',
                                                 font=('Consolas', 9), 
                                                 insertbackground='white',
                                                 wrap='word')
        self.log_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Add initial welcome message
        self.log_message("=" * 60)
        self.log_message("SAFETY SENTINEL - PPE Detection System")
        self.log_message(f"Version: {self.app_info['version']}")
        self.log_message("=" * 60)
        self.log_message("Ready to train custom PPE detection model...")
        self.log_message("")
    
    def create_detection_section(self):
        """Create the detection section with enhanced GUI"""
        detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(detection_frame, text="🔍 DETECTION")
        
        # Main container
        main_container = tk.Frame(detection_frame, bg='#16213e')
        main_container.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Header
        header_frame = tk.Frame(main_container, bg='#0f3460', height=60)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="👁️ REAL-TIME PPE DETECTION", 
                font=('Arial', 16, 'bold'), bg='#0f3460', fg='#96CEB4').pack(pady=15)
        
        # Content area
        content_frame = tk.Frame(main_container, bg='#16213e')
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left panel - Controls
        control_panel = tk.Frame(content_frame, bg='#16213e', width=300)
        control_panel.pack(side='left', fill='y', padx=(0, 20))
        control_panel.pack_propagate(False)
        
        # Model loading card
        model_card = self.create_card(control_panel, "🤖 MODEL SELECTION")
        
        # Model info display
        model_info_frame = tk.Frame(model_card, bg='#0f3460', relief='solid', borderwidth=1)
        model_info_frame.pack(fill='x', padx=15, pady=10)
        
        self.model_status_label = tk.Label(model_info_frame, text="⏳ No Model Loaded", 
                                          bg='#0f3460', fg='#FFD166', 
                                          font=('Arial', 12, 'bold'))
        self.model_status_label.pack(pady=8, padx=10)
        
        self.model_details_label = tk.Label(model_info_frame, text="Load a model to start detection", 
                                           bg='#0f3460', fg='#a9b7c6',
                                           font=('Arial', 9), wraplength=250)
        self.model_details_label.pack(pady=(0, 8), padx=10)
        
        # Model loading buttons
        load_frame = tk.Frame(model_card, bg='#1a1a2e')
        load_frame.pack(fill='x', padx=15, pady=10)
        
        # Button 1: Load default trained model
        default_btn = tk.Button(load_frame, text="📁 Load Trained Model", 
                               command=self.load_default_model, 
                               bg='#45B7D1', fg='white', font=('Arial', 11),
                               relief='raised', borderwidth=0, padx=20, pady=10,
                               width=20, cursor='hand2')
        default_btn.pack(pady=5)
        
        # Button 2: Select custom model
        custom_btn = tk.Button(load_frame, text="📂 Select Custom Model", 
                              command=self.select_custom_model, 
                              bg='#45B7D1', fg='white', font=('Arial', 11),
                              relief='raised', borderwidth=0, padx=20, pady=10,
                              width=20, cursor='hand2')
        custom_btn.pack(pady=5)
        
        # Image upload card
        upload_card = self.create_card(control_panel, "📤 IMAGE UPLOAD")
        
        upload_frame = tk.Frame(upload_card, bg='#1a1a2e')
        upload_frame.pack(fill='x', padx=15, pady=15)
        
        # Single image upload
        single_btn = tk.Button(upload_frame, text="🖼️ Upload Single Image", 
                              command=self.upload_single_image, 
                              bg='#96CEB4', fg='#1a1a2e', font=('Arial', 11, 'bold'),
                              relief='raised', borderwidth=0, padx=20, pady=12,
                              width=18, cursor='hand2')
        single_btn.pack(pady=8)
        
        # Folder upload
        folder_btn = tk.Button(upload_frame, text="📁 Upload Image Folder", 
                              command=self.upload_image_folder, 
                              bg='#96CEB4', fg='#1a1a2e', font=('Arial', 11, 'bold'),
                              relief='raised', borderwidth=0, padx=20, pady=12,
                              width=18, cursor='hand2')
        folder_btn.pack(pady=8)
        
        # Image info display
        self.image_info_label = tk.Label(upload_card, text="No image loaded", 
                                        bg='#1a1a2e', font=('Arial', 10), 
                                        fg='#a9b7c6')
        self.image_info_label.pack(pady=(0, 15))
        
        # Detection controls card
        detect_card = self.create_card(control_panel, "🎯 DETECTION CONTROLS")
        
        detect_frame = tk.Frame(detect_card, bg='#1a1a2e')
        detect_frame.pack(fill='x', padx=15, pady=15)
        
        # Detection button
        self.detect_btn = tk.Button(detect_frame, text="🚀 RUN DETECTION", 
                                   command=self.run_detection, 
                                   bg='#FF6B6B', fg='white', font=('Arial', 13, 'bold'),
                                   relief='raised', borderwidth=0, padx=30, pady=15,
                                   cursor='hand2', state='disabled')
        self.detect_btn.pack(pady=5)
        
        # Save option
        save_frame = tk.Frame(detect_frame, bg='#1a1a2e')
        save_frame.pack(pady=10)
        
        self.save_var = tk.BooleanVar(value=True)
        save_check = tk.Checkbutton(save_frame, text="💾 Save detection results", 
                                   variable=self.save_var, bg='#1a1a2e',
                                   fg='#a9b7c6', font=('Arial', 10), 
                                   selectcolor='#4ECDC4', activebackground='#1a1a2e')
        save_check.pack()
        
        # Right panel - Image preview
        preview_panel = tk.Frame(content_frame, bg='#16213e')
        preview_panel.pack(side='right', fill='both', expand=True)
        
        # Image preview card
        preview_card = self.create_card(preview_panel, "👁️‍🗨️ IMAGE PREVIEW")
        
        # Image container with two columns
        image_container = tk.Frame(preview_card, bg='#1a1a2e')
        image_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Original image section
        orig_frame = tk.LabelFrame(image_container, text="ORIGINAL IMAGE", 
                                  font=('Arial', 12, 'bold'),
                                  bg='#0f3460', fg='#96CEB4', 
                                  relief='solid', borderwidth=1)
        orig_frame.pack(side='left', fill='both', expand=True, padx=(0, 10), pady=5)
        
        # Canvas for original image (512x512)
        self.original_canvas = tk.Canvas(orig_frame, bg='#0f3460', highlightthickness=0,
                                        width=512, height=512)
        self.original_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Detected image section
        det_frame = tk.LabelFrame(image_container, text="DETECTED IMAGE", 
                                 font=('Arial', 12, 'bold'),
                                 bg='#0f3460', fg='#FFD166', 
                                 relief='solid', borderwidth=1)
        det_frame.pack(side='right', fill='both', expand=True, pady=5)
        
        # Canvas for detected image (512x512)
        self.detected_canvas = tk.Canvas(det_frame, bg='#0f3460', highlightthickness=0,
                                        width=512, height=512)
        self.detected_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Navigation controls
        nav_card = self.create_card(preview_panel, "🔄 NAVIGATION CONTROLS")
        
        nav_frame = tk.Frame(nav_card, bg='#1a1a2e')
        nav_frame.pack(fill='x', padx=15, pady=15)
        
        # Navigation buttons
        control_frame = tk.Frame(nav_frame, bg='#1a1a2e')
        control_frame.pack(pady=5)
        
        self.prev_btn = tk.Button(control_frame, text="◀ Previous Image",
                                 command=self.show_previous_image,
                                 bg='#45B7D1', fg='white', font=('Arial', 11),
                                 relief='raised', borderwidth=0, padx=20, pady=8,
                                 cursor='hand2', state='disabled')
        self.prev_btn.pack(side='left', padx=5)
        
        self.next_btn = tk.Button(control_frame, text="Next Image ▶",
                                 command=self.show_next_image,
                                 bg='#45B7D1', fg='white', font=('Arial', 11),
                                 relief='raised', borderwidth=0, padx=20, pady=8,
                                 cursor='hand2', state='disabled')
        self.next_btn.pack(side='left', padx=5)
        
        # Image counter
        counter_frame = tk.Frame(nav_frame, bg='#1a1a2e')
        counter_frame.pack(pady=10)
        
        self.image_counter_label = tk.Label(counter_frame, text="📄 Image 0/0",
                                           bg='#1a1a2e', font=('Arial', 12, 'bold'),
                                           fg='#4ECDC4')
        self.image_counter_label.pack()
        
        # Current images
        self.current_original_image = None
        self.current_detected_image = None
        self.image_paths = []
        self.current_image_index = 0
        self.detection_results_list = []
    
    def create_results_section(self):
        """Create the enhanced results and analysis section"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📊 RESULTS")
        
        # Main container
        main_container = tk.Frame(results_frame, bg='#16213e')
        main_container.pack(fill='both', expand=True, padx=2, pady=2)
        
        # Header
        header_frame = tk.Frame(main_container, bg='#0f3460', height=60)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="📈 DETECTION RESULTS & ANALYSIS", 
                font=('Arial', 16, 'bold'), bg='#0f3460', fg='#FFD166').pack(pady=15)
        
        # Content area with two columns
        content_frame = tk.Frame(main_container, bg='#16213e')
        content_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Left column - Summary and Safety
        left_column = tk.Frame(content_frame, bg='#16213e')
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 15))
        
        # Results summary card
        summary_card = self.create_card(left_column, "📋 DETECTION SUMMARY")
        
        # Two-column summary
        summary_container = tk.Frame(summary_card, bg='#1a1a2e')
        summary_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Detected items
        detected_frame = tk.Frame(summary_container, bg='#0f3460', relief='solid', borderwidth=1)
        detected_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 5), pady=5)
        
        tk.Label(detected_frame, text="✅ DETECTED ITEMS", font=('Arial', 12, 'bold'), 
                bg='#1e5128', fg='white').pack(fill='x', pady=5)
        
        self.detected_listbox = tk.Listbox(detected_frame, height=10, bg='#0f3460',
                                          fg='#96CEB4', font=('Arial', 10), 
                                          selectmode=tk.SINGLE, relief='flat',
                                          selectbackground='#4ECDC4', selectforeground='#1a1a2e')
        detected_scrollbar = tk.Scrollbar(detected_frame)
        detected_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.detected_listbox.config(yscrollcommand=detected_scrollbar.set)
        detected_scrollbar.config(command=self.detected_listbox.yview)
        self.detected_listbox.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Missing items
        missing_frame = tk.Frame(summary_container, bg='#0f3460', relief='solid', borderwidth=1)
        missing_frame.grid(row=0, column=1, sticky='nsew', padx=(5, 0), pady=5)
        
        tk.Label(missing_frame, text="⚠ MISSING ITEMS", font=('Arial', 12, 'bold'), 
                bg='#852747', fg='white').pack(fill='x', pady=5)
        
        self.missing_listbox = tk.Listbox(missing_frame, height=10, bg='#0f3460',
                                         fg='#FF9A8B', font=('Arial', 10), 
                                         selectmode=tk.SINGLE, relief='flat',
                                         selectbackground='#FF6B6B', selectforeground='white')
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
                                           bg='#1a1a2e', font=('Arial', 11, 'bold'),
                                           fg='#4ECDC4')
        self.summary_stats_label.pack(pady=10)
        
        # Safety analysis card
        safety_card = self.create_card(left_column, "🛡️ SAFETY COMPLIANCE")
        
        # Safety percentage display
        safety_display = tk.Frame(safety_card, bg='#0f3460', relief='solid', borderwidth=1)
        safety_display.pack(fill='x', padx=20, pady=20)
        
        self.safety_percentage_label = tk.Label(safety_display, 
                                               text="0.0%",
                                               font=('Arial', 48, 'bold'),
                                               bg='#0f3460', fg='#4ECDC4')
        self.safety_percentage_label.pack(pady=10)
        
        self.safety_status_label = tk.Label(safety_display,
                                          text="NOT ANALYZED",
                                          font=('Arial', 14, 'bold'),
                                          bg='#0f3460', fg='#a9b7c6')
        self.safety_status_label.pack(pady=(0, 10))
        
        # Safety breakdown
        breakdown_frame = tk.Frame(safety_card, bg='#1a1a2e')
        breakdown_frame.pack(fill='x', padx=20, pady=(0, 20))
        
        self.safety_breakdown_label = tk.Label(breakdown_frame, text="",
                                              bg='#1a1a2e', font=('Arial', 10),
                                              fg='#a9b7c6', justify='left')
        self.safety_breakdown_label.pack()
        
        # Right column - Visualizations
        right_column = tk.Frame(content_frame, bg='#16213e')
        right_column.pack(side='right', fill='both', expand=True)
        
        # Charts card
        charts_card = self.create_card(right_column, "📊 VISUAL ANALYTICS")
        
        # Chart container
        chart_container = tk.Frame(charts_card, bg='#1a1a2e')
        chart_container.pack(fill='both', expand=True, padx=15, pady=15)
        
        # Bar chart frame
        bar_chart_frame = tk.LabelFrame(chart_container, text="📈 DETECTION COUNT BY CLASS", 
                                       font=('Arial', 11, 'bold'),
                                       bg='#0f3460', fg='#96CEB4', 
                                       relief='solid', borderwidth=1)
        bar_chart_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        self.bar_chart_canvas = tk.Canvas(bar_chart_frame, bg='#0f3460', height=180)
        self.bar_chart_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Default message
        self.bar_chart_canvas.create_text(250, 90, 
                                         text="Detection chart will appear here", 
                                         fill='#a9b7c6', font=('Arial', 12))
        
        # Pie chart frame
        pie_chart_frame = tk.LabelFrame(chart_container, text="🥧 SAFETY DISTRIBUTION", 
                                       font=('Arial', 11, 'bold'),
                                       bg='#0f3460', fg='#FFD166', 
                                       relief='solid', borderwidth=1)
        pie_chart_frame.pack(fill='both', expand=True)
        
        self.pie_chart_canvas = tk.Canvas(pie_chart_frame, bg='#0f3460', height=180)
        self.pie_chart_canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Default message
        self.pie_chart_canvas.create_text(250, 90, 
                                         text="Safety distribution will appear here", 
                                         fill='#a9b7c6', font=('Arial', 12))
        
        # Detailed analysis card
        analysis_card = self.create_card(right_column, "📝 DETAILED ANALYSIS")
        
        # Create scrolled text widget for logs
        self.results_text = scrolledtext.ScrolledText(analysis_card, height=10,
                                                     bg='#0f3460', fg='#a9b7c6',
                                                     font=('Consolas', 9),
                                                     insertbackground='white',
                                                     wrap='word')
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def create_card(self, parent, title):
        """Create a styled card container"""
        card = tk.Frame(parent, bg='#1a1a2e', relief='solid', borderwidth=1)
        card.pack(fill='both', expand=True, pady=(0, 15))
        
        # Card header
        header = tk.Frame(card, bg='#0f3460', height=40)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        tk.Label(header, text=title, font=('Arial', 12, 'bold'), 
                bg='#0f3460', fg='#4ECDC4').pack(pady=10)
        
        # Card content area
        content = tk.Frame(card, bg='#1a1a2e')
        content.pack(fill='both', expand=True, padx=2, pady=2)
        
        return content
    
    def initialize_canvas_placeholders(self):
        """Initialize canvas placeholders after GUI is fully loaded"""
        self.show_placeholder_images()
    
    def show_placeholder_images(self):
        """Show placeholder text when no images are loaded"""
        # Clear both canvases
        self.original_canvas.delete("all")
        self.detected_canvas.delete("all")
        
        # Get canvas dimensions
        canvas_width, canvas_height = 512, 512
        
        # Draw placeholder on original canvas
        self.original_canvas.create_text(
            canvas_width // 2, canvas_height // 2,
            text="Upload an image to start\n(Click 'Upload Single Image' or 'Upload Image Folder')",
            fill='#a9b7c6', font=('Arial', 12),
            justify='center'
        )
        
        # Draw placeholder on detected canvas
        self.detected_canvas.create_text(
            canvas_width // 2, canvas_height // 2,
            text="Detection results will appear here\n(Load a model and click 'RUN DETECTION')",
            fill='#a9b7c6', font=('Arial', 12),
            justify='center'
        )
    
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
            self.validation_label.config(text="✓ Dataset path selected", fg='#4ECDC4')
    
    def validate_dataset(self):
        """Validate the dataset structure and data.yaml file"""
        dataset_path = self.dataset_entry.get()
        
        if not dataset_path:
            self.validation_label.config(text="⚠ Please select a dataset folder", fg='#FFD166')
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
                    
                    # Log the validation
                    self.log_message(f"Dataset validated: {class_count} classes found")
                    for i, name in enumerate(self.class_names):
                        self.log_message(f"  Class {i}: {name}")
                    
                else:
                    validation_msg = "⚠ data.yaml missing 'names' key"
                    all_valid = False
                
                self.validation_label.config(text=validation_msg, fg='#4ECDC4')
                
            except Exception as e:
                self.validation_label.config(text=f"✗ Error reading data.yaml: {str(e)}", fg='#FF6B6B')
                all_valid = False
        else:
            missing_text = ", ".join(missing_items)
            self.validation_label.config(text=f"✗ Missing: {missing_text}", fg='#FF6B6B')
        
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
    
    def update_status(self, status, color='#4ECDC4'):
        """Update the training status indicator"""
        status_text = f"Status: {status}"
        self.status_label.config(text=status_text, fg=color)
        self.status_indicator.config(fg=color)
        self.system_status_label.config(text=f"⚡ System: {status}")
        
        if status == "Training...":
            self.train_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.is_training = True
        elif status in ["Completed", "Error", "Ready"]:
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
            self.update_status("Training...", color='#FFD166')
            self.log_message("=" * 60)
            self.log_message("STARTING YOLOv8 TRAINING...")
            self.log_message(f"Parameters: Epochs={epochs}, Image Size={img_size}")
            self.log_message(f"Classes: {len(self.class_names)}")
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
            self.log_message("✓ TRAINING COMPLETED SUCCESSFULLY!")
            self.log_message(f"✓ Model saved as: {self.trained_model_path}")
            self.log_message(f"✓ Classes trained: {len(self.class_names)}")
            self.log_message("=" * 60)
            self.update_status("Completed", color='#4ECDC4')
            
            if os.path.exists(self.trained_model_path):
                self.log_message("Model is ready for detection.")
            
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            self.log_message("✗ " + error_msg)
            self.update_status("Error", color='#FF6B6B')
            messagebox.showerror("Training Error", error_msg)
    
    def stop_training(self):
        """Stop the training process"""
        if self.is_training:
            self.update_status("Stopping...", color='#FFD166')
            self.log_message("Training stopped by user")
            self.update_status("Ready", color='#4ECDC4')
    
    # ============================================================================
    # MODEL LOADING FUNCTIONS
    # ============================================================================
    
    def load_default_model(self):
        """Load the trained model from default location"""
        try:
            if not os.path.exists(self.trained_model_path):
                messagebox.showerror("Error", 
                    "No trained model found. Please train a model first.\n\n"
                    "Go to the TRAINING tab to train your custom PPE detection model.")
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
                text=f"✓ Model Loaded", 
                fg='#4ECDC4'
            )
            self.model_details_label.config(
                text=f"Classes: {len(self.class_names)}\n"
                     f"Path: {os.path.basename(self.trained_model_path)}"
            )
            self.footer_model_label.config(text=f"🤖 Model: Loaded ({len(self.class_names)} classes)")
            self.detect_btn.config(state='normal' if self.image_paths else 'disabled')
            
            self.log_message(f"Default model loaded: {len(self.class_names)} classes")
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.model_status_label.config(text=f"✗ {error_msg}", fg='#FF6B6B')
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
                text=f"✓ Custom Model Loaded", 
                fg='#4ECDC4'
            )
            self.model_details_label.config(
                text=f"Classes: {len(self.class_names)}\n"
                     f"Path: {os.path.basename(model_path)}"
            )
            self.footer_model_label.config(text=f"🤖 Model: Custom ({len(self.class_names)} classes)")
            self.detect_btn.config(state='normal' if self.image_paths else 'disabled')
            
            self.log_message(f"Custom model loaded: {model_path}")
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            self.model_status_label.config(text=f"✗ {error_msg}", fg='#FF6B6B')
            messagebox.showerror("Error", error_msg)
    
    # ============================================================================
    # DETECTION FUNCTIONS
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
            
            # Enable detect button if model is loaded
            if self.model:
                self.detect_btn.config(state='normal')
    
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
                
                # Enable detect button if model is loaded
                if self.model:
                    self.detect_btn.config(state='normal')
            else:
                messagebox.showwarning("Warning", "No valid images found in folder")
    
    def display_original_image(self, image_path):
        """Display the original image in the preview panel (512x512)"""
        try:
            # Clear canvas
            self.original_canvas.delete("all")
            
            # Load image
            img = Image.open(image_path)
            self.original_image_pil = img.copy()
            
            # Calculate dimensions to fit 512x512 while maintaining aspect ratio
            img_width, img_height = img.size
            canvas_width, canvas_height = 512, 512
            
            # Calculate scaling factor
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            # Calculate new dimensions
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with dark background
            new_img = Image.new('RGB', (canvas_width, canvas_height), color='#0f3460')
            
            # Calculate position to center the image
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            
            # Paste the resized image onto the center of the dark background
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
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load image: {str(e)}")
            self.show_placeholder_images()
    
    def display_detected_image(self, image_path, results):
        """Display the detected image with bounding boxes (512x512)"""
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
                    font_scale = 0.7
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
            
            # Calculate dimensions to fit 512x512 while maintaining aspect ratio
            img_width, img_height = img_pil.size
            canvas_width, canvas_height = 512, 512
            
            # Calculate scaling factor
            scale = min(canvas_width / img_width, canvas_height / img_height)
            
            # Calculate new dimensions
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image
            img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with dark background
            new_img = Image.new('RGB', (canvas_width, canvas_height), color='#0f3460')
            
            # Calculate position to center the image
            x_offset = (canvas_width - new_width) // 2
            y_offset = (canvas_height - new_height) // 2
            
            # Paste the resized image onto the center of the dark background
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
        canvas_width, canvas_height = 512, 512
        
        # Draw placeholder
        self.detected_canvas.create_text(
            canvas_width // 2,
            canvas_height // 2,
            text="Run detection to see results",
            fill='#a9b7c6',
            font=('Arial', 14),
            justify='center'
        )

    def get_class_color(self, class_id):
        """Get a unique color for each class"""
        colors = [
            (255, 100, 100),    # Soft Red
            (100, 255, 100),    # Soft Green
            (100, 100, 255),    # Soft Blue
            (255, 255, 100),    # Soft Yellow
            (255, 100, 255),    # Soft Magenta
            (100, 255, 255),    # Soft Cyan
            (255, 180, 100),    # Soft Orange
            (180, 100, 255),    # Soft Purple
            (100, 255, 180),    # Soft Teal
            (255, 200, 100)     # Soft Gold
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
        self.safety_status_label.config(text="NOT ANALYZED", fg='#a9b7c6')
        self.safety_breakdown_label.config(text="")
        
        # Clear charts
        self.bar_chart_canvas.delete("all")
        self.pie_chart_canvas.delete("all")
        self.bar_chart_canvas.create_text(250, 90, 
                                         text="Detection chart will appear here", 
                                         fill='#a9b7c6', font=('Arial', 12))
        self.pie_chart_canvas.create_text(250, 90, 
                                         text="Safety distribution will appear here", 
                                         fill='#a9b7c6', font=('Arial', 12))
    
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
            results_text += f"🎯 Confidence Threshold: 0.25\n"
            results_text += "-" * 70 + "\n\n"
            
            # Detected objects
            if detected_classes:
                results_text += "✅ DETECTED ITEMS:\n"
                results_text += "-" * 40 + "\n"
                
                for class_name, confidences in detected_classes.items():
                    count = len(confidences)
                    avg_conf = sum(confidences) / count if count > 0 else 0
                    
                    results_text += f"• {class_name}: {count} instances (avg confidence: {avg_conf:.1%})\n"
                    
                    # Add to detected listbox
                    display_text = f"{class_name} ({count}×, {avg_conf:.0%} conf)"
                    self.detected_listbox.insert(tk.END, display_text)
            else:
                results_text += "❌ NO ITEMS DETECTED\n"
            
            results_text += "\n" + "-" * 70 + "\n\n"
            
            # Missing items - Only show actual PPE classes from the model
            if self.class_names:
                # Define which classes are actual PPE items
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
            
            results_text += f"📊 STATISTICS:\n"
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
            self.safety_status_label.config(text="NOT ANALYZED", fg='#a9b7c6')
            self.safety_breakdown_label.config(text="")
            return
        
        safety_percentage = self.detection_results['safety_percentage']
        safety_status = self.get_safety_status(safety_percentage)
        
        # Update percentage display
        self.safety_percentage_label.config(text=f"{safety_percentage:.1f}%")
        
        # Update status with color coding
        if "EXCELLENT" in safety_status:
            color = '#4ECDC4'  # Teal
            status_color = '#4ECDC4'
        elif "GOOD" in safety_status:
            color = '#96CEB4'  # Green
            status_color = '#96CEB4'
        elif "MODERATE" in safety_status:
            color = '#FFD166'  # Yellow
            status_color = '#FFD166'
        elif "POOR" in safety_status:
            color = '#FF9A8B'  # Orange
            status_color = '#FF9A8B'
        else:
            color = '#FF6B6B'  # Red
            status_color = '#FF6B6B'
        
        self.safety_status_label.config(text=safety_status, fg=status_color)
        self.safety_percentage_label.config(fg=color)
        
        # Update breakdown
        detected = self.detection_results['detected_count']
        missing = self.detection_results['missing_count']
        total = self.detection_results['total_classes']
        avg_conf = self.detection_results.get('avg_confidence', 0)
        
        breakdown_text = f"Detected Items: {detected}/{total}\n"
        breakdown_text += f"Missing Items: {missing}/{total}\n"
        breakdown_text += f"Average Confidence: {avg_conf:.1%}\n"
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
        fig1.patch.set_facecolor('#0f3460')
        ax1.set_facecolor('#0f3460')
        
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
            
            ax1.set_xlabel('PPE Classes', fontweight='bold', color='#a9b7c6')
            ax1.set_ylabel('Detection Count', fontweight='bold', color='#a9b7c6')
            ax1.set_title('Top Detected Items', fontsize=12, fontweight='bold', 
                         pad=15, color='#4ECDC4')
            ax1.set_xticks(range(len(class_names)))
            ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9, color='#a9b7c6')
            ax1.tick_params(colors='#a9b7c6')
            
            # Add count labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9, 
                        color='white', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No PPE Items Detected', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax1.transAxes, fontsize=12, fontweight='bold',
                    color='#a9b7c6')
            ax1.set_title('Top Detected Items', fontsize=12, fontweight='bold', 
                         pad=15, color='#4ECDC4')
        
        plt.tight_layout()
        
        # Embed bar chart in tkinter
        canvas1 = FigureCanvasTkAgg(fig1, master=self.bar_chart_canvas)
        canvas1.draw()
        chart1_widget = canvas1.get_tk_widget()
        chart1_widget.pack(fill='both', expand=True)
        
        # ================== PIE CHART: Safety Distribution ==================
        fig2, ax2 = plt.subplots(figsize=(6, 3.5))
        fig2.patch.set_facecolor('#0f3460')
        ax2.set_facecolor('#0f3460')
        
        safe_percentage = safety_percentage
        unsafe_percentage = max(0, 100 - safe_percentage)
        
        sizes = [safe_percentage, unsafe_percentage]
        labels = [f'Safe\n{safe_percentage:.1f}%', f'Unsafe\n{unsafe_percentage:.1f}%']
        colors = ['#4ECDC4', '#FF6B6B']
        
        # Create pie chart with shadow and explode
        explode = (0.05, 0)
        wedges, texts, autotexts = ax2.pie(sizes, explode=explode, labels=labels, 
                                          colors=colors, autopct='%1.1f%%',
                                          shadow=True, startangle=90,
                                          textprops={'fontweight': 'bold', 'color': 'white'})
        
        # Style the autopct text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(10)
        
        ax2.set_title('Safety Distribution', fontsize=12, fontweight='bold', 
                     pad=15, color='#4ECDC4')
        
        # Add a circle at the center for a donut chart effect
        centre_circle = plt.Circle((0,0), 0.70, fc='#0f3460')
        ax2.add_artist(centre_circle)
        
        # Add safety status in center
        safety_status = self.get_safety_status(safety_percentage).replace("✅", "").replace("❌", "").replace("⚠", "").strip()
        ax2.text(0, 0, safety_status, ha='center', va='center', 
                fontsize=11, fontweight='bold', color='#4ECDC4')
        
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
            
            # Show success message
            messagebox.showinfo("Success", f"Detection results saved to:\n{output_dir}")
            
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