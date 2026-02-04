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
        self.root.geometry("1280x720")  # Reduced from 1400x900
        self.root.configure(bg='#2c3e50')
        
        # Application state variables
        self.dataset_path = ""
        self.model = None
        self.trained_model_path = "trained_model.pt"
        self.class_names = []
        self.detection_results = []
        self.training_thread = None
        self.is_training = False
        self.ppe_requirements = {}
        self.current_model_path = ""
        
        # Image display variables
        self.original_image_pil = None
        self.detected_image_pil = None
        self.current_original_photo = None
        self.current_detected_photo = None
        
        # Create the main notebook (tab system)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Style configuration
        self.setup_styles()
        
        # Create the three main sections
        self.create_training_section()
        self.create_detection_section()
        self.create_results_section()
        
        # Set initial status
        self.update_status("Idle")
        
        # Schedule canvas initialization
        self.root.after(100, self.initialize_canvas_placeholders)
    
    def setup_styles(self):
        """Configure custom styles for the application"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Custom.TNotebook', background='#34495e')
        style.configure('Custom.TNotebook.Tab', 
                       background='#2c3e50', 
                       foreground='white',
                       padding=[15, 3])
        style.map('Custom.TNotebook.Tab', 
                 background=[('selected', '#3498db')],
                 foreground=[('selected', 'white')])
        
        self.notebook.configure(style='Custom.TNotebook')
    
    def create_training_section(self):
        """Create compact training section"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🏋️ TRAINING")
        
        # Main container
        main_container = tk.Frame(training_frame, bg='#ecf0f1')
        main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Header - reduced height
        header_frame = tk.Frame(main_container, bg='#3498db', height=50)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="🧠 PPE Model Training", 
                font=('Arial', 16, 'bold'), bg='#3498db', fg='white').pack(pady=12)
        
        # Content area with two columns
        content_frame = tk.Frame(main_container, bg='white')
        content_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Configuration (40% width)
        config_frame = tk.Frame(content_frame, bg='white', width=480)
        config_frame.pack(side='left', fill='both', padx=(0, 5))
        config_frame.pack_propagate(False)
        
        # Dataset configuration card - more compact
        dataset_card = tk.Frame(config_frame, bg='#f8f9fa', relief='groove', borderwidth=1)
        dataset_card.pack(fill='x', pady=(0, 8))
        
        tk.Label(dataset_card, text="📂 Dataset", 
                font=('Arial', 12, 'bold'), bg='#f8f9fa').pack(pady=8, padx=15, anchor='w')
        
        # Dataset selection
        dataset_row = tk.Frame(dataset_card, bg='#f8f9fa')
        dataset_row.pack(fill='x', padx=15, pady=(0, 8))
        
        tk.Label(dataset_row, text="Folder:", bg='#f8f9fa', 
                font=('Arial', 10)).pack(side='left', padx=(0, 5))
        
        self.dataset_entry = tk.Entry(dataset_row, width=30, font=('Arial', 9),
                                     relief='solid', borderwidth=1)
        self.dataset_entry.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        browse_btn = tk.Button(dataset_row, text="Browse", 
                              command=self.browse_dataset, 
                              bg='#3498db', fg='white', font=('Arial', 9),
                              relief='raised', borderwidth=1, padx=10, pady=3)
        browse_btn.pack(side='right')
        
        # Format info - more compact
        format_frame = tk.Frame(dataset_card, bg='#e8f4f8', relief='solid', borderwidth=1)
        format_frame.pack(fill='x', padx=15, pady=5)
        
        format_text = """Required: data.yaml, train/images, train/labels,
valid/images, valid/labels"""
        
        format_label = tk.Label(format_frame, text=format_text, justify='left', 
                               bg='#e8f4f8', font=('Courier', 8), padx=8, pady=6)
        format_label.pack()
        
        # Validate button
        validate_btn = tk.Button(dataset_card, text="🔍 Validate", 
                                command=self.validate_dataset, 
                                bg='#2ecc71', fg='white', font=('Arial', 10, 'bold'),
                                relief='raised', borderwidth=1, padx=15, pady=5)
        validate_btn.pack(pady=8)
        
        self.validation_label = tk.Label(dataset_card, text="", bg='#f8f9fa',
                                        font=('Arial', 9))
        self.validation_label.pack(pady=(0, 8))
        
        # Training parameters - more compact
        params_card = tk.Frame(config_frame, bg='#f8f9fa', relief='groove', borderwidth=1)
        params_card.pack(fill='x', pady=8)
        
        tk.Label(params_card, text="⚙️ Parameters", 
                font=('Arial', 12, 'bold'), bg='#f8f9fa').pack(pady=8, padx=15, anchor='w')
        
        # Compact parameter inputs
        params_container = tk.Frame(params_card, bg='#f8f9fa')
        params_container.pack(fill='x', padx=15, pady=5)
        
        # Epochs
        tk.Label(params_container, text="Epochs:", bg='#f8f9fa', 
                font=('Arial', 10)).grid(row=0, column=0, sticky='w', pady=3)
        self.epoch_entry = tk.Entry(params_container, width=12, font=('Arial', 9))
        self.epoch_entry.insert(0, "100")
        self.epoch_entry.grid(row=0, column=1, sticky='w', padx=5, pady=3)
        
        # Image size
        tk.Label(params_container, text="Size:", bg='#f8f9fa', 
                font=('Arial', 10)).grid(row=1, column=0, sticky='w', pady=3)
        self.size_var = tk.StringVar(value="640")
        size_menu = ttk.Combobox(params_container, textvariable=self.size_var, 
                                values=["320", "416", "512", "640", "768"], 
                                state='readonly', width=10, font=('Arial', 9))
        size_menu.grid(row=1, column=1, sticky='w', padx=5, pady=3)
        
        # Right panel - Training controls (60% width)
        control_frame = tk.Frame(content_frame, bg='white')
        control_frame.pack(side='right', fill='both', expand=True)
        
        # Status card - compact
        status_card = tk.Frame(control_frame, bg='#f8f9fa', relief='groove', borderwidth=1)
        status_card.pack(fill='x', pady=(0, 8))
        
        tk.Label(status_card, text="📊 Status", 
                font=('Arial', 12, 'bold'), bg='#f8f9fa').pack(pady=8, padx=15, anchor='w')
        
        # Status indicator - reduced size
        self.status_frame = tk.Frame(status_card, bg='#e0e0e0', height=35)
        self.status_frame.pack(fill='x', padx=15, pady=(0, 8))
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(self.status_frame, text="Status: Idle", 
                                    bg='#e0e0e0', font=('Arial', 10, 'bold'))
        self.status_label.pack(side='left', padx=10)
        
        self.status_indicator = tk.Label(self.status_frame, text="●", 
                                        fg='gray', font=('Arial', 18))
        self.status_indicator.pack(side='right', padx=10)
        
        # Training buttons - compact
        button_frame = tk.Frame(status_card, bg='#f8f9fa')
        button_frame.pack(fill='x', padx=15, pady=(0, 8))
        
        self.train_btn = tk.Button(button_frame, text="▶ Start", 
                                  command=self.start_training, 
                                  bg='#2ecc71', fg='white', font=('Arial', 10, 'bold'),
                                  relief='raised', borderwidth=1, padx=20, pady=6)
        self.train_btn.pack(side='left', padx=(0, 5))
        
        self.stop_btn = tk.Button(button_frame, text="⏹ Stop", 
                                 command=self.stop_training, 
                                 bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'),
                                 relief='raised', borderwidth=1, padx=20, pady=6,
                                 state='disabled')
        self.stop_btn.pack(side='left')
        
        # Training logs - reduced height
        log_card = tk.Frame(control_frame, bg='#f8f9fa', relief='groove', borderwidth=1)
        log_card.pack(fill='both', expand=True)
        
        tk.Label(log_card, text="📝 Logs", 
                font=('Arial', 12, 'bold'), bg='#f8f9fa').pack(pady=8, padx=15, anchor='w')
        
        self.log_text = scrolledtext.ScrolledText(log_card, height=15, 
                                                 bg='#2c3e50', fg='#ecf0f1',
                                                 font=('Courier', 8))
        self.log_text.pack(fill='both', expand=True, padx=15, pady=(0, 8))
    
    def create_detection_section(self):
        """Create compact detection section"""
        detection_frame = ttk.Frame(self.notebook)
        self.notebook.add(detection_frame, text="🔍 DETECTION")
        
        # Main container
        main_container = tk.Frame(detection_frame, bg='#ecf0f1')
        main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Header - reduced
        header_frame = tk.Frame(main_container, bg='#9b59b6', height=50)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="👁️ PPE Detection", 
                font=('Arial', 16, 'bold'), bg='#9b59b6', fg='white').pack(pady=12)
        
        # Content area
        content_frame = tk.Frame(main_container, bg='white')
        content_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel - Controls (25% width) - more compact
        control_panel = tk.Frame(content_frame, bg='white', width=280)
        control_panel.pack(side='left', fill='y', padx=(0, 5))
        control_panel.pack_propagate(False)
        
        # Model card - compact
        model_card = tk.Frame(control_panel, bg='#f8f9fa', relief='groove', borderwidth=1)
        model_card.pack(fill='x', pady=(0, 8))
        
        tk.Label(model_card, text="🤖 Model", 
                font=('Arial', 11, 'bold'), bg='#f8f9fa').pack(pady=6, padx=10, anchor='w')
        
        # Model buttons
        load_frame = tk.Frame(model_card, bg='#f8f9fa')
        load_frame.pack(fill='x', padx=10, pady=5)
        
        default_btn = tk.Button(load_frame, text="📁 Trained Model", 
                               command=self.load_default_model, 
                               bg='#3498db', fg='white', font=('Arial', 9),
                               relief='raised', borderwidth=1, padx=10, pady=5,
                               width=18)
        default_btn.pack(pady=3)
        
        custom_btn = tk.Button(load_frame, text="📂 Custom Model", 
                              command=self.select_custom_model, 
                              bg='#3498db', fg='white', font=('Arial', 9),
                              relief='raised', borderwidth=1, padx=10, pady=5,
                              width=18)
        custom_btn.pack(pady=3)
        
        # Model status
        status_frame = tk.Frame(model_card, bg='#e8f4f8', relief='solid', borderwidth=1)
        status_frame.pack(fill='x', padx=10, pady=8)
        
        self.model_status_label = tk.Label(status_frame, text="⏳ No model", 
                                          bg='#e8f4f8', fg='#7f8c8d', 
                                          font=('Arial', 9, 'bold'))
        self.model_status_label.pack(pady=6, padx=8)
        
        self.model_details_label = tk.Label(status_frame, text="", 
                                           bg='#e8f4f8', fg='#95a5a6',
                                           font=('Arial', 7), wraplength=240)
        self.model_details_label.pack(pady=(0, 6), padx=8)
        
        # Upload card - compact
        upload_card = tk.Frame(control_panel, bg='#f8f9fa', relief='groove', borderwidth=1)
        upload_card.pack(fill='x', pady=8)
        
        tk.Label(upload_card, text="📤 Upload", 
                font=('Arial', 11, 'bold'), bg='#f8f9fa').pack(pady=6, padx=10, anchor='w')
        
        upload_frame = tk.Frame(upload_card, bg='#f8f9fa')
        upload_frame.pack(fill='x', padx=10, pady=5)
        
        single_btn = tk.Button(upload_frame, text="🖼️ Image", 
                              command=self.upload_single_image, 
                              bg='#2ecc71', fg='white', font=('Arial', 9),
                              relief='raised', borderwidth=1, padx=10, pady=5,
                              width=18)
        single_btn.pack(pady=3)
        
        folder_btn = tk.Button(upload_frame, text="📁 Folder", 
                              command=self.upload_image_folder, 
                              bg='#2ecc71', fg='white', font=('Arial', 9),
                              relief='raised', borderwidth=1, padx=10, pady=5,
                              width=18)
        folder_btn.pack(pady=3)
        
        self.image_info_label = tk.Label(upload_card, text="", bg='#f8f9fa',
                                        font=('Arial', 8), fg='#2c3e50', wraplength=240)
        self.image_info_label.pack(pady=(0, 6))
        
        # Detect button
        self.detect_btn = tk.Button(control_panel, text="🚀 Detect", 
                                   command=self.run_detection, 
                                   bg='#e74c3c', fg='white', font=('Arial', 12, 'bold'),
                                   relief='raised', borderwidth=2, padx=20, pady=10,
                                   state='disabled')
        self.detect_btn.pack(pady=10)
        
        # Save option
        self.save_var = tk.BooleanVar(value=True)
        save_check = tk.Checkbutton(control_panel, text="💾 Save output", 
                                   variable=self.save_var, bg='white',
                                   font=('Arial', 9))
        save_check.pack(pady=5)
        
        # Right panel - Images (75% width)
        preview_panel = tk.Frame(content_frame, bg='white')
        preview_panel.pack(side='right', fill='both', expand=True)
        
        # Images side by side - reduced size for better fit
        image_container = tk.Frame(preview_panel, bg='#f8f9fa')
        image_container.pack(fill='both', expand=True)
        
        # Original image - 450x450
        orig_frame = tk.LabelFrame(image_container, text="Original", 
                                  font=('Arial', 10, 'bold'),
                                  bg='#2c3e50', fg='white', 
                                  relief='solid', borderwidth=1)
        orig_frame.pack(side='left', fill='both', expand=True, padx=(0, 3), pady=3)
        
        self.orig_image_frame = tk.Frame(orig_frame, bg='#34495e', width=450, height=450)
        self.orig_image_frame.pack(fill='both', expand=True, padx=3, pady=3)
        self.orig_image_frame.pack_propagate(False)
        
        self.original_canvas = tk.Canvas(self.orig_image_frame, bg='#34495e', 
                                        highlightthickness=0, width=444, height=444)
        self.original_canvas.pack(fill='both', expand=True)
        
        # Detected image - 450x450
        det_frame = tk.LabelFrame(image_container, text="Detection", 
                                 font=('Arial', 10, 'bold'),
                                 bg='#2c3e50', fg='white', 
                                 relief='solid', borderwidth=1)
        det_frame.pack(side='right', fill='both', expand=True, padx=(3, 0), pady=3)
        
        self.det_image_frame = tk.Frame(det_frame, bg='#34495e', width=450, height=450)
        self.det_image_frame.pack(fill='both', expand=True, padx=3, pady=3)
        self.det_image_frame.pack_propagate(False)
        
        self.detected_canvas = tk.Canvas(self.det_image_frame, bg='#34495e', 
                                        highlightthickness=0, width=444, height=444)
        self.detected_canvas.pack(fill='both', expand=True)
        
        # Navigation - compact
        nav_frame = tk.Frame(preview_panel, bg='#f8f9fa')
        nav_frame.pack(pady=8)
        
        self.prev_btn = tk.Button(nav_frame, text="◀ Prev",
                                 command=self.show_previous_image,
                                 bg='#3498db', fg='white', font=('Arial', 9, 'bold'),
                                 relief='raised', borderwidth=1, state='disabled',
                                 padx=12, pady=5)
        self.prev_btn.pack(side='left', padx=3)
        
        self.image_counter_label = tk.Label(nav_frame, text="Image 0/0",
                                           bg='#f8f9fa', font=('Arial', 10, 'bold'),
                                           fg='#2c3e50')
        self.image_counter_label.pack(side='left', padx=15)
        
        self.next_btn = tk.Button(nav_frame, text="Next ▶",
                                 command=self.show_next_image,
                                 bg='#3498db', fg='white', font=('Arial', 9, 'bold'),
                                 relief='raised', borderwidth=1, state='disabled',
                                 padx=12, pady=5)
        self.next_btn.pack(side='left', padx=3)
        
        # Initialize variables
        self.current_original_image = None
        self.current_detected_image = None
        self.image_paths = []
        self.current_image_index = 0
        self.detection_results_list = []
    
    def create_results_section(self):
        """Create compact results section"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📊 RESULTS")
        
        # Main container
        main_container = tk.Frame(results_frame, bg='#ecf0f1')
        main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Header - reduced
        header_frame = tk.Frame(main_container, bg='#e67e22', height=50)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame, text="📈 Detection Results", 
                font=('Arial', 16, 'bold'), bg='#e67e22', fg='white').pack(pady=12)
        
        # Content with two columns
        content_frame = tk.Frame(main_container, bg='white')
        content_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left column (45%)
        left_column = tk.Frame(content_frame, bg='white')
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Summary card - compact
        summary_card = tk.Frame(left_column, bg='#f8f9fa', relief='groove', borderwidth=1)
        summary_card.pack(fill='both', expand=True, pady=(0, 8))
        
        tk.Label(summary_card, text="📋 Summary", 
                font=('Arial', 11, 'bold'), bg='#f8f9fa').pack(pady=6, padx=10, anchor='w')
        
        # Two-column lists - reduced height
        summary_container = tk.Frame(summary_card, bg='#f8f9fa')
        summary_container.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Detected
        detected_frame = tk.Frame(summary_container, bg='#e8f6f3', relief='solid', borderwidth=1)
        detected_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 3), pady=3)
        
        tk.Label(detected_frame, text="✅ DETECTED", font=('Arial', 9, 'bold'), 
                bg='#27ae60', fg='white').pack(fill='x', pady=3)
        
        self.detected_listbox = tk.Listbox(detected_frame, height=6, bg='#e8f6f3',
                                          font=('Arial', 8), selectmode=tk.SINGLE, relief='flat')
        det_scroll = tk.Scrollbar(detected_frame, command=self.detected_listbox.yview)
        det_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.detected_listbox.config(yscrollcommand=det_scroll.set)
        self.detected_listbox.pack(fill='both', expand=True, padx=3, pady=3)
        
        # Missing
        missing_frame = tk.Frame(summary_container, bg='#fdebd0', relief='solid', borderwidth=1)
        missing_frame.grid(row=0, column=1, sticky='nsew', padx=(3, 0), pady=3)
        
        tk.Label(missing_frame, text="⚠ MISSING", font=('Arial', 9, 'bold'), 
                bg='#e67e22', fg='white').pack(fill='x', pady=3)
        
        self.missing_listbox = tk.Listbox(missing_frame, height=6, bg='#fdebd0',
                                         font=('Arial', 8), selectmode=tk.SINGLE, relief='flat')
        miss_scroll = tk.Scrollbar(missing_frame, command=self.missing_listbox.yview)
        miss_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.missing_listbox.config(yscrollcommand=miss_scroll.set)
        self.missing_listbox.pack(fill='both', expand=True, padx=3, pady=3)
        
        summary_container.columnconfigure(0, weight=1)
        summary_container.columnconfigure(1, weight=1)
        summary_container.rowconfigure(0, weight=1)
        
        self.summary_stats_label = tk.Label(summary_card, text="📊 No results yet", 
                                           bg='#f8f9fa', font=('Arial', 9, 'bold'))
        self.summary_stats_label.pack(pady=6)
        
        # Safety card - compact
        safety_card = tk.Frame(left_column, bg='#f8f9fa', relief='groove', borderwidth=1)
        safety_card.pack(fill='both', expand=True)
        
        tk.Label(safety_card, text="🛡️ Safety", 
                font=('Arial', 11, 'bold'), bg='#f8f9fa').pack(pady=6, padx=10, anchor='w')
        
        safety_display = tk.Frame(safety_card, bg='#2c3e50', relief='solid', borderwidth=1)
        safety_display.pack(fill='x', padx=10, pady=5)
        
        self.safety_percentage_label = tk.Label(safety_display, text="0.0%",
                                               font=('Arial', 28, 'bold'),
                                               bg='#2c3e50', fg='#ecf0f1')
        self.safety_percentage_label.pack(pady=6)
        
        self.safety_status_label = tk.Label(safety_display, text="NOT CALCULATED",
                                          font=('Arial', 12, 'bold'),
                                          bg='#2c3e50', fg='#bdc3c7')
        self.safety_status_label.pack(pady=(0, 6))
        
        self.safety_breakdown_label = tk.Label(safety_card, text="",
                                              bg='#f8f9fa', font=('Arial', 8), justify='left')
        self.safety_breakdown_label.pack(pady=6)
        
        # Right column (55%)
        right_column = tk.Frame(content_frame, bg='white')
        right_column.pack(side='right', fill='both', expand=True)
        
        # Charts - reduced size
        charts_card = tk.Frame(right_column, bg='#f8f9fa', relief='groove', borderwidth=1)
        charts_card.pack(fill='both', expand=True, pady=(0, 8))
        
        tk.Label(charts_card, text="📊 Analytics", 
                font=('Arial', 11, 'bold'), bg='#f8f9fa').pack(pady=6, padx=10, anchor='w')
        
        chart_container = tk.Frame(charts_card, bg='#f8f9fa')
        chart_container.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Bar chart - reduced height
        bar_chart_frame = tk.Frame(chart_container, bg='white', relief='solid', borderwidth=1)
        bar_chart_frame.pack(fill='both', expand=True, pady=(0, 5))
        
        tk.Label(bar_chart_frame, text="📈 Detection Count", 
                font=('Arial', 10, 'bold'), bg='white').pack(pady=5)
        
        self.bar_chart_canvas = tk.Canvas(bar_chart_frame, bg='white', height=130)
        self.bar_chart_canvas.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        
        # Pie chart - reduced height
        pie_chart_frame = tk.Frame(chart_container, bg='white', relief='solid', borderwidth=1)
        pie_chart_frame.pack(fill='both', expand=True)
        
        tk.Label(pie_chart_frame, text="🥧 Safety Distribution", 
                font=('Arial', 10, 'bold'), bg='white').pack(pady=5)
        
        self.pie_chart_canvas = tk.Canvas(pie_chart_frame, bg='white', height=130)
        self.pie_chart_canvas.pack(fill='both', expand=True, padx=5, pady=(0, 5))
        
        # Logs - compact
        logs_card = tk.Frame(right_column, bg='#f8f9fa', relief='groove', borderwidth=1)
        logs_card.pack(fill='both', expand=True)
        
        tk.Label(logs_card, text="📝 Details", 
                font=('Arial', 11, 'bold'), bg='#f8f9fa').pack(pady=6, padx=10, anchor='w')
        
        self.results_text = scrolledtext.ScrolledText(logs_card, height=8,
                                                     bg='#2c3e50', fg='#ecf0f1',
                                                     font=('Arial', 8))
        self.results_text.pack(fill='both', expand=True, padx=10, pady=(0, 8))

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
        """Validate the dataset structure"""
        dataset_path = self.dataset_entry.get()
        
        if not dataset_path:
            self.validation_label.config(text="⚠ Select folder", fg='#e67e22')
            return False
        
        required_folders = [
            ('train/images', 'Train images'),
            ('train/labels', 'Train labels'),
            ('valid/images', 'Valid images'),
            ('valid/labels', 'Valid labels')
        ]
        
        all_valid = True
        missing_items = []
        
        for folder, name in required_folders:
            if not os.path.exists(os.path.join(dataset_path, folder)):
                all_valid = False
                missing_items.append(name)
        
        yaml_path = os.path.join(dataset_path, 'data.yaml')
        if not os.path.exists(yaml_path):
            all_valid = False
            missing_items.append('data.yaml')
        
        if all_valid:
            try:
                with open(yaml_path, 'r') as f:
                    data_yaml = yaml.safe_load(f)
                
                if 'names' in data_yaml:
                    self.class_names = data_yaml['names']
                    validation_msg = f"✓ Valid! {len(self.class_names)} classes"
                    self.initialize_ppe_requirements()
                else:
                    validation_msg = "⚠ Missing 'names'"
                    all_valid = False
                
                self.validation_label.config(text=validation_msg, fg='#27ae60')
                
            except Exception as e:
                self.validation_label.config(text=f"✗ Error: {str(e)[:30]}", fg='#e74c3c')
                all_valid = False
        else:
            self.validation_label.config(text=f"✗ Missing: {', '.join(missing_items)[:40]}", fg='#e74c3c')
        
        return all_valid
    
    def initialize_ppe_requirements(self):
        """Initialize PPE requirements"""
        self.ppe_requirements = {}
        
        critical_ppe = {
            'helmet': {'min_instances': 1, 'critical': True, 'weight': 2.0},
            'hardhat': {'min_instances': 1, 'critical': True, 'weight': 2.0},
            'vest': {'min_instances': 1, 'critical': True, 'weight': 1.8},
            'glove': {'min_instances': 2, 'critical': True, 'weight': 1.5},
            'boot': {'min_instances': 2, 'critical': True, 'weight': 1.5},
            'goggle': {'min_instances': 1, 'critical': True, 'weight': 1.7},
            'mask': {'min_instances': 1, 'critical': True, 'weight': 1.9}
        }
        
        for i, class_name in enumerate(self.class_names):
            class_name_lower = class_name.lower()
            matched = False
            
            for key, requirements in critical_ppe.items():
                if key in class_name_lower:
                    self.ppe_requirements[i] = requirements.copy()
                    matched = True
                    break
            
            if not matched:
                self.ppe_requirements[i] = {'min_instances': 1, 'critical': False, 'weight': 1.0}
    
    def update_status(self, status, color='#95a5a6'):
        """Update training status"""
        self.status_label.config(text=f"Status: {status}")
        self.status_indicator.config(fg=color)
        
        if status == "Training...":
            self.train_btn.config(state='disabled')
            self.stop_btn.config(state='normal')
            self.is_training = True
        else:
            self.train_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.is_training = False
    
    def log_message(self, message):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def start_training(self):
        """Start training"""
        if not self.validate_dataset():
            messagebox.showerror("Error", "Fix dataset errors first")
            return
        
        try:
            epochs = int(self.epoch_entry.get())
            img_size = int(self.size_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid parameters")
            return
        
        if self.is_training:
            return
        
        self.log_text.delete(1.0, tk.END)
        self.training_thread = threading.Thread(
            target=self.run_training, args=(epochs, img_size), daemon=True
        )
        self.training_thread.start()
    
    def run_training(self, epochs, img_size):
        """Run training"""
        try:
            self.update_status("Training...", '#f39c12')
            self.log_message("Starting YOLOv8 training...")
            self.log_message(f"Epochs={epochs}, Size={img_size}")
            
            model = YOLO("yolov8n.yaml")
            
            if os.path.exists(self.trained_model_path):
                self.log_message("Resuming from checkpoint...")
                model = YOLO(self.trained_model_path)
            
            yaml_path = os.path.join(self.dataset_path, 'data.yaml')
            
            results = model.train(
                data=yaml_path,
                epochs=epochs,
                imgsz=img_size,
                pretrained=False,
                resume=os.path.exists(self.trained_model_path),
                project=".",
                name="runs",
                save=True,
                exist_ok=True
            )
            
            model.save(self.trained_model_path)
            
            self.log_message("✓ Training complete!")
            self.log_message(f"✓ Saved: {self.trained_model_path}")
            self.update_status("Completed", '#27ae60')
            
        except Exception as e:
            self.log_message(f"✗ Error: {str(e)}")
            self.update_status("Error", '#e74c3c')
            messagebox.showerror("Error", str(e))
    
    def stop_training(self):
        """Stop training"""
        if self.is_training:
            self.update_status("Stopping...", '#e67e22')
            self.log_message("Stopped by user")
            self.update_status("Idle", '#95a5a6')
    
    # ============================================================================
    # MODEL LOADING
    # ============================================================================
    
    def load_default_model(self):
        """Load default trained model"""
        try:
            if not os.path.exists(self.trained_model_path):
                messagebox.showerror("Error", "No trained model found")
                return
            
            self.model = YOLO(self.trained_model_path)
            self.current_model_path = self.trained_model_path
            
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            
            self.initialize_ppe_requirements()
            
            self.model_status_label.config(text=f"✓ Loaded: {len(self.class_names)} classes", fg='#27ae60')
            self.model_details_label.config(text=f"{self.trained_model_path}")
            self.detect_btn.config(state='normal')
            self.log_message(f"Model loaded: {len(self.class_names)} classes")
            
        except Exception as e:
            self.model_status_label.config(text=f"✗ Error", fg='#e74c3c')
            messagebox.showerror("Error", str(e))
    
    def select_custom_model(self):
        """Select custom model"""
        file_path = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("PyTorch", "*.pt"), ("All", "*.*")]
        )
        if file_path:
            self.load_custom_model(file_path)
    
    def load_custom_model(self, model_path):
        """Load custom model"""
        try:
            self.model = YOLO(model_path)
            self.current_model_path = model_path
            
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            
            self.initialize_ppe_requirements()
            
            display_path = model_path if len(model_path) <= 40 else "..." + model_path[-37:]
            
            self.model_status_label.config(text=f"✓ Custom: {len(self.class_names)} classes", fg='#27ae60')
            self.model_details_label.config(text=display_path)
            self.detect_btn.config(state='normal')
            self.log_message(f"Custom model: {model_path}")
            
        except Exception as e:
            self.model_status_label.config(text=f"✗ Error", fg='#e74c3c')
            messagebox.showerror("Error", str(e))
    
    # ============================================================================
    # DETECTION
    # ============================================================================
    
    def initialize_canvas_placeholders(self):
        """Show placeholders"""
        self.show_placeholder_images()
    
    def show_placeholder_images(self):
        """Show placeholder text"""
        self.original_canvas.delete("all")
        self.detected_canvas.delete("all")
        
        self.original_canvas.create_text(
            222, 222, text="Upload image\n(Click button above)",
            fill='#95a5a6', font=('Arial', 11), justify='center'
        )
        
        self.detected_canvas.create_text(
            222, 222, text="Detection results\n(After detection)",
            fill='#95a5a6', font=('Arial', 11), justify='center'
        )
    
    def upload_single_image(self):
        """Upload single image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.image_paths = [file_path]
            self.current_image_index = 0
            self.display_original_image(file_path)
            self.image_info_label.config(text=f"📁 {os.path.basename(file_path)[:30]}")
            self.update_image_navigation()
    
    def upload_image_folder(self):
        """Upload folder"""
        folder_path = filedialog.askdirectory(title="Select Folder")
        
        if folder_path:
            valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'}
            self.image_paths = [
                os.path.join(folder_path, f) 
                for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in valid_ext
            ]
            
            if self.image_paths:
                self.current_image_index = 0
                self.display_original_image(self.image_paths[0])
                self.image_info_label.config(text=f"📂 {len(self.image_paths)} images")
                self.update_image_navigation()
            else:
                messagebox.showwarning("Warning", "No valid images")
    
    def display_original_image(self, image_path):
        """Display original image at 450x450"""
        try:
            self.original_canvas.delete("all")
            
            img = Image.open(image_path)
            self.original_image_pil = img.copy()
            
            # Fit to 444x444
            img_width, img_height = img.size
            scale = min(444 / img_width, 444 / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            new_img = Image.new('RGB', (444, 444), color='#34495e')
            x_offset = (444 - new_width) // 2
            y_offset = (444 - new_height) // 2
            new_img.paste(img_resized, (x_offset, y_offset))
            
            self.current_original_photo = ImageTk.PhotoImage(new_img)
            
            self.original_canvas.create_image(222, 222, image=self.current_original_photo, anchor='center')
            self.original_canvas.image = self.current_original_photo
            
            if self.model:
                self.detect_btn.config(state='normal')
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load: {str(e)}")
            self.show_placeholder_images()
    
    def display_detected_image(self, image_path, results):
        """Display detected image at 450x450"""
        try:
            self.detected_canvas.delete("all")
            
            img_cv = cv2.imread(image_path)
            if img_cv is None:
                raise ValueError(f"Cannot read: {image_path}")
            
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class{cls}"
                    color = self.get_class_color(cls)
                    
                    cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    label = f"{class_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    
                    cv2.rectangle(img_rgb, (int(x1), int(y1) - th - 8), 
                                (int(x1) + tw, int(y1)), color, -1)
                    cv2.putText(img_rgb, label, (int(x1), int(y1) - 4),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            img_pil = Image.fromarray(img_rgb)
            self.detected_image_pil = img_pil.copy()
            
            # Fit to 444x444
            img_width, img_height = img_pil.size
            scale = min(444 / img_width, 444 / img_height)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            img_resized = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            new_img = Image.new('RGB', (444, 444), color='#34495e')
            x_offset = (444 - new_width) // 2
            y_offset = (444 - new_height) // 2
            new_img.paste(img_resized, (x_offset, y_offset))
            
            self.current_detected_photo = ImageTk.PhotoImage(new_img)
            
            self.detected_canvas.create_image(222, 222, image=self.current_detected_photo, anchor='center')
            self.detected_canvas.image = self.current_detected_photo
            
        except Exception as e:
            messagebox.showerror("Error", f"Display error: {str(e)}")
    
    def get_class_color(self, class_id):
        """Get color for class"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (255, 165, 0), (128, 0, 128), (0, 128, 128), (128, 128, 0)
        ]
        return colors[class_id % len(colors)]
    
    def run_detection(self):
        """Run detection"""
        if not self.model:
            messagebox.showerror("Error", "Load model first")
            return
        
        if not self.image_paths:
            messagebox.showwarning("Warning", "Upload image first")
            return
        
        try:
            self.detection_results_list = []
            current_image_path = self.image_paths[self.current_image_index]
            
            results = self.model(current_image_path, conf=0.25)
            self.detection_results_list.append(results)
            
            self.display_detected_image(current_image_path, results)
            self.process_detection_results(results, current_image_path)
            self.update_image_navigation()
            
            if self.save_var.get():
                self.save_detection_output(results, current_image_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Detection failed: {str(e)}")
    
    def show_previous_image(self):
        """Previous image"""
        if self.image_paths and self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_original_image(self.image_paths[self.current_image_index])
            self.update_image_navigation()
            
            if self.current_image_index < len(self.detection_results_list):
                results = self.detection_results_list[self.current_image_index]
                self.display_detected_image(self.image_paths[self.current_image_index], results)
                self.process_detection_results(results, self.image_paths[self.current_image_index])
            else:
                self.show_placeholder_images()
                self.clear_results()
    
    def show_next_image(self):
        """Next image"""
        if self.image_paths and self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.display_original_image(self.image_paths[self.current_image_index])
            self.update_image_navigation()
            
            if self.current_image_index < len(self.detection_results_list):
                results = self.detection_results_list[self.current_image_index]
                self.display_detected_image(self.image_paths[self.current_image_index], results)
                self.process_detection_results(results, self.image_paths[self.current_image_index])
            else:
                self.show_placeholder_images()
                self.clear_results()
    
    def clear_results(self):
        """Clear results"""
        self.results_text.delete(1.0, tk.END)
        self.detected_listbox.delete(0, tk.END)
        self.missing_listbox.delete(0, tk.END)
        self.summary_stats_label.config(text="📊 No results")
        self.safety_percentage_label.config(text="0.0%")
        self.safety_status_label.config(text="NOT CALCULATED")
        self.safety_breakdown_label.config(text="")
        
        self.bar_chart_canvas.delete("all")
        self.pie_chart_canvas.delete("all")
    
    def update_image_navigation(self):
        """Update navigation"""
        if len(self.image_paths) > 1:
            self.prev_btn.config(state='normal' if self.current_image_index > 0 else 'disabled')
            self.next_btn.config(state='normal' if self.current_image_index < len(self.image_paths) - 1 else 'disabled')
            self.image_counter_label.config(text=f"📄 {self.current_image_index + 1}/{len(self.image_paths)}")
        else:
            self.prev_btn.config(state='disabled')
            self.next_btn.config(state='disabled')
            self.image_counter_label.config(text="")
    
    # ============================================================================
    # RESULTS PROCESSING
    # ============================================================================
    
    def process_detection_results(self, results, image_path):
        """Process results"""
        self.results_text.delete(1.0, tk.END)
        self.detected_listbox.delete(0, tk.END)
        self.missing_listbox.delete(0, tk.END)
        
        detected_classes = {}
        class_counts = {}
        all_classes = set(range(len(self.class_names))) if self.class_names else set()
        detected_class_ids = set()
        total_confidence = 0
        total_detections = 0
        
        if hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes
            
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
            
            results_text = f"{'='*50}\nDETECTION RESULTS\n{'='*50}\n\n"
            results_text += f"📁 {os.path.basename(image_path)}\n"
            results_text += f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            results_text += f"{'-'*50}\n\n"
            
            if detected_classes:
                results_text += "✅ DETECTED:\n"
                for class_name, confidences in detected_classes.items():
                    count = len(confidences)
                    avg_conf = sum(confidences) / count
                    results_text += f"• {class_name}: {count}× ({avg_conf:.1%})\n"
                    self.detected_listbox.insert(tk.END, f"{class_name} ({count}×)")
            else:
                results_text += "❌ NO ITEMS DETECTED\n"
            
            results_text += f"\n{'-'*50}\n\n"
            
            if self.class_names:
                non_ppe = ['person', 'human', 'people']
                ppe_classes = [i for i, n in enumerate(self.class_names) 
                             if not any(x in n.lower() for x in non_ppe)]
                missing = set(ppe_classes) - detected_class_ids
                
                if missing:
                    results_text += "⚠ MISSING:\n"
                    for cls in missing:
                        if cls < len(self.class_names):
                            results_text += f"• {self.class_names[cls]}\n"
                            self.missing_listbox.insert(tk.END, self.class_names[cls])
                else:
                    results_text += "✅ ALL DETECTED\n"
            
            avg_confidence = total_confidence / total_detections if total_detections > 0 else 0
            safety_percentage = self.calculate_safety_percentage(class_counts, all_classes)
            
            results_text += f"\n{'-'*50}\n\n"
            results_text += f"📈 SAFETY: {safety_percentage:.1f}%\n"
            results_text += f"Status: {self.get_safety_status(safety_percentage)}\n\n"
            results_text += f"Statistics:\n"
            results_text += f"  Detected: {len(detected_classes)}/{len(all_classes)}\n"
            results_text += f"  Total: {total_detections}\n"
            results_text += f"  Avg Conf: {avg_confidence:.1%}\n"
            
            self.detection_results = {
                'detected': detected_classes,
                'missing': missing if 'missing' in locals() else set(),
                'class_counts': class_counts,
                'safety_percentage': safety_percentage,
                'total_classes': len(all_classes),
                'detected_count': len(detected_classes),
                'missing_count': self.missing_listbox.size(),
                'total_detections': total_detections,
                'avg_confidence': avg_confidence
            }
        else:
            results_text = "❌ No detections\n"
            self.detection_results = {
                'detected': {}, 'missing': set(), 'class_counts': {},
                'safety_percentage': 0.0, 'total_classes': len(all_classes),
                'detected_count': 0, 'missing_count': 0,
                'total_detections': 0, 'avg_confidence': 0.0
            }
        
        self.results_text.insert(tk.END, results_text)
        self.update_summary_statistics()
        self.update_safety_display()
        self.update_charts()
    
    def calculate_safety_percentage(self, class_counts, all_classes):
        """Calculate safety percentage"""
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
                
                if detected >= required:
                    class_score = weight * 100
                elif detected > 0:
                    class_score = weight * 100 * (detected / required)
                else:
                    class_score = 0
                
                total_score += class_score
                max_score += weight * 100
        
        return round((total_score / max_score) * 100, 1) if max_score > 0 else 0.0
    
    def get_safety_status(self, percentage):
        """Get safety status"""
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
        """Update summary stats"""
        if not self.detection_results:
            self.summary_stats_label.config(text="📊 No results")
            return
        
        detected = self.detection_results['detected_count']
        missing = self.detection_results['missing_count']
        total = self.detection_results['total_classes']
        
        self.summary_stats_label.config(text=f"📊 {detected}/{total} detected | {missing}/{total} missing")
    
    def update_safety_display(self):
        """Update safety display"""
        if not self.detection_results:
            return
        
        safety_percentage = self.detection_results['safety_percentage']
        safety_status = self.get_safety_status(safety_percentage)
        
        self.safety_percentage_label.config(text=f"{safety_percentage:.1f}%")
        
        if "EXCELLENT" in safety_status:
            color = '#27ae60'
        elif "GOOD" in safety_status:
            color = '#2ecc71'
        elif "MODERATE" in safety_status:
            color = '#f39c12'
        elif "POOR" in safety_status:
            color = '#e67e22'
        else:
            color = '#e74c3c'
        
        self.safety_status_label.config(text=safety_status, fg=color)
        self.safety_percentage_label.config(fg=color)
        
        detected = self.detection_results['detected_count']
        missing = self.detection_results['missing_count']
        total = self.detection_results['total_classes']
        avg_conf = self.detection_results.get('avg_confidence', 0)
        
        breakdown = f"Detected: {detected}/{total}\nMissing: {missing}/{total}\nConf: {avg_conf:.1%}"
        self.safety_breakdown_label.config(text=breakdown)
    
    def update_charts(self):
        """Update charts"""
        if not self.detection_results:
            return
        
        for widget in self.bar_chart_canvas.winfo_children():
            widget.destroy()
        for widget in self.pie_chart_canvas.winfo_children():
            widget.destroy()
        
        detected_classes = self.detection_results.get('detected', {})
        safety_percentage = self.detection_results['safety_percentage']
        
        # Bar chart - smaller
        fig1, ax1 = plt.subplots(figsize=(5.5, 2.2))
        fig1.patch.set_facecolor('#f8f9fa')
        ax1.set_facecolor('#ffffff')
        
        if detected_classes:
            class_names = list(detected_classes.keys())
            counts = [len(detected_classes[name]) for name in class_names]
            
            sorted_idx = np.argsort(counts)[::-1]
            class_names = [class_names[i] for i in sorted_idx]
            counts = [counts[i] for i in sorted_idx]
            
            max_items = min(6, len(class_names))
            class_names = class_names[:max_items]
            counts = counts[:max_items]
            
            colors = plt.cm.Set3(np.arange(len(class_names)) / len(class_names))
            bars = ax1.bar(range(len(class_names)), counts, color=colors, alpha=0.8)
            
            ax1.set_xlabel('PPE', fontweight='bold', fontsize=8)
            ax1.set_ylabel('Count', fontweight='bold', fontsize=8)
            ax1.set_title('Detections', fontsize=9, fontweight='bold', pad=8)
            ax1.set_xticks(range(len(class_names)))
            ax1.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
            ax1.tick_params(axis='y', labelsize=7)
            
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{int(height)}', ha='center', va='bottom', fontsize=7)
        else:
            ax1.text(0.5, 0.5, 'No PPE Detected', 
                    ha='center', va='center', transform=ax1.transAxes, 
                    fontsize=10, fontweight='bold', color='#7f8c8d')
        
        plt.tight_layout()
        canvas1 = FigureCanvasTkAgg(fig1, master=self.bar_chart_canvas)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill='both', expand=True)
        
        # Pie chart - smaller
        fig2, ax2 = plt.subplots(figsize=(5.5, 2.2))
        fig2.patch.set_facecolor('#f8f9fa')
        ax2.set_facecolor('#ffffff')
        
        safe = safety_percentage
        unsafe = max(0, 100 - safe)
        
        sizes = [safe, unsafe]
        labels = [f'Safe\n{safe:.1f}%', f'Unsafe\n{unsafe:.1f}%']
        colors = ['#27ae60', '#e74c3c']
        
        wedges, texts, autotexts = ax2.pie(sizes, explode=(0.05, 0), labels=labels, 
                                          colors=colors, autopct='%1.1f%%',
                                          shadow=True, startangle=90,
                                          textprops={'fontweight': 'bold', 'fontsize': 7})
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(8)
        
        ax2.set_title('Safety', fontsize=9, fontweight='bold', pad=8)
        
        centre = plt.Circle((0,0), 0.70, fc='white')
        ax2.add_artist(centre)
        
        status = self.get_safety_status(safe).replace("✅", "").replace("❌", "").replace("⚠", "").strip()
        ax2.text(0, 0, status, ha='center', va='center', 
                fontsize=8, fontweight='bold', color='#2c3e50')
        
        plt.tight_layout()
        canvas2 = FigureCanvasTkAgg(fig2, master=self.pie_chart_canvas)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill='both', expand=True)
    
    def save_detection_output(self, results, image_path):
        """Save detection output"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"detection_output_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            original_image = cv2.imread(image_path)
            
            if hasattr(results[0], 'boxes'):
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    class_name = self.class_names[cls] if cls < len(self.class_names) else f"Class{cls}"
                    color = self.get_class_color(cls)
                    color_bgr = (color[2], color[1], color[0])
                    
                    cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), color_bgr, 2)
                    label = f"{class_name}: {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(original_image, (int(x1), int(y1) - th - 8), 
                                (int(x1) + tw, int(y1)), color_bgr, -1)
                    cv2.putText(original_image, label, (int(x1), int(y1) - 4),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
            cv2.imwrite(output_path, original_image)
            
            results_path = os.path.join(output_dir, f"{base_name}_results.txt")
            with open(results_path, 'w') as f:
                f.write(self.results_text.get(1.0, tk.END))
            
            self.log_message(f"✓ Saved to: {output_dir}")
            
        except Exception as e:
            self.log_message(f"✗ Save error: {str(e)}")


def main():
    """Main function"""
    root = tk.Tk()
    app = PPE_Detection_App(root)
    root.mainloop()

if __name__ == "__main__":
    main()