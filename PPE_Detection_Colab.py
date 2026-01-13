"""
PPE Image Detection Desktop Application
AI Lab / Final Year Project
University Project - Deployment Phase

This application loads a pre-trained YOLOv8 model for PPE detection
and performs inference on selected images.

Training was conducted on Google Colab, and this app represents
the local deployment phase.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os
import threading

class PPEDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PPE Image Detection using YOLOv8")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.model = None
        self.model_loaded = False
        self.current_image = None
        self.original_image = None
        self.results = None
        
        # Set style
        self.setup_styles()
        
        # Build GUI
        self.create_widgets()
        
        # Status
        self.update_status("Ready to load model")
        
    def setup_styles(self):
        """Configure modern styling for the application"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.bg_color = "#f0f0f0"
        self.header_bg = "#2c3e50"
        self.section_bg = "#ffffff"
        self.accent_color = "#3498db"
        self.success_color = "#27ae60"
        self.warning_color = "#e74c3c"
        
        # Apply background
        self.root.configure(bg=self.bg_color)
        
    def create_widgets(self):
        """Create all GUI widgets organized in sections"""
        
        # Main container with padding
        main_container = tk.Frame(self.root, bg=self.bg_color, padx=20, pady=20)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # ==================== HEADER SECTION ====================
        header_frame = tk.Frame(main_container, bg=self.header_bg, padx=20, pady=15)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = tk.Label(
            header_frame,
            text="PPE Image Detection using YOLOv8",
            font=("Arial", 24, "bold"),
            fg="white",
            bg=self.header_bg
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame,
            text="AI Lab / University Project - Deployment Phase",
            font=("Arial", 12),
            fg="#ecf0f1",
            bg=self.header_bg
        )
        subtitle_label.pack(pady=(5, 0))
        
        # ==================== TWO-COLUMN LAYOUT ====================
        content_frame = tk.Frame(main_container, bg=self.bg_color)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column for controls
        left_column = tk.Frame(content_frame, bg=self.bg_color, width=400)
        left_column.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10))
        
        # Right column for images
        right_column = tk.Frame(content_frame, bg=self.bg_color)
        right_column.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # ==================== MODEL SECTION ====================
        model_section = self.create_section(left_column, "1. Model Configuration")
        
        # Training info (display only)
        train_info = tk.Label(
            model_section,
            text="Training Workflow:",
            font=("Arial", 10, "bold"),
            bg=self.section_bg
        )
        train_info.pack(anchor="w", pady=(0, 5))
        
        train_desc = tk.Label(
            model_section,
            text="✓ Model trained on Google Colab using YOLOv8\n✓ Dataset follows YOLO format\n✓ Training recommended on Colab due to hardware requirements",
            font=("Arial", 9),
            justify=tk.LEFT,
            bg=self.section_bg,
            fg="#7f8c8d"
        )
        train_desc.pack(anchor="w", pady=(0, 15))
        
        # Load model button
        self.load_btn = ttk.Button(
            model_section,
            text="Load trained_model.pt",
            command=self.load_model,
            style="Accent.TButton"
        )
        self.load_btn.pack(pady=(0, 10))
        
        # Model status
        self.model_status = tk.Label(
            model_section,
            text="❌ Model not loaded",
            font=("Arial", 10),
            bg=self.section_bg,
            fg=self.warning_color
        )
        self.model_status.pack()
        
        # ==================== IMAGE SELECTION SECTION ====================
        image_section = self.create_section(left_column, "2. Image Selection")
        
        # Image selection button
        self.select_btn = ttk.Button(
            image_section,
            text="Select Image for Detection",
            command=self.select_image,
            state="disabled"
        )
        self.select_btn.pack(pady=(0, 10))
        
        # Original image preview label
        orig_label = tk.Label(
            image_section,
            text="Original Image Preview:",
            font=("Arial", 10, "bold"),
            bg=self.section_bg
        )
        orig_label.pack(anchor="w", pady=(10, 5))
        
        # Original image canvas
        self.original_canvas = tk.Canvas(
            image_section,
            width=350,
            height=200,
            bg="#ecf0f1",
            highlightthickness=1,
            highlightbackground="#bdc3c7"
        )
        self.original_canvas.pack(pady=(0, 10))
        
        # Placeholder text
        self.original_canvas.create_text(
            175, 100,
            text="No image selected",
            fill="#95a5a6",
            font=("Arial", 10)
        )
        
        # ==================== DETECTION SECTION ====================
        detect_section = self.create_section(left_column, "3. Detection")
        
        # Run detection button
        self.detect_btn = ttk.Button(
            detect_section,
            text="Run PPE Detection",
            command=self.run_detection,
            state="disabled",
            style="Accent.TButton"
        )
        self.detect_btn.pack(pady=(0, 10))
        
        # ==================== RESULTS PANEL ====================
        results_section = self.create_section(left_column, "4. Detection Results")
        
        # Results text box with scrollbar
        results_frame = tk.Frame(results_section, bg=self.section_bg)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        self.results_text = tk.Text(
            results_frame,
            height=8,
            width=40,
            font=("Consolas", 10),
            bg="#f8f9fa",
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        # Insert initial message
        self.results_text.insert(tk.END, "Detection results will appear here.\n\n")
        self.results_text.insert(tk.END, "Expected PPE classes:\n")
        self.results_text.insert(tk.END, "• helmet\n• vest\n• gloves\n• boots\n• mask\n• goggles")
        self.results_text.config(state=tk.DISABLED)
        
        # ==================== IMAGE DISPLAY SECTION (Right Column) ====================
        # Output image label
        output_label = tk.Label(
            right_column,
            text="Detection Output with Bounding Boxes",
            font=("Arial", 12, "bold"),
            bg=self.bg_color
        )
        output_label.pack(anchor="w", pady=(0, 10))
        
        # Output image canvas
        self.output_canvas = tk.Canvas(
            right_column,
            width=700,
            height=500,
            bg="#ecf0f1",
            highlightthickness=2,
            highlightbackground="#bdc3c7"
        )
        self.output_canvas.pack(pady=(0, 10))
        
        # Placeholder text
        self.output_canvas.create_text(
            350, 250,
            text="Detection output will appear here\n\nBounding boxes with class names\nand confidence scores",
            fill="#95a5a6",
            font=("Arial", 11),
            justify=tk.CENTER
        )
        
        # Detection legend
        legend_frame = tk.Frame(right_column, bg=self.bg_color)
        legend_frame.pack(anchor="w")
        
        legend_label = tk.Label(
            legend_frame,
            text="Detection Legend:",
            font=("Arial", 10, "bold"),
            bg=self.bg_color
        )
        legend_label.pack(anchor="w", pady=(0, 5))
        
        legend_text = tk.Label(
            legend_frame,
            text="• Bounding boxes show detected PPE items\n• Each box includes class name and confidence %",
            font=("Arial", 9),
            bg=self.bg_color,
            fg="#7f8c8d",
            justify=tk.LEFT
        )
        legend_text.pack(anchor="w")
        
        # ==================== STATUS BAR ====================
        status_frame = tk.Frame(main_container, bg="#34495e", height=30)
        status_frame.pack(fill=tk.X, pady=(20, 0))
        status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(
            status_frame,
            text="Ready",
            font=("Arial", 10),
            bg="#34495e",
            fg="white"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        # Version label
        version_label = tk.Label(
            status_frame,
            text="AI Lab Project v1.0 | Deployment Phase",
            font=("Arial", 9),
            bg="#34495e",
            fg="#bdc3c7"
        )
        version_label.pack(side=tk.RIGHT, padx=10)
        
    def create_section(self, parent, title):
        """Create a styled section frame with title"""
        section = tk.Frame(parent, bg=self.section_bg, padx=15, pady=15)
        section.pack(fill=tk.X, pady=(0, 15))
        
        # Section title
        title_label = tk.Label(
            section,
            text=title,
            font=("Arial", 11, "bold"),
            bg=self.section_bg
        )
        title_label.pack(anchor="w", pady=(0, 10))
        
        return section
    
    def update_status(self, message, is_error=False):
        """Update the status bar message"""
        color = self.warning_color if is_error else "white"
        self.status_label.config(text=message, fg=color)
        self.root.update_idletasks()
    
    def load_model(self):
        """Load the pre-trained YOLOv8 model"""
        model_path = "trained_model.pt"
        
        if not os.path.exists(model_path):
            messagebox.showerror(
                "Model Not Found",
                f"Model file '{model_path}' not found in current directory.\n"
                f"Please ensure trained_model.pt is in:\n{os.getcwd()}"
            )
            self.update_status("Error: Model file not found", is_error=True)
            return
        
        try:
            self.update_status("Loading model...")
            self.load_btn.config(state="disabled", text="Loading...")
            
            # Load YOLOv8 model
            self.model = YOLO(model_path)
            self.model_loaded = True
            
            # Update UI
            self.model_status.config(text="✅ Model loaded successfully", fg=self.success_color)
            self.select_btn.config(state="normal")
            self.load_btn.config(text="Model Loaded", state="disabled")
            
            # Verify model classes
            class_names = self.model.names if hasattr(self.model, 'names') else {}
            self.update_status(f"Model loaded. Classes: {len(class_names)} detected")
            
        except Exception as e:
            self.model_loaded = False
            self.model_status.config(text="❌ Failed to load model", fg=self.warning_color)
            messagebox.showerror("Model Loading Error", f"Failed to load model:\n{str(e)}")
            self.update_status("Error loading model", is_error=True)
            self.load_btn.config(state="normal", text="Load trained_model.pt")
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(filetypes=file_types)
        
        if file_path:
            try:
                self.update_status(f"Loading image: {os.path.basename(file_path)}")
                
                # Load and display original image
                self.current_image_path = file_path
                self.original_image = cv2.imread(file_path)
                
                if self.original_image is None:
                    raise ValueError("Failed to load image")
                
                # Display original image
                self.display_image(self.original_image, self.original_canvas, 350, 200)
                
                # Clear output canvas
                self.output_canvas.delete("all")
                self.output_canvas.create_text(
                    350, 250,
                    text="Click 'Run PPE Detection' to analyze",
                    fill="#95a5a6",
                    font=("Arial", 11)
                )
                
                # Clear results text
                self.results_text.config(state=tk.NORMAL)
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "Image loaded. Ready for detection.")
                self.results_text.config(state=tk.DISABLED)
                
                # Enable detection button
                self.detect_btn.config(state="normal")
                
                self.update_status(f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Image Error", f"Failed to load image:\n{str(e)}")
                self.update_status("Error loading image", is_error=True)
    
    def display_image(self, image, canvas, max_width, max_height):
        """Display image on a canvas with proper scaling"""
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Calculate scaling factor
        width, height = pil_image.size
        scale = min(max_width/width, max_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(max_width//2, max_height//2, image=photo, anchor=tk.CENTER)
        canvas.image = photo  # Keep reference
    
    def run_detection(self):
        """Run PPE detection on the selected image"""
        if not self.model_loaded:
            messagebox.showwarning("Model Not Loaded", "Please load the model first.")
            return
        
        if self.original_image is None:
            messagebox.showwarning("No Image", "Please select an image first.")
            return
        
        # Disable button and update status
        self.detect_btn.config(state="disabled", text="Detecting...")
        self.update_status("Running PPE detection...")
        
        # Run detection in a separate thread to keep UI responsive
        threading.Thread(target=self._perform_detection, daemon=True).start()
    
    def _perform_detection(self):
        """Perform detection (runs in separate thread)"""
        try:
            # Run YOLOv8 inference
            results = self.model(self.original_image, conf=0.25)
            
            # Process results on main thread
            self.root.after(0, self._display_detection_results, results)
            
        except Exception as e:
            self.root.after(0, self._detection_error, str(e))
    
    def _detection_error(self, error_msg):
        """Handle detection error"""
        messagebox.showerror("Detection Error", f"Failed to run detection:\n{error_msg}")
        self.update_status("Detection failed", is_error=True)
        self.detect_btn.config(state="normal", text="Run PPE Detection")
    
    def _display_detection_results(self, results):
        """Display detection results on the main thread"""
        try:
            self.results = results[0]
            
            # Get annotated image
            annotated_image = self.results.plot()
            
            # Display annotated image
            self.display_image(annotated_image, self.output_canvas, 700, 500)
            
            # Update results text
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            
            if len(self.results.boxes) == 0:
                self.results_text.insert(tk.END, "No PPE items detected.\n")
                self.results_text.insert(tk.END, "Try adjusting confidence threshold or use different image.")
            else:
                self.results_text.insert(tk.END, f"Detected {len(self.results.boxes)} PPE item(s):\n\n")
                
                # Get class names from model
                class_names = self.results.names
                
                # Extract detection information
                detections = []
                for box in self.results.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = class_names[class_id]
                    detections.append((class_name, confidence))
                
                # Sort by confidence (highest first)
                detections.sort(key=lambda x: x[1], reverse=True)
                
                # Display each detection
                for i, (class_name, confidence) in enumerate(detections, 1):
                    self.results_text.insert(tk.END, 
                        f"{i}. {class_name.upper()}\n"
                        f"   Confidence: {confidence:.1%}\n\n")
            
            self.results_text.config(state=tk.DISABLED)
            
            # Update status and button
            self.update_status(f"Detection complete: {len(self.results.boxes)} items found")
            self.detect_btn.config(state="normal", text="Run PPE Detection")
            
        except Exception as e:
            messagebox.showerror("Display Error", f"Failed to display results:\n{str(e)}")
            self.update_status("Error displaying results", is_error=True)
            self.detect_btn.config(state="normal", text="Run PPE Detection")

def main():
    """Main application entry point"""
    root = tk.Tk()
    
    # Configure ttk styles
    style = ttk.Style()
    style.configure("Accent.TButton",
        foreground="white",
        background="#3498db",
        font=("Arial", 10, "bold"),
        borderwidth=0,
        focuscolor="none"
    )
    style.map("Accent.TButton",
        background=[("active", "#2980b9"), ("disabled", "#95a5a6")]
    )
    
    # Create and run app
    app = PPEDetectionApp(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()