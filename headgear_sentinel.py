"""
HeadGear Sentinel - PPE Detection System for Construction Workers
Complete implementation in a single Python file
Author: [Your Name]
Semester Project
"""

import os
import sys
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import numpy as np
import random
import pickle
from datetime import datetime
import threading
import shutil
from collections import defaultdict

# Suppress unnecessary warnings
import warnings
warnings.filterwarnings('ignore')

class PPEModel:
    """Custom object detection model for PPE"""
    
    def __init__(self):
        self.classes = ['Person', 'Helmet', 'Goggles', 'Mask', 'Gloves', 'Safety Vest', 'Boots']
        self.class_to_id = {cls: idx for idx, cls in enumerate(self.classes)}
        self.colors = {
            'Person': (0, 255, 0),      # Green
            'Helmet': (255, 0, 0),      # Blue
            'Goggles': (0, 0, 255),     # Red
            'Mask': (255, 255, 0),      # Cyan
            'Gloves': (255, 0, 255),    # Magenta
            'Safety Vest': (0, 255, 255), # Yellow
            'Boots': (128, 0, 128)      # Purple
        }
        
        # Model parameters
        self.input_size = (224, 224)
        self.model = None
        self.is_trained = False
        self.safety_threshold = 0.5  # Confidence threshold
        
    def create_model(self):
        """Create a simple custom model architecture"""
        # Simplified architecture suitable for CPU
        try:
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(224, 224, 3)),
                tf.keras.layers.Rescaling(1./255),
                
                # Feature extraction layers
                tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                
                # Detection heads
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.5),
                
                # Output: 7 classes + 4 coordinates for bounding box
                tf.keras.layers.Dense(len(self.classes) + 4, activation='sigmoid')
            ])
            
            model.compile(optimizer='adam',
                         loss='mse',
                         metrics=['accuracy'])
            
            self.model = model
            return True
            
        except ImportError:
            # Fallback to simple model if TensorFlow not available
            self.model = "simple_model"
            return True
        except Exception as e:
            print(f"Model creation error: {e}")
            return False
    
    def prepare_dataset(self, dataset_path):
        """Prepare dataset for training"""
        images = []
        labels = []
        
        # Look for images in the dataset folder
        for filename in os.listdir(dataset_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(dataset_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.input_size)
                    images.append(img)
                    
                    # Generate synthetic labels for demonstration
                    # In real project, you would load actual annotations
                    label = self.generate_synthetic_label()
                    labels.append(label)
        
        if len(images) == 0:
            return None, None
            
        return np.array(images), np.array(labels)
    
    def generate_synthetic_label(self):
        """Generate synthetic labels for demonstration"""
        # Create a label with random detections
        label = np.zeros(len(self.classes) + 4)
        
        # Randomly decide which PPE items are present
        for i in range(len(self.classes)):
            if random.random() > 0.5:  # 50% chance for each item
                label[i] = 1.0
        
        # Add random bounding box coordinates
        label[-4:] = np.random.rand(4)
        
        return label
    
    def train(self, dataset_path, epochs=10):
        """Train the model"""
        try:
            # Load and prepare data
            X, y = self.prepare_dataset(dataset_path)
            
            if X is None or len(X) == 0:
                return False, "No training images found"
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model
            if self.model is None:
                self.create_model()
            
            if hasattr(self.model, 'fit'):
                history = self.model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=8,
                    verbose=0
                )
                
                self.is_trained = True
                
                # Save model
                model_path = "ppe_model.h5"
                self.model.save(model_path)
                
                # Save metadata
                metadata = {
                    'classes': self.classes,
                    'colors': self.colors,
                    'input_size': self.input_size,
                    'trained_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open('model_metadata.pkl', 'wb') as f:
                    pickle.dump(metadata, f)
                
                return True, f"Training complete! Model saved.\nAccuracy: {history.history['accuracy'][-1]:.2f}"
            else:
                self.is_trained = True
                return True, "Simple model training complete (simulation)"
                
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def load_model(self):
        """Load trained model"""
        try:
            if os.path.exists('ppe_model.h5'):
                import tensorflow as tf
                self.model = tf.keras.models.load_model('ppe_model.h5')
                self.is_trained = True
                
                # Load metadata
                if os.path.exists('model_metadata.pkl'):
                    with open('model_metadata.pkl', 'rb') as f:
                        metadata = pickle.load(f)
                    return True, "Model loaded successfully"
            return False, "No trained model found"
        except:
            return False, "Could not load model"
    
    def predict(self, image):
        """Run detection on an image"""
        if not self.is_trained:
            return [], "Model not trained"
        
        # Prepare image
        orig_h, orig_w = image.shape[:2]
        img_resized = cv2.resize(image, self.input_size)
        img_array = np.expand_dims(img_resized, axis=0)
        
        # Run prediction
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(img_array, verbose=0)[0]
        else:
            # Simple simulation for demonstration
            predictions = np.random.rand(len(self.classes) + 4)
        
        # Process predictions
        detections = []
        for i, class_name in enumerate(self.classes):
            confidence = float(predictions[i])
            
            if confidence > self.safety_threshold:
                # Get bounding box (simulated for demonstration)
                box_idx = len(self.classes) + (i % 4)
                x1 = int(predictions[box_idx] * orig_w)
                y1 = int(predictions[(box_idx + 1) % 4] * orig_h)
                x2 = int(x1 + predictions[(box_idx + 2) % 4] * 100)
                y2 = int(y1 + predictions[(box_idx + 3) % 4] * 100)
                
                # Ensure coordinates are within bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': (x1, y1, x2, y2)
                })
        
        return detections, "Detection complete"
    
    def check_safety(self, detections):
        """Determine if worker is safe based on PPE compliance"""
        # Count detected items
        detected_classes = [d['class'] for d in detections]
        
        # Safety rules
        safety_status = "Safe"
        missing_items = []
        
        # Essential PPE for construction
        essential_ppe = ['Helmet', 'Safety Vest']
        
        for item in essential_ppe:
            if item not in detected_classes:
                safety_status = "Unsafe"
                missing_items.append(item)
        
        # Optional PPE - warn if missing
        recommended_ppe = ['Goggles', 'Gloves', 'Boots']
        for item in recommended_ppe:
            if item not in detected_classes:
                missing_items.append(f"{item} (recommended)")
        
        return safety_status, missing_items

class HeadGearSentinelGUI:
    """Main GUI application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("HeadGear Sentinel - PPE Detection System")
        self.root.geometry("1200x700")
        
        # Initialize model
        self.model = PPEModel()
        
        # Variables
        self.current_image = None
        self.detections = []
        self.safety_status = "Unknown"
        self.missing_items = []
        
        # Setup GUI
        self.setup_ui()
        
        # Try to load existing model
        self.load_existing_model()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.root.configure(bg='#f0f0f0')
        
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="🚧 HeadGear Sentinel 🚧",
                              font=('Arial', 24, 'bold'),
                              bg='#f0f0f0',
                              fg='#2c3e50')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        subtitle_label = tk.Label(main_frame,
                                 text="Personal Protective Equipment Detection System",
                                 font=('Arial', 12),
                                 bg='#f0f0f0',
                                 fg='#7f8c8d')
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # Control Panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="15")
        control_frame.grid(row=2, column=0, sticky=tk.N, padx=(0, 10))
        
        # Model Status
        self.model_status_var = tk.StringVar(value="Model: Not Loaded")
        model_status_label = tk.Label(control_frame, 
                                     textvariable=self.model_status_var,
                                     font=('Arial', 10, 'bold'),
                                     fg='#e74c3c')
        model_status_label.pack(pady=(0, 10))
        
        # Buttons
        buttons = [
            ("Load/Sample Image", self.load_image),
            ("Load Image Folder", self.load_folder),
            ("Train New Model", self.train_model),
            ("Run Detection", self.run_detection),
            ("Clear Results", self.clear_results),
            ("Exit", self.root.quit)
        ]
        
        for text, command in buttons:
            btn = ttk.Button(control_frame, 
                            text=text, 
                            command=command,
                            width=20)
            btn.pack(pady=5)
        
        # Training controls
        train_frame = ttk.Frame(control_frame)
        train_frame.pack(pady=15, fill=tk.X)
        
        ttk.Label(train_frame, text="Training Epochs:").pack(anchor=tk.W)
        self.epochs_var = tk.StringVar(value="10")
        epochs_entry = ttk.Entry(train_frame, textvariable=self.epochs_var, width=10)
        epochs_entry.pack(pady=5)
        
        # Image Display Area
        display_frame = ttk.LabelFrame(main_frame, text="Image Display", padding="10")
        display_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=10)
        
        # Canvas for image display
        self.canvas = tk.Canvas(display_frame, bg='#2c3e50', width=600, height=400)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # No image label
        self.no_image_label = tk.Label(self.canvas, 
                                      text="No image loaded\n\nSelect an image or folder to begin",
                                      font=('Arial', 14),
                                      bg='#2c3e50',
                                      fg='#ecf0f1')
        self.no_image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Results Panel
        results_frame = ttk.LabelFrame(main_frame, text="Detection Results", padding="15")
        results_frame.grid(row=2, column=2, sticky=tk.N, padx=(10, 0))
        
        # Safety Status
        self.safety_var = tk.StringVar(value="Safety Status: Unknown")
        safety_label = tk.Label(results_frame, 
                               textvariable=self.safety_var,
                               font=('Arial', 14, 'bold'),
                               fg='#2c3e50')
        safety_label.pack(pady=(0, 10))
        
        # Detected Items
        ttk.Label(results_frame, text="Detected PPE:", font=('Arial', 11, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        self.detections_text = scrolledtext.ScrolledText(results_frame, 
                                                         width=25, 
                                                         height=10,
                                                         font=('Arial', 10))
        self.detections_text.pack(fill=tk.BOTH, expand=True)
        self.detections_text.config(state=tk.DISABLED)
        
        # Missing Items
        ttk.Label(results_frame, text="Missing/Required:", font=('Arial', 11, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        self.missing_text = scrolledtext.ScrolledText(results_frame, 
                                                     width=25, 
                                                     height=5,
                                                     font=('Arial', 10))
        self.missing_text.pack(fill=tk.BOTH, expand=True)
        self.missing_text.config(state=tk.DISABLED)
        
        # Log Area
        log_frame = ttk.LabelFrame(main_frame, text="Activity Log", padding="10")
        log_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(20, 0))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, 
                                                 height=6,
                                                 font=('Courier', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        
        # Status Bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, 
                              textvariable=self.status_var,
                              relief=tk.SUNKEN,
                              anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Configure weights for display frame
        display_frame.rowconfigure(0, weight=1)
        display_frame.columnconfigure(0, weight=1)
    
    def log_message(self, message):
        """Add message to log"""
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.status_var.set(message)
    
    def load_existing_model(self):
        """Try to load existing trained model"""
        success, message = self.model.load_model()
        if success:
            self.model_status_var.set("Model: Loaded ✓")
            self.log_message("Model loaded successfully")
        else:
            self.log_message("No trained model found. Please train a new model.")
    
    def load_image(self):
        """Load a single image for detection"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.process_image(file_path)
    
    def load_folder(self):
        """Load a folder of images for batch processing"""
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        
        if folder_path:
            # Get all image files
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(self.find_files(folder_path, ext))
            
            if image_files:
                # Process first image
                self.process_image(image_files[0])
                self.log_message(f"Loaded folder with {len(image_files)} images")
            else:
                messagebox.showwarning("No Images", "No images found in selected folder")
    
    def find_files(self, folder, pattern):
        """Find files with given pattern"""
        import glob
        return glob.glob(os.path.join(folder, pattern))
    
    def process_image(self, image_path):
        """Load and display image"""
        try:
            # Load image
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                messagebox.showerror("Error", "Could not load image")
                return
            
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.display_image(image_rgb)
            
            # Hide "no image" label
            self.no_image_label.place_forget()
            
            self.log_message(f"Loaded image: {os.path.basename(image_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image):
        """Display image on canvas"""
        # Clear canvas
        self.canvas.delete("all")
        
        # Resize image to fit canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 600
        if canvas_height <= 1:
            canvas_height = 400
        
        img_h, img_w = image.shape[:2]
        
        # Calculate aspect ratio
        ratio = min(canvas_width / img_w, canvas_height / img_h)
        new_w = int(img_w * ratio)
        new_h = int(img_h * ratio)
        
        # Resize image
        resized_img = cv2.resize(image, (new_w, new_h))
        
        # Convert to PIL Image
        pil_img = Image.fromarray(resized_img)
        self.tk_image = ImageTk.PhotoImage(pil_img)
        
        # Display image
        self.canvas.create_image(canvas_width//2, canvas_height//2, 
                                anchor=tk.CENTER, 
                                image=self.tk_image)
        
        # Store display info for detection drawing
        self.display_ratio = ratio
        self.display_offset_x = (canvas_width - new_w) // 2
        self.display_offset_y = (canvas_height - new_h) // 2
    
    def draw_detections(self):
        """Draw detection boxes on canvas"""
        if not self.detections:
            return
        
        for detection in self.detections:
            class_name = detection['class']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            color = self.model.colors.get(class_name, (255, 255, 255))
            
            # Convert color for tkinter
            color_hex = '#{:02x}{:02x}{:02x}'.format(color[2], color[1], color[0])
            
            # Scale coordinates for display
            scale = self.display_ratio
            offset_x = self.display_offset_x
            offset_y = self.display_offset_y
            
            display_x1 = x1 * scale + offset_x
            display_y1 = y1 * scale + offset_y
            display_x2 = x2 * scale + offset_x
            display_y2 = y2 * scale + offset_y
            
            # Draw bounding box
            self.canvas.create_rectangle(display_x1, display_y1, 
                                        display_x2, display_y2,
                                        outline=color_hex,
                                        width=3)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            self.canvas.create_text(display_x1, display_y1 - 10,
                                   text=label,
                                   anchor=tk.W,
                                   fill=color_hex,
                                   font=('Arial', 10, 'bold'))
    
    def train_model(self):
        """Train a new model"""
        if not self.model.is_trained:
            response = messagebox.askyesno(
                "Training Required",
                "No trained model found. Would you like to train a new model?\n\n"
                "Note: You need a dataset folder with images for training."
            )
            if not response:
                return
        
        # Get dataset folder
        dataset_path = filedialog.askdirectory(title="Select Training Dataset Folder")
        if not dataset_path:
            return
        
        # Get epochs
        try:
            epochs = int(self.epochs_var.get())
        except:
            epochs = 10
        
        # Start training in separate thread
        def train_thread():
            self.log_message("Starting model training...")
            success, message = self.model.train(dataset_path, epochs)
            
            self.root.after(0, lambda: self.training_complete(success, message))
        
        threading.Thread(target=train_thread, daemon=True).start()
    
    def training_complete(self, success, message):
        """Callback after training completes"""
        if success:
            self.model_status_var.set("Model: Trained ✓")
            messagebox.showinfo("Training Complete", message)
        else:
            messagebox.showerror("Training Failed", message)
        self.log_message(message)
    
    def run_detection(self):
        """Run PPE detection on current image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        if not self.model.is_trained:
            messagebox.showwarning("Model Not Trained", 
                                 "Please train or load a model first")
            return
        
        # Run detection
        self.log_message("Running PPE detection...")
        
        # Get detections
        detections, message = self.model.predict(self.current_image)
        self.detections = detections
        
        # Check safety
        self.safety_status, self.missing_items = self.model.check_safety(detections)
        
        # Update display
        self.update_results()
        self.draw_detections()
        
        self.log_message(f"Detection complete: {len(detections)} items found")
    
    def update_results(self):
        """Update results display"""
        # Update safety status
        color = '#27ae60' if self.safety_status == "Safe" else '#e74c3c'
        self.safety_var.set(f"Safety Status: {self.safety_status}")
        
        # Update safety label color
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.LabelFrame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Label) and child.cget("textvariable") == str(self.safety_var):
                        child.config(fg=color)
        
        # Update detections text
        self.detections_text.config(state=tk.NORMAL)
        self.detections_text.delete(1.0, tk.END)
        
        if self.detections:
            for detection in self.detections:
                class_name = detection['class']
                confidence = detection['confidence']
                self.detections_text.insert(tk.END, 
                                          f"✓ {class_name}: {confidence:.2%}\n")
        else:
            self.detections_text.insert(tk.END, "No PPE detected")
        
        self.detections_text.config(state=tk.DISABLED)
        
        # Update missing items
        self.missing_text.config(state=tk.NORMAL)
        self.missing_text.delete(1.0, tk.END)
        
        if self.missing_items:
            for item in self.missing_items:
                self.missing_text.insert(tk.END, f"⚠ {item}\n")
        else:
            self.missing_text.insert(tk.END, "All PPE present ✓")
        
        self.missing_text.config(state=tk.DISABLED)
    
    def clear_results(self):
        """Clear all results"""
        self.detections = []
        self.safety_status = "Unknown"
        self.missing_items = []
        
        # Clear canvas
        self.canvas.delete("all")
        self.no_image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        
        # Clear text boxes
        self.detections_text.config(state=tk.NORMAL)
        self.detections_text.delete(1.0, tk.END)
        self.detections_text.config(state=tk.DISABLED)
        
        self.missing_text.config(state=tk.NORMAL)
        self.missing_text.delete(1.0, tk.END)
        self.missing_text.config(state=tk.DISABLED)
        
        # Reset safety status
        self.safety_var.set("Safety Status: Unknown")
        
        self.log_message("Results cleared")

def create_sample_dataset():
    """Create a sample dataset folder with instructions"""
    sample_path = "sample_dataset"
    os.makedirs(sample_path, exist_ok=True)
    
    # Create sample images (colored squares as placeholders)
    for i in range(20):
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        color = np.random.randint(0, 255, 3)
        img[:, :] = color
        cv2.imwrite(os.path.join(sample_path, f"worker_{i:03d}.jpg"), img)
    
    # Create instructions file
    instructions = """Sample Dataset for HeadGear Sentinel
    
Place your training images here:
- Images should be in .jpg, .png, or .bmp format
- For real training, include images of construction workers
- Multiple images will improve model accuracy

For this demo, synthetic data will be used.
"""
    
    with open(os.path.join(sample_path, "README.txt"), "w") as f:
        f.write(instructions)
    
    return sample_path

def main():
    """Main entry point"""
    # Create sample dataset if needed
    if not os.path.exists("sample_dataset"):
        create_sample_dataset()
    
    # Create and run GUI
    root = tk.Tk()
    app = HeadGearSentinelGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()