"""
HeadGear Sentinel - Construction Worker PPE Object Detection
Real HOG + SVM Sliding Window Detector with GUI
Semester Project | University of Wah

APPROACH: Classical Computer Vision (HOG) + Lightweight SVM
- NO deep learning, NO GPU required
- Sliding window + Image pyramid for multi-scale detection
- CPU-optimized feature extraction and classification
- Real training from folder structure
- Threading for non-blocking GUI

AUTHOR: CS Student
SYSTEM: Linux Mint / Ubuntu, CPU-only
"""

import os
import cv2
import numpy as np
from pathlib import Path
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import json
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'window_size': (64, 64),          # HOG window size (reasonable for CPU)
    'step_size': 16,                  # Sliding window step (pixels)
    'scale_factor': 1.3,              # Image pyramid scale
    'min_neighbors': 2,               # Min detections for clustering
    'class_names': [
        'Person',
        'Helmet',
        'Goggles',
        'Mask',
        'Gloves',
        'Vest',
        'Boots'
    ],
    'safe_classes': [1, 2, 3, 4, 5, 6],  # Classes 1-6 = PPE (not Person)
}


# ============================================================================
# FEATURE EXTRACTION - HOG
# ============================================================================

class HOGExtractor:
    """Extract HOG features from image patches"""
    
    def __init__(self):
        self.orientations = 9
        self.pixels_per_cell = (8, 8)
        self.cells_per_block = (2, 2)
    
    def extract(self, image):
        """
        Extract HOG descriptor from image patch
        
        Args:
            image: grayscale image patch
            
        Returns:
            1D feature vector
        """
        if image.shape != CONFIG['window_size']:
            image = cv2.resize(image, CONFIG['window_size'])
        
        # Compute HOG
        features = hog(
            image,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            visualize=False,
            channel_axis=None
        )
        
        return features


# ============================================================================
# DETECTOR - SLIDING WINDOW + SVM
# ============================================================================

class SlidingWindowDetector:
    """Sliding window object detector with multi-scale search"""
    
    def __init__(self):
        self.hog_extractor = HOGExtractor()
        self.detections = {}  # class_id -> list of (x, y, w, h, confidence)
    
    def image_pyramid(self, image, scale=1.3, min_size=(64, 64)):
        """
        Generate image pyramid for multi-scale detection
        
        Args:
            image: input image
            scale: scale factor between levels
            min_size: minimum image size
            
        Yields:
            (resized_image, scale_factor)
        """
        h, w = image.shape[:2]
        
        while min(h, w) >= min_size[0]:
            yield image, 1.0 if h == image.shape[0] else (h / image.shape[0])
            
            h = int(h / scale)
            w = int(w / scale)
            
            if min(h, w) < min_size[0]:
                break
                
            image = cv2.resize(image, (w, h))
    
    def sliding_window(self, image, step_size=16, window_size=(64, 64)):
        """
        Slide window across image
        
        Args:
            image: input image
            step_size: pixels between windows
            window_size: (width, height) of window
            
        Yields:
            (x, y, window_patch)
        """
        h, w = image.shape[:2]
        win_h, win_w = window_size
        
        for y in range(0, h - win_h, step_size):
            for x in range(0, w - win_w, step_size):
                window = image[y:y+win_h, x:x+win_w]
                
                if window.shape == window_size:
                    yield (x, y, window)
    
    def detect_single_class(self, image, classifier, scaler, class_name):
        """
        Detect single class in image using sliding window
        
        Args:
            image: input image
            classifier: trained SVM model
            scaler: fitted feature scaler
            class_name: class being detected
            
        Returns:
            list of (x, y, w, h, confidence)
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        detections = []
        
        # Multi-scale sliding window search
        for scaled_img, scale in self.image_pyramid(gray, CONFIG['scale_factor']):
            h, w = scaled_img.shape[:2]
            
            if min(h, w) < CONFIG['window_size'][0]:
                continue
            
            # Slide window across pyramid level
            for (x, y, window) in self.sliding_window(
                scaled_img,
                CONFIG['step_size'],
                CONFIG['window_size']
            ):
                # Extract HOG features
                features = self.hog_extractor.extract(window)
                features = features.reshape(1, -1)
                
                # Normalize
                features = scaler.transform(features)
                
                # Predict
                prediction = classifier.predict(features)[0]
                confidence = classifier.decision_function(features)[0]
                
                # If positive detection
                if prediction == 1:
                    # Scale coordinates back to original image size
                    x_orig = int(x / scale) if scale != 1.0 else x
                    y_orig = int(y / scale) if scale != 1.0 else y
                    w_orig = int(CONFIG['window_size'][0] / scale) if scale != 1.0 else CONFIG['window_size'][0]
                    h_orig = int(CONFIG['window_size'][1] / scale) if scale != 1.0 else CONFIG['window_size'][1]
                    
                    detections.append({
                        'x': x_orig,
                        'y': y_orig,
                        'w': w_orig,
                        'h': h_orig,
                        'confidence': float(confidence)
                    })
        
        return detections
    
    def non_maximum_suppression(self, detections, threshold=0.3):
        """
        Remove overlapping detections
        
        Args:
            detections: list of detection dicts
            threshold: IoU threshold
            
        Returns:
            filtered detections
        """
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        keep = []
        
        for i, det in enumerate(detections):
            if i == 0:
                keep.append(det)
                continue
            
            overlaps = False
            for kept in keep:
                # Compute IoU
                x1_min, x1_max = det['x'], det['x'] + det['w']
                y1_min, y1_max = det['y'], det['y'] + det['h']
                
                x2_min, x2_max = kept['x'], kept['x'] + kept['w']
                y2_min, y2_max = kept['y'], kept['y'] + kept['h']
                
                inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                inter = inter_x * inter_y
                
                union = (det['w'] * det['h']) + (kept['w'] * kept['h']) - inter
                iou = inter / union if union > 0 else 0
                
                if iou > threshold:
                    overlaps = True
                    break
            
            if not overlaps:
                keep.append(det)
        
        return keep


# ============================================================================
# TRAINER - BATCH TRAINING WITH INCREMENTAL UPDATES
# ============================================================================

class ModelTrainer:
    """Train one-vs-rest binary classifiers for each PPE class"""
    
    def __init__(self):
        self.hog_extractor = HOGExtractor()
        self.classifiers = {}      # class_id -> classifier
        self.scalers = {}          # class_id -> scaler
        self.training_stats = {}
    
    def load_dataset(self, dataset_path):
        """
        Load training samples from folder structure
        
        Args:
            dataset_path: path to dataset root
            
        Returns:
            dict: class_id -> list of image paths
        """
        dataset = {}
        dataset_path = Path(dataset_path)
        
        for class_id in range(len(CONFIG['class_names'])):
            class_folder = dataset_path / f'class_{class_id}_{CONFIG["class_names"][class_id].lower()}'
            
            if not class_folder.exists():
                print(f"WARNING: {class_folder} not found")
                dataset[class_id] = []
                continue
            
            # Find all image files
            images = list(class_folder.glob('*.jpg')) + \
                    list(class_folder.glob('*.jpeg')) + \
                    list(class_folder.glob('*.png')) + \
                    list(class_folder.glob('*.JPG')) + \
                    list(class_folder.glob('*.PNG'))
            
            dataset[class_id] = [str(img) for img in images]
            print(f"Class {class_id} ({CONFIG['class_names'][class_id]}): {len(images)} images")
        
        return dataset
    
    def extract_features_batch(self, image_paths, class_label):
        """
        Extract HOG features from batch of images
        
        Args:
            image_paths: list of image file paths
            class_label: 1 (positive) or 0 (negative)
            
        Returns:
            tuple: (features_array, labels_array)
        """
        features_list = []
        labels_list = []
        
        for img_path in image_paths:
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Resize to standard size
                gray = cv2.resize(gray, CONFIG['window_size'])
                
                # Extract HOG
                features = self.hog_extractor.extract(gray)
                
                features_list.append(features)
                labels_list.append(class_label)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        if not features_list:
            return np.array([]), np.array([])
        
        return np.array(features_list), np.array(labels_list)
    
    def train_binary_classifier(self, positive_paths, negative_paths, class_id, callback=None):
        """
        Train binary (one-vs-rest) classifier for one class
        
        Args:
            positive_paths: paths to positive samples (this class)
            negative_paths: paths to negative samples (other classes)
            class_id: class being trained
            callback: callback function for progress (message, progress)
        """
        class_name = CONFIG['class_names'][class_id]
        print(f"\n--- Training Class {class_id}: {class_name} ---")
        
        # Extract positive features
        if callback:
            callback(f"Extracting positive samples for {class_name}...", 10)
        
        pos_features, pos_labels = self.extract_features_batch(positive_paths, 1)
        
        if len(pos_features) == 0:
            print(f"WARNING: No positive samples for class {class_id}")
            return False
        
        print(f"  Positive samples: {len(pos_features)}")
        
        # Extract negative features
        if callback:
            callback(f"Extracting negative samples for {class_name}...", 30)
        
        # Mix of all other classes
        neg_features_list = []
        for other_id in range(len(CONFIG['class_names'])):
            if other_id != class_id:
                other_features, _ = self.extract_features_batch(negative_paths.get(other_id, []), 0)
                if len(other_features) > 0:
                    neg_features_list.append(other_features)
        
        if neg_features_list:
            neg_features = np.vstack(neg_features_list)
        else:
            print(f"WARNING: No negative samples for class {class_id}")
            neg_features = np.array([])
        
        if len(neg_features) > 0:
            print(f"  Negative samples: {len(neg_features)}")
        
        # Combine
        if len(neg_features) > 0:
            X = np.vstack([pos_features, neg_features])
            y = np.hstack([pos_labels, np.zeros(len(neg_features), dtype=int)])
        else:
            X = pos_features
            y = pos_labels
        
        print(f"  Total training samples: {len(X)}")
        
        # Scale features
        if callback:
            callback(f"Scaling features for {class_name}...", 50)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train SVM
        if callback:
            callback(f"Training SVM for {class_name}...", 70)
        
        clf = LinearSVC(
            C=1.0,
            max_iter=2000,
            dual=False,
            random_state=42,
            verbose=0
        )
        clf.fit(X_scaled, y)
        
        # Evaluate on training set
        train_score = clf.score(X_scaled, y)
        print(f"  Training accuracy: {train_score:.2%}")
        
        # Store model and scaler
        self.classifiers[class_id] = clf
        self.scalers[class_id] = scaler
        self.training_stats[class_id] = {
            'positive_samples': len(pos_features),
            'negative_samples': len(neg_features),
            'training_accuracy': float(train_score),
            'timestamp': datetime.now().isoformat()
        }
        
        return True
    
    def train_all_classes(self, dataset_path, callback=None):
        """
        Train classifiers for all classes (one-vs-rest)
        
        Args:
            dataset_path: path to dataset
            callback: progress callback
        """
        print("\n" + "="*60)
        print("HEADGEAR SENTINEL - TRAINING STARTED")
        print("="*60)
        
        # Load dataset
        if callback:
            callback("Loading dataset...", 0)
        
        dataset = self.load_dataset(dataset_path)
        
        total_classes = len(CONFIG['class_names'])
        trained = 0
        
        # Train each class
        for class_id in range(total_classes):
            if not dataset[class_id]:
                print(f"Skipping class {class_id} (no samples)")
                continue
            
            # Progress
            progress = int((class_id / total_classes) * 100)
            if callback:
                callback(f"Training {class_id + 1}/{total_classes}...", progress)
            
            # Get positive samples (this class)
            positive = dataset[class_id]
            
            # Get negative samples (all other classes)
            negative = {}
            for other_id in range(total_classes):
                if other_id != class_id:
                    negative[other_id] = dataset[other_id]
            
            # Train
            success = self.train_binary_classifier(positive, negative, class_id, callback)
            
            if success:
                trained += 1
        
        print(f"\nTraining complete: {trained}/{total_classes} classifiers trained")
        
        if callback:
            callback("Training complete!", 100)
        
        return trained > 0
    
    def save_models(self, model_path):
        """Save all trained models to disk"""
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        for class_id in self.classifiers:
            clf_file = Path(model_path) / f'clf_{class_id}.pkl'
            scaler_file = Path(model_path) / f'scaler_{class_id}.pkl'
            
            joblib.dump(self.classifiers[class_id], clf_file)
            joblib.dump(self.scalers[class_id], scaler_file)
        
        # Save stats
        stats_file = Path(model_path) / 'training_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        print(f"Models saved to {model_path}")
    
    def load_models(self, model_path):
        """Load trained models from disk"""
        model_path = Path(model_path)
        
        for class_id in range(len(CONFIG['class_names'])):
            clf_file = model_path / f'clf_{class_id}.pkl'
            scaler_file = model_path / f'scaler_{class_id}.pkl'
            
            if clf_file.exists() and scaler_file.exists():
                self.classifiers[class_id] = joblib.load(clf_file)
                self.scalers[class_id] = joblib.load(scaler_file)
        
        return len(self.classifiers) > 0


# ============================================================================
# GUI - TKINTER INTERFACE
# ============================================================================

class HeadGearSentinelGUI:
    """Main GUI application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("HeadGear Sentinel - PPE Detection")
        self.root.geometry("1200x900")
        self.root.resizable(False, False)
        
        # Center window on screen
        self.center_window()
        
        # Data
        self.trainer = ModelTrainer()
        self.detector = SlidingWindowDetector()
        self.current_image = None
        self.current_image_path = None
        self.dataset_path = None
        self.model_path = "./models"
        
        self.training_thread = None
        self.is_training = False
        
        # Build UI
        self.build_ui()
    
    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f'{w}x{h}+{x}+{y}')
    
    def build_ui(self):
        """Build GUI components"""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # ===== CONTROL PANEL =====
        control_frame = ttk.LabelFrame(main_frame, text='Control Panel', padding=10)
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Row 1: Dataset
        ttk.Label(control_frame, text='Dataset Path:').grid(row=0, column=0, sticky='w')
        self.dataset_entry = ttk.Entry(control_frame, width=50)
        self.dataset_entry.grid(row=0, column=1, padx=5)
        self.dataset_entry.insert(0, './dataset')
        
        ttk.Button(control_frame, text='Browse...', command=self.browse_dataset).grid(row=0, column=2, padx=5)
        
        # Row 2: Buttons
        ttk.Button(control_frame, text='Load Dataset', command=self.load_dataset).grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        ttk.Button(control_frame, text='Train Model', command=self.start_training).grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        ttk.Button(control_frame, text='Load Trained Model', command=self.load_model).grid(row=1, column=2, sticky='ew', padx=5, pady=5)
        
        # Row 3: Image loading
        ttk.Button(control_frame, text='Load Image', command=self.load_image).grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        ttk.Button(control_frame, text='Run Detection', command=self.run_detection).grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        ttk.Button(control_frame, text='Clear', command=self.clear_results).grid(row=2, column=2, sticky='ew', padx=5, pady=5)
        
        # ===== PROGRESS AREA =====
        progress_frame = ttk.LabelFrame(main_frame, text='Training Progress', padding=10)
        progress_frame.pack(fill='x', pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=5)
        
        self.status_label = ttk.Label(progress_frame, text='Ready', foreground='blue')
        self.status_label.pack(anchor='w')
        
        # ===== MAIN CONTENT =====
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Left: Image display
        left_frame = ttk.LabelFrame(content_frame, text='Image Viewer', padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.image_label = ttk.Label(left_frame, background='lightgray', text='No image loaded')
        self.image_label.pack(fill='both', expand=True)
        
        # Right: Results panel
        right_frame = ttk.LabelFrame(content_frame, text='Detection Results', padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Safety status
        self.safety_label = ttk.Label(right_frame, text='Status: -', font=('Arial', 12, 'bold'), foreground='gray')
        self.safety_label.pack(pady=5)
        
        # Detected objects
        ttk.Label(right_frame, text='Detected Objects:').pack(anchor='w', pady=(10, 5))
        
        self.results_frame = ttk.Frame(right_frame)
        self.results_frame.pack(fill='both', expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.results_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.results_text = tk.Text(
            self.results_frame,
            height=20,
            width=40,
            yscrollcommand=scrollbar.set,
            font=('Courier', 9)
        )
        self.results_text.pack(side='left', fill='both', expand=True)
        scrollbar.config(command=self.results_text.yview)
        
        # ===== LOG AREA =====
        log_frame = ttk.LabelFrame(main_frame, text='System Log', padding=10, height=100)
        log_frame.pack(fill='both', expand=False)
        
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side='right', fill='y')
        
        self.log_text = tk.Text(log_frame, height=6, yscrollcommand=scrollbar.set, font=('Courier', 8))
        self.log_text.pack(fill='both', expand=True)
        scrollbar.config(command=self.log_text.yview)
        
        self.log("HeadGear Sentinel initialized")
    
    def log(self, message):
        """Add message to log safely"""
        # If root or log_text is gone, just print to console and return
        try:
            if not hasattr(self, "log_text") or self.log_text is None:
                print(f"[LOG] {message}")
                return
            if not self.root.winfo_exists():
                print(f"[LOG] {message}")
                return

            self.log_text.insert('end', f"[{datetime.now().strftime('%H:%M:%S')}] {message}\n")
            self.log_text.see('end')
            self.root.update_idletasks()
        except tk.TclError:
            # Widget destroyed or mainloop ended – avoid crash
            print(f"[LOG-TCL-ERROR] {message}")

    
    def update_progress(self, message, progress):
        """Update training progress"""
        self.status_label.config(text=message)
        self.progress_var.set(progress)
        self.root.update()
    
    def browse_dataset(self):
        """Browse for dataset folder"""
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            self.dataset_entry.delete(0, 'end')
            self.dataset_entry.insert(0, folder)
    
    def load_dataset(self):
        """Load and verify dataset"""
        dataset_path = self.dataset_entry.get()
        
        if not Path(dataset_path).exists():
            messagebox.showerror("Error", f"Dataset path not found: {dataset_path}")
            return
        
        dataset = self.trainer.load_dataset(dataset_path)
        total = sum(len(paths) for paths in dataset.values())
        
        self.log(f"Dataset loaded: {total} total images")
        self.dataset_path = dataset_path
    
    def start_training(self):
        """Start training in background thread"""
        if not self.dataset_path:
            messagebox.showwarning("Warning", "Please load dataset first")
            return
        
        if self.is_training:
            messagebox.showwarning("Warning", "Training already in progress")
            return
        
        self.is_training = True
        
        # Start in background thread
        self.training_thread = threading.Thread(
            target=self._train_worker,
            daemon=True
        )
        self.training_thread.start()
    
    def _train_worker(self):
        """Training worker (runs in thread)"""
        try:
            self.log("Training started...")
            
            success = self.trainer.train_all_classes(
                self.dataset_path,
                callback=self.update_progress
            )
            
            if success:
                # Save models
                self.trainer.save_models(self.model_path)
                self.log(f"Training complete! Models saved to {self.model_path}")
                self.update_progress("Training complete!", 100)
                messagebox.showinfo("Success", "Model training complete!")
            else:
                self.log("Training failed")
                messagebox.showerror("Error", "Training failed")
        
        except Exception as e:
            self.log(f"Training error: {e}")
            messagebox.showerror("Error", f"Training error: {e}")
        
        finally:
            self.is_training = False
    
    def load_model(self):
        """Load trained model"""
        if not Path(self.model_path).exists():
            messagebox.showerror("Error", f"Model path not found: {self.model_path}")
            return
        
        if self.trainer.load_models(self.model_path):
            self.log(f"Loaded {len(self.trainer.classifiers)} class models")
            messagebox.showinfo("Success", f"Loaded {len(self.trainer.classifiers)} class models")
        else:
            messagebox.showerror("Error", "No trained models found")
    
    def load_image(self):
        """Load image for detection"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self.current_image = cv2.imread(file_path)
        self.current_image_path = file_path
        
        if self.current_image is None:
            messagebox.showerror("Error", "Failed to load image")
            return
        
        # Display image
        self.display_image(self.current_image)
        self.log(f"Image loaded: {Path(file_path).name}")
        self.clear_results_text()
    
    def display_image(self, image):
        """Display image in GUI"""
        h, w = image.shape[:2]
        
        # Resize for display (max 400x500)
        max_w, max_h = 400, 500
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            w = int(w * scale)
            h = int(h * scale)
            image = cv2.resize(image, (w, h))
        
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Display
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=image_tk, text='')
        self.image_label.image = image_tk
    
    def run_detection(self):
        """Run object detection"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if not self.trainer.classifiers:
            messagebox.showwarning("Warning", "Please train or load a model first")
            return
        
        self.log("Running detection...")
        self.clear_results_text()
        
        try:
            detections_by_class = {}
            all_detections = []
            
            # Detect each class
            for class_id in range(len(CONFIG['class_names'])):
                if class_id not in self.trainer.classifiers:
                    continue
                
                class_name = CONFIG['class_names'][class_id]
                clf = self.trainer.classifiers[class_id]
                scaler = self.trainer.scalers[class_id]
                
                self.log(f"Detecting {class_name}...")
                
                # Detect
                dets = self.detector.detect_single_class(
                    self.current_image,
                    clf,
                    scaler,
                    class_name
                )
                
                # Filter with NMS
                dets = self.detector.non_maximum_suppression(dets, threshold=0.3)
                
                detections_by_class[class_id] = dets
                
                # Add class info
                for det in dets:
                    det['class_id'] = class_id
                    det['class_name'] = class_name
                    all_detections.append(det)
            
            # Display results
            self.display_detection_results(detections_by_class, all_detections)
            
            # Draw on image
            result_image = self.current_image.copy()
            for det in all_detections:
                x, y, w, h = det['x'], det['y'], det['w'], det['h']
                cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Label
                label = f"{det['class_name']} ({det['confidence']:.2f})"
                cv2.putText(result_image, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display result
            self.display_image(result_image)
            
            self.log("Detection complete")
            
        except Exception as e:
            self.log(f"Detection error: {e}")
            messagebox.showerror("Error", f"Detection error: {e}")
    
    def display_detection_results(self, by_class, all_dets):
        """Display detection results in text widget"""
        # Determine safety
        is_safe = all(
            len(by_class.get(c, [])) > 0
            for c in CONFIG['safe_classes']
        )
        
        if len(all_dets) == 0:
            is_safe = False
        
        # Safety status
        status_text = "SAFE" if is_safe else "UNSAFE"
        status_color = 'green' if is_safe else 'red'
        
        self.safety_label.config(
            text=f"Status: {status_text}",
            foreground=status_color
        )
        
        # Results text
        results = "=" * 45 + "\n"
        results += f"SAFETY STATUS: {status_text}\n"
        results += "=" * 45 + "\n\n"
        
        if len(all_dets) == 0:
            results += "No objects detected\n"
        else:
            results += f"Total Detections: {len(all_dets)}\n\n"
            
            for det in all_dets:
                results += f"Class: {det['class_name']}\n"
                results += f"  Position: ({det['x']}, {det['y']})\n"
                results += f"  Size: {det['w']}x{det['h']}\n"
                results += f"  Confidence: {det['confidence']:.3f}\n\n"
        
        # Required PPE check
        results += "\n" + "=" * 45 + "\n"
        results += "PPE REQUIREMENTS:\n"
        results += "=" * 45 + "\n"
        
        for class_id in CONFIG['safe_classes']:
            if class_id >= len(CONFIG['class_names']):
                continue
            
            class_name = CONFIG['class_names'][class_id]
            count = len(by_class.get(class_id, []))
            status = "✓ YES" if count > 0 else "✗ NO"
            
            results += f"{class_name:12s}: {status}\n"
        
        self.results_text.insert('end', results)
    
    def clear_results(self):
        """Clear all results"""
        self.current_image = None
        self.current_image_path = None
        self.image_label.config(image='', text='No image loaded')
        self.image_label.image = None
        self.clear_results_text()
        self.safety_label.config(text='Status: -', foreground='gray')
    
    def clear_results_text(self):
        """Clear results text widget"""
        self.results_text.delete('1.0', 'end')
        self.log_text.delete('1.0', 'end')


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Launch application"""
    root = tk.Tk()
    app = HeadGearSentinelGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
