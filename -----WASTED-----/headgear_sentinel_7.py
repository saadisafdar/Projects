"""
HeadGear Sentinel - Construction Worker PPE Object Detection
Complete Semester Project in Single Python File
Author: [Your Name]
Date: [Current Date]
"""

import os
import sys
import cv2
import numpy as np
import pickle
import json
import time
import threading
import random
from datetime import datetime
from pathlib import Path
import math

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk

# Lightweight ML imports
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# OBJECT DETECTION MODEL CLASS
# ============================================================================

class PPEDetector:
    """
    Lightweight Object Detector for PPE items
    Uses sliding window + HOG features + SVM (CPU-friendly)
    """
    
    def __init__(self):
        # 7 classes as specified
        self.classes = ['Person', 'Helmet', 'Goggles', 'Mask', 'Gloves', 'Vest', 'Boots']
        self.class_colors = {
            'Person': (0, 255, 0),      # Green
            'Helmet': (255, 0, 0),      # Blue
            'Goggles': (0, 0, 255),     # Red
            'Mask': (255, 255, 0),      # Cyan
            'Gloves': (255, 0, 255),    # Magenta
            'Vest': (0, 165, 255),      # Orange
            'Boots': (128, 0, 128)      # Purple
        }
        
        # Detection parameters
        self.window_size = (64, 128)  # Detection window size
        self.step_size = 16           # Sliding window step
        self.scales = [0.5, 0.75, 1.0, 1.25, 1.5]  # Multi-scale
        self.confidence_threshold = 0.6
        self.nms_threshold = 0.3
        
        # Model components
        self.hog = None
        self.classifiers = {}
        self.scalers = {}
        self.is_trained = False
        
        # Initialize HOG descriptor
        self.init_hog()
        
    def init_hog(self):
        """Initialize HOG descriptor for feature extraction"""
        winSize = self.window_size
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        self.hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    
    def extract_features(self, image_patch):
        """Extract HOG features from image patch"""
        if image_patch.shape[:2] != self.window_size:
            image_patch = cv2.resize(image_patch, self.window_size)
        
        # Get HOG features
        features = self.hog.compute(image_patch)
        if features is not None:
            return features.flatten()
        return np.array([])
    
    def extract_color_features(self, image_patch):
        """Extract color histogram features"""
        patch = cv2.resize(image_patch, (32, 32))
        features = []
        
        for i in range(3):  # BGR channels
            hist = cv2.calcHist([patch], [i], None, [8], [0, 256])
            cv2.normalize(hist, hist)
            features.extend(hist.flatten())
        
        return np.array(features)
    
    def get_synthetic_labels(self, image_path, class_name):
        """Generate synthetic labels based on image analysis"""
        img = cv2.imread(str(image_path))
        if img is None:
            return 0
        
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        if class_name == 'Person':
            # Person is usually present in construction images
            return 0.8 + random.uniform(-0.1, 0.1)
        
        elif class_name == 'Helmet':
            # Check for bright region at top
            top_region = img[0:h//4, :]
            brightness = np.mean(top_region) / 255
            return min(0.5 + brightness, 0.9)
        
        elif class_name == 'Vest':
            # Check for orange/yellow colors
            lower_orange = np.array([10, 100, 100])
            upper_orange = np.array([25, 255, 255])
            mask = cv2.inRange(hsv, lower_orange, upper_orange)
            ratio = np.sum(mask > 0) / (h * w)
            return min(0.3 + ratio * 2, 0.8)
        
        elif class_name == 'Gloves':
            # Check hand regions
            return 0.4 + random.uniform(-0.1, 0.1)
        
        elif class_name == 'Boots':
            # Check bottom region
            bottom_region = img[3*h//4:h, :]
            darkness = 1.0 - (np.mean(bottom_region) / 255)
            return min(0.3 + darkness * 0.4, 0.7)
        
        else:  # Goggles, Mask
            return 0.3 + random.uniform(-0.1, 0.1)
    
    def prepare_training_data(self, dataset_path, progress_callback=None):
        """Prepare training data from folder structure"""
        X = []
        y = []
        
        total_images = 0
        processed = 0
        
        # Count total images
        for class_idx, class_name in enumerate(self.classes):
            folder = f"class_{class_idx}_{class_name.lower()}"
            folder_path = os.path.join(dataset_path, folder)
            
            if os.path.exists(folder_path):
                images = list(Path(folder_path).glob("*.[pj][np]*[gG]"))
                total_images += min(len(images), 20)  # Limit per class
        
        # Process each class
        for class_idx, class_name in enumerate(self.classes):
            folder = f"class_{class_idx}_{class_name.lower()}"
            folder_path = os.path.join(dataset_path, folder)
            
            if not os.path.exists(folder_path):
                print(f"Warning: {folder} not found")
                continue
            
            images = list(Path(folder_path).glob("*.[pj][np]*[gG]"))[:20]
            
            for img_path in images:
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    h, w = img.shape[:2]
                    
                    # Extract positive samples (object present)
                    for _ in range(5):  # 5 positive samples per image
                        if h > self.window_size[1] and w > self.window_size[0]:
                            y_start = random.randint(0, h - self.window_size[1])
                            x_start = random.randint(0, w - self.window_size[0])
                            
                            patch = img[y_start:y_start+self.window_size[1],
                                       x_start:x_start+self.window_size[0]]
                            
                            if patch.shape[:2] == (self.window_size[1], self.window_size[0]):
                                # Combine HOG and color features
                                hog_features = self.extract_features(patch)
                                color_features = self.extract_color_features(patch)
                                features = np.concatenate([hog_features, color_features])
                                
                                X.append(features)
                                # Create one-hot label
                                label = np.zeros(len(self.classes))
                                label[class_idx] = 1
                                y.append(label)
                    
                    # Extract negative samples (background)
                    for _ in range(5):  # 5 negative samples per image
                        if h > self.window_size[1] and w > self.window_size[0]:
                            y_start = random.randint(0, h - self.window_size[1])
                            x_start = random.randint(0, w - self.window_size[0])
                            
                            patch = img[y_start:y_start+self.window_size[1],
                                       x_start:x_start+self.window_size[0]]
                            
                            if patch.shape[:2] == (self.window_size[1], self.window_size[0]):
                                hog_features = self.extract_features(patch)
                                color_features = self.extract_color_features(patch)
                                features = np.concatenate([hog_features, color_features])
                                
                                X.append(features)
                                y.append(np.zeros(len(self.classes)))
                    
                    processed += 1
                    if progress_callback:
                        progress = (processed / total_images) * 50
                        progress_callback(progress, 
                                        f"Processing {class_name} ({processed}/{total_images})")
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        if len(X) == 0:
            return None, None
        
        return np.array(X), np.array(y)
    
    def train(self, dataset_path, progress_callback=None):
        """Train object detection model"""
        try:
            start_time = time.time()
            
            if progress_callback:
                progress_callback(0, "Loading dataset...")
            
            # Prepare training data
            X, y = self.prepare_training_data(dataset_path, progress_callback)
            
            if X is None:
                return False, "No training data found"
            
            if progress_callback:
                progress_callback(50, f"Training on {len(X)} samples...")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train one classifier per class
            trained_classes = 0
            for class_idx, class_name in enumerate(self.classes):
                # Get binary labels for this class
                y_binary = y[:, class_idx]
                
                if np.sum(y_binary) > 0:  # Has positive samples
                    # Train SVM classifier
                    classifier = LinearSVC(
                        C=1.0,
                        max_iter=1000,
                        random_state=42,
                        verbose=0
                    )
                    
                    classifier.fit(X_scaled, y_binary)
                    self.classifiers[class_name] = classifier
                    self.scalers[class_name] = scaler
                    trained_classes += 1
                
                if progress_callback:
                    progress = 50 + ((class_idx + 1) / len(self.classes)) * 45
                    progress_callback(progress, f"Training {class_name}...")
            
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            training_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(100, "Training complete!")
            
            return True, f"Trained {trained_classes} classifiers in {training_time:.1f}s"
            
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def sliding_window(self, image, scale=1.0):
        """Generate sliding windows over image"""
        h, w = image.shape[:2]
        window_h, window_w = self.window_size
        
        # Scale image
        new_w = int(w * scale)
        new_h = int(h * scale)
        if new_w < window_w or new_h < window_h:
            return []
        
        scaled_img = cv2.resize(image, (new_w, new_h))
        windows = []
        
        for y in range(0, new_h - window_h + 1, self.step_size):
            for x in range(0, new_w - window_w + 1, self.step_size):
                windows.append((x, y, window_w, window_h))
        
        return windows, scaled_img
    
    def detect(self, image, progress_callback=None):
        """Detect objects using sliding window"""
        if not self.is_trained:
            return [], "Model not trained"
        
        try:
            detections = []
            original_h, original_w = image.shape[:2]
            
            # Multi-scale detection
            total_scales = len(self.scales)
            
            for scale_idx, scale in enumerate(self.scales):
                if progress_callback:
                    progress = (scale_idx / total_scales) * 90
                    progress_callback(progress, f"Scale {scale:.2f}x")
                
                # Get windows for this scale
                windows, scaled_img = self.sliding_window(image, scale)
                
                for x, y, w, h in windows:
                    # Extract patch
                    patch = scaled_img[y:y+h, x:x+w]
                    
                    if patch.shape[:2] != (h, w):
                        continue
                    
                    # Extract features
                    hog_features = self.extract_features(patch)
                    color_features = self.extract_color_features(patch)
                    features = np.concatenate([hog_features, color_features]).reshape(1, -1)
                    
                    # Check each classifier
                    for class_name, classifier in self.classifiers.items():
                        if class_name in self.scalers:
                            scaler = self.scalers[class_name]
                            features_scaled = scaler.transform(features)
                            
                            # Get confidence
                            if hasattr(classifier, 'decision_function'):
                                confidence = classifier.decision_function(features_scaled)[0]
                                confidence = 1 / (1 + np.exp(-confidence))  # Sigmoid
                            else:
                                continue
                            
                            if confidence > self.confidence_threshold:
                                # Scale back to original coordinates
                                x_orig = int(x / scale)
                                y_orig = int(y / scale)
                                w_orig = int(w / scale)
                                h_orig = int(h / scale)
                                
                                detections.append({
                                    'class': class_name,
                                    'confidence': float(confidence),
                                    'bbox': (x_orig, y_orig, x_orig + w_orig, y_orig + h_orig)
                                })
            
            # Apply non-maximum suppression
            detections = self.non_max_suppression(detections)
            
            # Ensure all classes are represented
            all_detections = self.ensure_all_classes(detections)
            
            if progress_callback:
                progress_callback(100, f"Found {len(detections)} objects")
            
            return all_detections, "Detection complete"
            
        except Exception as e:
            return [], f"Detection error: {str(e)}"
    
    def non_max_suppression(self, detections):
        """Apply non-maximum suppression to remove overlapping boxes"""
        if len(detections) == 0:
            return detections
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        suppressed = []
        while detections:
            current = detections.pop(0)
            suppressed.append(current)
            
            # Remove overlapping detections of same class
            detections = [
                d for d in detections 
                if not (d['class'] == current['class'] and self.iou(d['bbox'], current['bbox']) > self.nms_threshold)
            ]
        
        return suppressed
    
    def iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def ensure_all_classes(self, detections):
        """Ensure we have detection result for each class"""
        detected_classes = {d['class'] for d in detections}
        all_detections = []
        
        # Add actual detections
        all_detections.extend(detections)
        
        # Add "not detected" for missing classes
        for class_name in self.classes:
            if class_name not in detected_classes:
                all_detections.append({
                    'class': class_name,
                    'confidence': 0.0,
                    'bbox': None,
                    'present': False
                })
            else:
                # Mark as present
                for d in all_detections:
                    if d['class'] == class_name:
                        d['present'] = True
        
        return all_detections
    
    def check_safety(self, detections):
        """Check if worker is safe based on PPE detection"""
        present_classes = [d['class'] for d in detections if d['present']]
        person_detected = 'Person' in present_classes
        
        if not person_detected:
            return "NO PERSON", ["Person not detected"], []
        
        # Critical PPE
        critical = ['Helmet', 'Vest']
        missing_critical = [item for item in critical if item not in present_classes]
        
        if missing_critical:
            status = "UNSAFE"
            missing = [f"Missing: {item}" for item in missing_critical]
        else:
            status = "SAFE"
            missing = []
        
        # Warnings for other PPE
        warnings = []
        other_ppe = ['Gloves', 'Boots', 'Goggles', 'Mask']
        for item in other_ppe:
            if item not in present_classes:
                warnings.append(f"Not wearing: {item}")
        
        return status, missing, warnings
    
    def save_model(self):
        """Save trained model to file"""
        model_data = {
            'classes': self.classes,
            'classifiers': self.classifiers,
            'scalers': self.scalers,
            'is_trained': self.is_trained,
            'window_size': self.window_size,
            'params': {
                'step_size': self.step_size,
                'scales': self.scales,
                'confidence_threshold': self.confidence_threshold
            }
        }
        
        with open('ppe_detector.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        print("Model saved to ppe_detector.pkl")
    
    def load_model(self):
        """Load trained model from file"""
        try:
            if os.path.exists('ppe_detector.pkl'):
                with open('ppe_detector.pkl', 'rb') as f:
                    data = pickle.load(f)
                
                self.classes = data['classes']
                self.classifiers = data.get('classifiers', {})
                self.scalers = data.get('scalers', {})
                self.is_trained = data.get('is_trained', False)
                
                if 'window_size' in data:
                    self.window_size = data['window_size']
                    self.init_hog()
                
                print("Model loaded successfully")
                return True, "Model loaded"
            return False, "No model file found"
        except Exception as e:
            return False, f"Error loading: {str(e)}"

# ============================================================================
# MAIN APPLICATION GUI
# ============================================================================

class HeadGearApp:
    """Main GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("HeadGear Sentinel - PPE Detection")
        self.root.geometry("1200x700")
        self.root.configure(bg='#f0f0f0')
        
        # Center window
        self.center_window()
        
        # Initialize detector
        self.detector = PPEDetector()
        
        # Current state
        self.current_image = None
        self.image_path = None
        self.detections = []
        
        # Thread control
        self.training = False
        self.detecting = False
        
        # Setup GUI
        self.setup_gui()
        
        # Load existing model
        self.load_model()
    
    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")
    
    def setup_gui(self):
        """Setup the complete GUI interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Header
        self.create_header(main_frame)
        
        # Content area
        content_frame = tk.Frame(main_frame, bg='#f0f0f0')
        content_frame.pack(fill='both', expand=True, pady=10)
        
        # Left panel - Controls
        self.create_control_panel(content_frame)
        
        # Center panel - Image
        self.create_image_panel(content_frame)
        
        # Right panel - Results
        self.create_results_panel(content_frame)
        
        # Status bar
        self.create_status_bar(main_frame)
    
    def create_header(self, parent):
        """Create application header"""
        header = tk.Frame(parent, bg='#2c3e50', height=80)
        header.pack(fill='x', pady=(0, 10))
        
        title = tk.Label(
            header,
            text="🚧 HEADGEAR SENTINEL",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white',
            pady=20
        )
        title.pack(side='left', padx=30)
        
        subtitle = tk.Label(
            header,
            text="Construction Worker PPE Detection System",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle.pack(side='left', padx=10)
        
        # Model status
        self.model_status = tk.StringVar(value="Model: Not Loaded")
        status_label = tk.Label(
            header,
            textvariable=self.model_status,
            font=('Arial', 11, 'bold'),
            bg='#2c3e50',
            fg='#e74c3c'
        )
        status_label.pack(side='right', padx=30)
    
    def create_control_panel(self, parent):
        """Create left control panel"""
        control_frame = tk.LabelFrame(
            parent,
            text="CONTROL PANEL",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            relief='raised',
            borderwidth=2
        )
        control_frame.pack(side='left', fill='y', padx=(0, 10))
        
        # Buttons
        btn_frame = tk.Frame(control_frame, bg='white', padx=20, pady=20)
        btn_frame.pack()
        
        buttons = [
            ("📂 Load Dataset", self.load_dataset),
            ("⚙️ Train Model", self.train_model),
            ("🖼️ Load Image", self.load_image),
            ("🔍 Run Detection", self.run_detection),
            ("🗑️ Clear", self.clear_all),
            ("ℹ️ Info", self.show_info),
            ("🚪 Exit", self.root.quit)
        ]
        
        for text, command in buttons:
            btn = tk.Button(
                btn_frame,
                text=text,
                command=command,
                font=('Arial', 10, 'bold'),
                bg='#3498db',
                fg='white',
                activebackground='#2980b9',
                activeforeground='white',
                relief='flat',
                padx=20,
                pady=12,
                cursor='hand2',
                width=15
            )
            btn.pack(pady=8)
            setattr(self, f'btn_{text.split()[1].lower()}', btn)
        
        # Progress frame
        self.progress_frame = tk.Frame(control_frame, bg='white')
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            length=180,
            mode='determinate'
        )
        self.progress_bar.pack(pady=(10, 5))
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="Ready",
            bg='white',
            font=('Arial', 9),
            fg='#7f8c8d'
        )
        self.progress_label.pack()
    
    def create_image_panel(self, parent):
        """Create center image display panel"""
        image_frame = tk.LabelFrame(
            parent,
            text="IMAGE PREVIEW",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            relief='raised',
            borderwidth=2
        )
        image_frame.pack(side='left', fill='both', expand=True, padx=10)
        
        # Canvas for image
        self.canvas = tk.Canvas(
            image_frame,
            bg='#ecf0f1',
            highlightthickness=0
        )
        self.canvas.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Placeholder
        self.canvas.create_text(
            400, 250,
            text="No Image Loaded\n\nLoad an image to begin detection",
            font=('Arial', 14),
            fill='#7f8c8d',
            justify='center'
        )
        
        # Image info
        self.image_info = tk.StringVar(value="")
        info_label = tk.Label(
            image_frame,
            textvariable=self.image_info,
            font=('Arial', 10),
            bg='white',
            fg='#2c3e50'
        )
        info_label.pack(fill='x', padx=5, pady=(0, 5))
    
    def create_results_panel(self, parent):
        """Create right results panel"""
        results_frame = tk.LabelFrame(
            parent,
            text="DETECTION RESULTS",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            relief='raised',
            borderwidth=2
        )
        results_frame.pack(side='right', fill='y', padx=(10, 0))
        
        # Safety status
        safety_card = tk.Frame(
            results_frame,
            bg='#f8f9fa',
            relief='sunken',
            borderwidth=1,
            padx=20,
            pady=20
        )
        safety_card.pack(fill='x', padx=10, pady=(10, 15))
        
        tk.Label(
            safety_card,
            text="SAFETY STATUS",
            font=('Arial', 11, 'bold'),
            bg='#f8f9fa',
            fg='#2c3e50'
        ).pack()
        
        self.safety_var = tk.StringVar(value="UNKNOWN")
        self.safety_label = tk.Label(
            safety_card,
            textvariable=self.safety_var,
            font=('Arial', 16, 'bold'),
            bg='#f8f9fa',
            fg='#f39c12'
        )
        self.safety_label.pack(pady=5)
        
        # Results notebook
        notebook = ttk.Notebook(results_frame)
        notebook.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Tab 1: Detections
        det_frame = tk.Frame(notebook, bg='white')
        notebook.add(det_frame, text="Detected Objects")
        
        self.det_text = scrolledtext.ScrolledText(
            det_frame,
            width=30,
            height=12,
            font=('Courier New', 10),
            bg='#f8f9fa',
            relief='flat'
        )
        self.det_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 2: Alerts
        alert_frame = tk.Frame(notebook, bg='white')
        notebook.add(alert_frame, text="Safety Alerts")
        
        self.alert_text = scrolledtext.ScrolledText(
            alert_frame,
            width=30,
            height=12,
            font=('Courier New', 10),
            bg='#f8f9fa',
            fg='#e74c3c',
            relief='flat'
        )
        self.alert_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Statistics
        self.stats_var = tk.StringVar(value="Objects: 0/7")
        stats_label = tk.Label(
            results_frame,
            textvariable=self.stats_var,
            font=('Arial', 10),
            bg='white',
            fg='#7f8c8d'
        )
        stats_label.pack(pady=(0, 10))
    
    def create_status_bar(self, parent):
        """Create bottom status bar"""
        status_bar = tk.Frame(parent, bg='#34495e', height=30)
        status_bar.pack(fill='x', pady=(10, 0))
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(
            status_bar,
            textvariable=self.status_var,
            font=('Arial', 10),
            bg='#34495e',
            fg='white',
            anchor='w'
        )
        status_label.pack(side='left', padx=15)
        
        # Time
        self.time_var = tk.StringVar(value=datetime.now().strftime("%H:%M:%S"))
        time_label = tk.Label(
            status_bar,
            textvariable=self.time_var,
            font=('Arial', 10),
            bg='#34495e',
            fg='#bdc3c7'
        )
        time_label.pack(side='right', padx=15)
        
        # Update time
        self.update_time()
    
    def update_time(self):
        """Update time display"""
        self.time_var.set(datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self.update_time)
    
    # ========================================================================
    # APPLICATION LOGIC
    # ========================================================================
    
    def load_model(self):
        """Load existing trained model"""
        success, msg = self.detector.load_model()
        if success:
            self.model_status.set(f"✅ Model: Loaded")
            self.update_status("Model loaded successfully")
        else:
            self.update_status("No trained model found")
    
    def load_dataset(self):
        """Load dataset folder"""
        path = filedialog.askdirectory(title="Select Dataset Folder")
        if path:
            # Verify structure
            valid = True
            for i, cls in enumerate(self.detector.classes):
                folder = f"class_{i}_{cls.lower()}"
                if not os.path.exists(os.path.join(path, folder)):
                    valid = False
                    break
            
            if valid:
                self.update_status(f"Dataset loaded: {path}")
                messagebox.showinfo("Success", "Dataset structure verified!")
            else:
                messagebox.showerror("Error", "Invalid dataset structure!")
    
    def train_model(self):
        """Start model training"""
        if self.training:
            return
        
        path = filedialog.askdirectory(title="Select Dataset for Training")
        if not path:
            return
        
        # Verify
        valid = True
        for i, cls in enumerate(self.detector.classes):
            folder = f"class_{i}_{cls.lower()}"
            if not os.path.exists(os.path.join(path, folder)):
                valid = False
                break
        
        if not valid:
            messagebox.showerror("Error", "Invalid dataset structure!")
            return
        
        # Show progress
        self.progress_frame.pack(pady=10)
        self.progress_var.set(0)
        self.progress_label.config(text="Starting training...")
        
        # Disable buttons
        self.set_buttons_state('disabled')
        self.training = True
        
        # Start thread
        thread = threading.Thread(target=self.training_thread, args=(path,))
        thread.daemon = True
        thread.start()
    
    def training_thread(self, path):
        """Training in background thread"""
        def callback(progress, msg):
            self.root.after(0, self.update_training_progress, progress, msg)
        
        success, message = self.detector.train(path, callback)
        self.root.after(0, self.training_complete, success, message)
    
    def update_training_progress(self, progress, message):
        """Update training progress"""
        self.progress_var.set(progress)
        self.progress_label.config(text=message)
        self.update_status(f"Training: {message}")
    
    def training_complete(self, success, message):
        """Handle training completion"""
        self.progress_frame.pack_forget()
        self.set_buttons_state('normal')
        self.training = False
        
        if success:
            self.model_status.set(f"✅ Model: Trained")
            messagebox.showinfo("Success", message)
            self.update_status("Training completed")
        else:
            messagebox.showerror("Error", message)
            self.update_status("Training failed")
    
    def load_image(self):
        """Load image for detection"""
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if path:
            self.process_image(path)
    
    def process_image(self, path):
        """Load and display image"""
        try:
            self.current_image = cv2.imread(path)
            if self.current_image is None:
                messagebox.showerror("Error", "Failed to load image")
                return
            
            self.image_path = path
            
            # Convert for display
            rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.display_image(rgb)
            
            # Update info
            h, w = self.current_image.shape[:2]
            name = os.path.basename(path)
            self.image_info.set(f"{name} | {w}x{h}")
            
            self.update_status(f"Loaded: {name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
    
    def display_image(self, image):
        """Display image on canvas"""
        self.canvas.delete("all")
        
        # Convert to PIL
        pil_img = Image.fromarray(image)
        
        # Get canvas size
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        
        if cw < 10:
            cw = 600
        if ch < 10:
            ch = 400
        
        # Calculate scale
        iw, ih = pil_img.size
        scale = min(cw / iw, ch / ih) * 0.9
        
        nw = int(iw * scale)
        nh = int(ih * scale)
        
        # Resize
        pil_img = pil_img.resize((nw, nh), Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        
        # Display
        x = (cw - nw) // 2
        y = (ch - nh) // 2
        self.canvas.create_image(x, y, anchor='nw', image=self.tk_img)
        
        # Store for drawing
        self.display_scale = scale
        self.display_offset = (x, y)
    
    def run_detection(self):
        """Run object detection"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Load an image first")
            return
        
        if not self.detector.is_trained:
            if messagebox.askyesno("Model Not Trained", "Train model first?"):
                self.train_model()
                return
        
        # Show progress
        self.progress_frame.pack(pady=10)
        self.progress_var.set(0)
        self.progress_label.config(text="Starting detection...")
        
        # Disable buttons
        self.set_buttons_state('disabled')
        self.detecting = True
        
        # Start thread
        thread = threading.Thread(target=self.detection_thread)
        thread.daemon = True
        thread.start()
    
    def detection_thread(self):
        """Detection in background thread"""
        def callback(progress, msg):
            self.root.after(0, self.update_detection_progress, progress, msg)
        
        self.detections, message = self.detector.detect(self.current_image, callback)
        self.root.after(0, self.detection_complete, message)
    
    def update_detection_progress(self, progress, message):
        """Update detection progress"""
        self.progress_var.set(progress)
        self.progress_label.config(text=message)
        self.update_status(f"Detection: {message}")
    
    def detection_complete(self, message):
        """Handle detection completion"""
        self.progress_frame.pack_forget()
        self.set_buttons_state('normal')
        self.detecting = False
        
        # Evaluate safety
        status, missing, warnings = self.detector.check_safety(self.detections)
        
        # Update display
        self.update_results(status, missing, warnings)
        self.draw_detections()
        
        self.update_status(f"Detection: {message}")
    
    def draw_detections(self):
        """Draw bounding boxes on image"""
        for det in self.detections:
            if not det['present'] or det['bbox'] is None:
                continue
            
            cls = det['class']
            conf = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            color = self.detector.class_colors.get(cls, (255, 255, 255))
            
            # Scale coordinates
            scale = self.display_scale
            ox, oy = self.display_offset
            
            dx1 = x1 * scale + ox
            dy1 = y1 * scale + oy
            dx2 = x2 * scale + ox
            dy2 = y2 * scale + oy
            
            # Draw box
            self.canvas.create_rectangle(
                dx1, dy1, dx2, dy2,
                outline=f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}',
                width=3
            )
            
            # Draw label
            label = f"{cls}: {conf:.0%}"
            self.canvas.create_text(
                dx1 + 5, dy1 - 20,
                text=label,
                anchor='nw',
                fill=f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}',
                font=('Arial', 10, 'bold')
            )
    
    def update_results(self, status, missing, warnings):
        """Update results display"""
        # Safety status
        self.safety_var.set(status)
        if status == "SAFE":
            self.safety_label.config(fg='#27ae60')
        elif status == "UNSAFE":
            self.safety_label.config(fg='#e74c3c')
        else:
            self.safety_label.config(fg='#f39c12')
        
        # Detections text
        self.det_text.delete(1.0, tk.END)
        
        detected = 0
        for det in self.detections:
            cls = det['class']
            conf = det['confidence']
            present = det['present']
            
            if present:
                detected += 1
                if conf > 0.7:
                    prefix = "✅ "
                elif conf > 0.4:
                    prefix = "⚠️ "
                else:
                    prefix = "❓ "
                
                line = f"{prefix}{cls:12} {conf:>5.0%}\n"
                if det['bbox']:
                    x1, y1, x2, y2 = det['bbox']
                    line += f"    [({x1},{y1})-({x2},{y2})]\n"
            else:
                line = f"❌ {cls:12} Not detected\n"
            
            self.det_text.insert(tk.END, line)
        
        # Alerts text
        self.alert_text.delete(1.0, tk.END)
        
        all_alerts = missing + warnings
        if all_alerts:
            for alert in all_alerts:
                self.alert_text.insert(tk.END, f"• {alert}\n")
        else:
            self.alert_text.insert(tk.END, "✅ All requirements met!\n")
        
        # Statistics
        self.stats_var.set(f"Objects: {detected}/7 detected")
    
    def set_buttons_state(self, state):
        """Enable/disable buttons"""
        buttons = ['dataset', 'model', 'image', 'detection', 'clear', 'info']
        for name in buttons:
            btn = getattr(self, f'btn_{name}', None)
            if btn:
                btn.config(state=state)
    
    def clear_all(self):
        """Clear all results"""
        self.canvas.delete("all")
        self.canvas.create_text(
            400, 250,
            text="No Image Loaded\n\nLoad an image to begin detection",
            font=('Arial', 14),
            fill='#7f8c8d',
            justify='center'
        )
        
        self.det_text.delete(1.0, tk.END)
        self.alert_text.delete(1.0, tk.END)
        
        self.safety_var.set("UNKNOWN")
        self.safety_label.config(fg='#f39c12')
        
        self.stats_var.set("Objects: 0/7")
        self.image_info.set("")
        
        self.current_image = None
        self.detections = []
        
        self.update_status("Cleared all")
    
    def show_info(self):
        """Show model information"""
        info = tk.Toplevel(self.root)
        info.title("Model Information")
        info.geometry("500x400")
        
        text = f"""
        Model: HeadGear Sentinel PPE Detector
        Status: {'Trained' if self.detector.is_trained else 'Not Trained'}
        
        Detection Method:
        • Sliding Window Object Detection
        • HOG Feature Extraction
        • SVM Classification
        • Multi-scale Processing
        
        Classes (7):
        1. Person
        2. Helmet
        3. Goggles
        4. Mask
        5. Gloves
        6. Vest
        7. Boots
        
        Safety Rules:
        • Person + Helmet + Vest = SAFE
        • Missing critical PPE = UNSAFE
        
        System: CPU-Only, Linux Compatible
        """
        
        tk.Label(
            info,
            text=text,
            font=('Courier New', 10),
            justify='left',
            padx=20,
            pady=20
        ).pack(fill='both', expand=True)
    
    def update_status(self, message):
        """Update status bar"""
        self.status_var.set(f"Status: {message}")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point"""
    print("Starting HeadGear Sentinel...")
    print("CPU-Only Object Detection System")
    
    # Create root window
    root = tk.Tk()
    
    # Create application
    app = HeadGearApp(root)
    
    # Center window
    root.update_idletasks()
    w = root.winfo_width()
    h = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (w // 2)
    y = (root.winfo_screenheight() // 2) - (h // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main()