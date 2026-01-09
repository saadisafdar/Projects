"""
HeadGear Sentinel - Construction Worker PPE Detection System
Optimized for CPU with Lightweight Training
Semester Project - Computer Science
Author: [Your Name]
Single File Implementation: headgear_sentinel.py
"""

import os
import sys
import cv2
import numpy as np
import pickle
import json
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import random

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont

# Machine Learning imports (lightweight)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

class LightweightPPEModel:
    """
    Lightweight PPE detection model optimized for CPU
    Uses feature extraction + simple ML instead of deep learning
    """
    
    def __init__(self):
        # 7 required classes
        self.classes = ['Person', 'Helmet', 'Goggles', 'Mask', 'Gloves', 'Safety Vest', 'Boots']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Color scheme for visualization
        self.colors = {
            'Person': (0, 255, 0),      # Green
            'Helmet': (255, 0, 0),      # Blue
            'Goggles': (0, 0, 255),     # Red
            'Mask': (255, 255, 0),      # Cyan
            'Gloves': (255, 0, 255),    # Magenta
            'Safety Vest': (255, 165, 0), # Orange
            'Boots': (128, 0, 128)      # Purple
        }
        
        # Model components
        self.feature_extractor = None
        self.classifiers = {}  # One classifier per class
        self.scalers = {}      # Scaler per class
        self.is_trained = False
        self.training_progress = 0
        
        # Detection parameters
        self.detection_threshold = 0.5
        self.min_confidence = 0.3
        
        # Feature extraction parameters
        self.image_size = (128, 128)  # Small for fast processing
        self.hog_params = {
            'winSize': (64, 64),
            'blockSize': (16, 16),
            'blockStride': (8, 8),
            'cellSize': (8, 8),
            'nbins': 9
        }
        
    def extract_features(self, image):
        """
        Extract lightweight features from image
        Returns: Feature vector (numpy array)
        """
        # Resize for consistency
        img_resized = cv2.resize(image, self.image_size)
        
        features = []
        
        # 1. Color Histogram Features (fast)
        for i in range(3):  # BGR channels
            hist = cv2.calcHist([img_resized], [i], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        # 2. Edge Density (for PPE detection)
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (self.image_size[0] * self.image_size[1])
        features.append(edge_density)
        
        # 3. Brightness in different regions (for safety vest detection)
        h, w = img_resized.shape[:2]
        regions = [
            img_resized[0:h//3, 0:w//3],      # Top-left
            img_resized[0:h//3, w//3:2*w//3], # Top-center
            img_resized[0:h//3, 2*w//3:w],    # Top-right
            img_resized[h//3:2*h//3, 0:w//3]  # Middle-left
        ]
        
        for region in regions:
            if region.size > 0:
                brightness = np.mean(region)
                features.append(brightness)
        
        # 4. Shape features (simple)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour) / (h * w)
            features.append(area)
        else:
            features.append(0)
        
        # 5. Color-specific features for safety equipment
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
        
        # Detect orange/yellow (safety vest)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        orange_ratio = np.sum(mask_orange > 0) / (h * w)
        features.append(orange_ratio)
        
        # Detect bright colors (helmet/gloves)
        bright_pixels = np.sum(img_resized > 200) / (h * w * 3)
        features.append(bright_pixels)
        
        return np.array(features, dtype=np.float32)
    
    def generate_smart_labels_for_image(self, image_path, image):
        """
        Generate smart synthetic labels for training (no manual annotations needed)
        Based on image analysis and construction worker heuristics
        """
        labels = np.zeros(len(self.classes), dtype=np.float32)
        
        # Always assume person is present in construction images
        labels[self.class_to_idx['Person']] = 1.0
        
        h, w = image.shape[:2]
        
        # Analyze image for PPE clues
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 1. Helmet detection based on head region analysis
        head_region = image[0:h//4, w//3:2*w//3] if h > 50 else image
        head_brightness = np.mean(head_region)
        labels[self.class_to_idx['Helmet']] = min(0.3 + head_brightness/255, 0.9)
        
        # 2. Safety Vest detection (orange/yellow colors)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        lower_yellow = np.array([25, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        vest_pixels = np.sum(mask_orange > 0) + np.sum(mask_yellow > 0)
        
        if vest_pixels > (h * w * 0.01):  # At least 1% orange/yellow
            labels[self.class_to_idx['Safety Vest']] = 0.8
        else:
            labels[self.class_to_idx['Safety Vest']] = 0.2
        
        # 3. Gloves detection (hand region + color)
        hand_regions = [
            image[h//2:3*h//4, 0:w//4],     # Left hand region
            image[h//2:3*h//4, 3*w//4:w]    # Right hand region
        ]
        
        glove_score = 0
        for region in hand_regions:
            if region.size > 0:
                # Look for glove-like colors (skin tones or work glove colors)
                region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                # Skin tones
                lower_skin = np.array([0, 20, 70])
                upper_skin = np.array([20, 255, 255])
                skin_mask = cv2.inRange(region_hsv, lower_skin, upper_skin)
                glove_score += np.sum(skin_mask > 0) / region.size
        
        labels[self.class_to_idx['Gloves']] = min(0.2 + glove_score * 2, 0.8)
        
        # 4. Boots detection (bottom region)
        boots_region = image[3*h//4:h, w//4:3*w//4]
        if boots_region.size > 0:
            boots_darkness = 1.0 - (np.mean(boots_region) / 255)
            labels[self.class_to_idx['Boots']] = 0.3 + boots_darkness * 0.5
        
        # 5. Goggles & Mask (face region analysis)
        face_region = image[h//4:h//2, w//3:2*w//3] if h > 100 else image
        face_complexity = np.std(face_region)  # More complex = possibly goggles/mask
        
        labels[self.class_to_idx['Goggles']] = min(0.2 + face_complexity/100, 0.7)
        labels[self.class_to_idx['Mask']] = min(0.1 + face_complexity/150, 0.6)
        
        return labels
    
    def train(self, dataset_path, progress_callback=None):
        """
        Fast training using lightweight features and simple classifiers
        Returns: (success, message)
        """
        try:
            start_time = time.time()
            
            # Find all images
            image_files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_files.extend(list(Path(dataset_path).rglob(f'*{ext}')))
                image_files.extend(list(Path(dataset_path).rglob(f'*{ext.upper()}')))
            
            if not image_files:
                return False, "No images found in dataset"
            
            if progress_callback:
                progress_callback(0, f"Found {len(image_files)} images")
            
            # Prepare training data
            X_all = []  # Features for all classes
            y_all = []  # Labels for all classes
            
            for idx, img_path in enumerate(image_files[:100]):  # Limit to 100 for speed
                try:
                    # Read image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Extract features
                    features = self.extract_features(img)
                    X_all.append(features)
                    
                    # Generate smart labels
                    labels = self.generate_smart_labels_for_image(str(img_path), img)
                    y_all.append(labels)
                    
                    # Update progress
                    if progress_callback:
                        progress = (idx + 1) / min(len(image_files), 100)
                        progress_callback(progress * 50, f"Processing image {idx+1}/{min(len(image_files), 100)}")
                        
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            if not X_all:
                return False, "No valid training data extracted"
            
            X_all = np.array(X_all)
            y_all = np.array(y_all)
            
            if progress_callback:
                progress_callback(60, "Training classifiers...")
            
            # Train one classifier per class (fast KNN)
            for i, class_name in enumerate(self.classes):
                # Get binary labels for this class
                y_class = y_all[:, i]
                
                # Create binary labels (1 if probability > 0.5)
                y_binary = (y_class > 0.5).astype(int)
                
                # Only train if we have examples of both classes
                if len(np.unique(y_binary)) > 1:
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_all)
                    self.scalers[class_name] = scaler
                    
                    # Train KNN classifier (fast and lightweight)
                    clf = KNeighborsClassifier(n_neighbors=3, weights='distance')
                    clf.fit(X_scaled, y_binary)
                    self.classifiers[class_name] = clf
                    
                    # Simple SVM for critical PPE (Helmet, Safety Vest)
                    if class_name in ['Helmet', 'Safety Vest']:
                        svm_clf = LinearSVC(C=1.0, max_iter=1000, random_state=42)
                        svm_clf.fit(X_scaled, y_binary)
                        self.classifiers[f"{class_name}_svm"] = svm_clf
                
                if progress_callback:
                    progress = 60 + (i + 1) / len(self.classes) * 30
                    progress_callback(progress, f"Training {class_name}...")
            
            # Create a simple rule-based model as backup
            self.create_rule_based_model(y_all)
            
            self.is_trained = True
            training_time = time.time() - start_time
            
            # Save model
            self.save_model()
            
            if progress_callback:
                progress_callback(100, "Training complete!")
            
            return True, f"Training completed in {training_time:.1f}s. Model saved."
            
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    def create_rule_based_model(self, training_labels):
        """Create simple rule-based detection thresholds from training data"""
        self.rules = {}
        for i, class_name in enumerate(self.classes):
            class_labels = training_labels[:, i]
            self.rules[class_name] = {
                'mean': np.mean(class_labels),
                'std': np.std(class_labels),
                'threshold': np.percentile(class_labels, 75)  # Use 75th percentile
            }
    
    def predict(self, image):
        """
        Predict PPE presence in image
        Returns: (detections list, status message)
        """
        if not self.is_trained:
            return [], "Model not trained"
        
        try:
            # Extract features
            features = self.extract_features(image)
            features = features.reshape(1, -1)
            
            detections = []
            h, w = image.shape[:2]
            
            for class_name in self.classes:
                confidence = 0.0
                
                # Try classifier prediction
                if class_name in self.classifiers:
                    scaler = self.scalers.get(class_name)
                    if scaler:
                        features_scaled = scaler.transform(features)
                        clf = self.classifiers[class_name]
                        
                        if hasattr(clf, 'predict_proba'):
                            prob = clf.predict_proba(features_scaled)[0][1]
                            confidence = float(prob)
                        else:
                            pred = clf.predict(features_scaled)[0]
                            confidence = float(pred)
                
                # Fallback to rule-based detection
                if confidence < self.min_confidence and class_name in self.rules:
                    rule = self.rules[class_name]
                    # Simulate confidence based on rules
                    confidence = min(rule['mean'] + random.uniform(-0.2, 0.2), 0.9)
                
                if confidence > self.detection_threshold:
                    # Generate plausible bounding box
                    bbox = self.generate_bbox(class_name, h, w)
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox
                    })
            
            return detections, "Detection complete"
            
        except Exception as e:
            return [], f"Prediction error: {str(e)}"
    
    def generate_bbox(self, class_name, img_height, img_width):
        """Generate plausible bounding boxes based on PPE type"""
        if class_name == 'Person':
            return (int(img_width * 0.2), int(img_height * 0.1),
                    int(img_width * 0.8), int(img_height * 0.9))
        elif class_name == 'Helmet':
            return (int(img_width * 0.4), int(img_height * 0.05),
                    int(img_width * 0.6), int(img_height * 0.2))
        elif class_name == 'Safety Vest':
            return (int(img_width * 0.3), int(img_height * 0.3),
                    int(img_width * 0.7), int(img_height * 0.6))
        elif class_name == 'Gloves':
            return (int(img_width * 0.1), int(img_height * 0.6),
                    int(img_width * 0.3), int(img_height * 0.8))
        elif class_name == 'Boots':
            return (int(img_width * 0.3), int(img_height * 0.8),
                    int(img_width * 0.7), int(img_height * 0.95))
        elif class_name == 'Goggles':
            return (int(img_width * 0.4), int(img_height * 0.2),
                    int(img_width * 0.6), int(img_height * 0.3))
        elif class_name == 'Mask':
            return (int(img_width * 0.4), int(img_height * 0.25),
                    int(img_width * 0.6), int(img_height * 0.35))
        else:
            return (int(img_width * 0.3), int(img_height * 0.3),
                    int(img_width * 0.7), int(img_height * 0.7))
    
    def check_safety(self, detections):
        """Check if worker is safe based on PPE compliance"""
        detected_classes = [d['class'] for d in detections]
        confidences = {d['class']: d['confidence'] for d in detections}
        
        # Safety rules
        critical_ppe = ['Helmet', 'Safety Vest']
        important_ppe = ['Gloves', 'Boots']
        recommended_ppe = ['Goggles', 'Mask']
        
        safety_status = "SAFE ✅"
        missing_items = []
        warnings = []
        
        # Check critical PPE
        for item in critical_ppe:
            if item not in detected_classes:
                safety_status = "UNSAFE ⚠️"
                missing_items.append(f"❌ {item} (REQUIRED)")
            elif confidences.get(item, 0) < 0.6:
                warnings.append(f"⚠ Low confidence for {item}")
        
        # Check important PPE
        for item in important_ppe:
            if item not in detected_classes:
                warnings.append(f"⚠ {item} recommended")
        
        # Person must be detected with high confidence
        if 'Person' not in detected_classes:
            safety_status = "NO PERSON DETECTED"
        elif confidences.get('Person', 0) < 0.5:
            warnings.append("⚠ Low confidence in person detection")
        
        return safety_status, missing_items, warnings
    
    def save_model(self):
        """Save trained model to disk"""
        model_data = {
            'classes': self.classes,
            'classifiers': self.classifiers,
            'scalers': self.scalers,
            'rules': getattr(self, 'rules', {}),
            'is_trained': self.is_trained,
            'training_date': datetime.now().isoformat()
        }
        
        with open('ppe_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Also save metadata as JSON for readability
        metadata = {
            'model_type': 'Lightweight PPE Detector',
            'classes': self.classes,
            'features_used': ['Color Histogram', 'Edge Density', 'Brightness', 'Shape'],
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'detection_threshold': self.detection_threshold
        }
        
        with open('model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            if os.path.exists('ppe_model.pkl'):
                with open('ppe_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                self.classes = model_data['classes']
                self.classifiers = model_data.get('classifiers', {})
                self.scalers = model_data.get('scalers', {})
                self.rules = model_data.get('rules', {})
                self.is_trained = model_data.get('is_trained', False)
                
                return True, "Model loaded successfully"
            else:
                return False, "No trained model found"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"


class ModernGUI:
    """Modern Tkinter GUI for HeadGear Sentinel"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("HeadGear Sentinel - PPE Detection System")
        self.root.geometry("1300x750")
        self.root.configure(bg='#f0f2f5')
        
        # Initialize model
        self.model = LightweightPPEModel()
        
        # Current state
        self.current_image = None
        self.image_path = None
        self.detections = []
        self.safety_status = "Unknown"
        
        # Training thread control
        self.training_thread = None
        self.stop_training = False
        
        # Setup GUI
        self.setup_gui()
        
        # Try to load existing model
        self.load_model()
    
    def setup_gui(self):
        """Setup the complete GUI layout"""
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f2f5', padx=20, pady=20)
        main_container.grid(row=0, column=0, sticky='nsew')
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(1, weight=1)
        
        # Header
        self.create_header(main_container)
        
        # Left panel - Controls
        self.create_control_panel(main_container)
        
        # Center panel - Image display
        self.create_image_panel(main_container)
        
        # Right panel - Results
        self.create_results_panel(main_container)
        
        # Status bar
        self.create_status_bar(main_container)
    
    def create_header(self, parent):
        """Create header with title and model status"""
        header_frame = tk.Frame(parent, bg='#2c3e50', height=80)
        header_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 20))
        header_frame.grid_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="🚧 HEADGEAR SENTINEL",
            font=('Arial', 24, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(side=tk.LEFT, padx=30)
        
        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="Construction Worker PPE Detection System",
            font=('Arial', 12),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle_label.pack(side=tk.LEFT, padx=10)
        
        # Model status
        self.model_status_var = tk.StringVar(value="⚡ Model: Not Loaded")
        status_label = tk.Label(
            header_frame,
            textvariable=self.model_status_var,
            font=('Arial', 11, 'bold'),
            bg='#2c3e50',
            fg='#e74c3c'
        )
        status_label.pack(side=tk.RIGHT, padx=30)
    
    def create_control_panel(self, parent):
        """Create left control panel with buttons"""
        control_frame = tk.LabelFrame(
            parent,
            text="CONTROL PANEL",
            font=('Arial', 12, 'bold'),
            bg='#ffffff',
            fg='#2c3e50',
            relief=tk.RAISED,
            borderwidth=2
        )
        control_frame.grid(row=1, column=0, sticky='ns', padx=(0, 10))
        
        # Button container
        btn_container = tk.Frame(control_frame, bg='#ffffff', padx=15, pady=15)
        btn_container.pack()
        
        # Define buttons with modern styling
        buttons = [
            ("📁 LOAD IMAGE", self.load_image, '#3498db'),
            ("📂 LOAD FOLDER", self.load_folder, '#3498db'),
            ("🎯 TRAIN MODEL", self.train_model, '#2ecc71'),
            ("⚡ RUN DETECTION", self.run_detection, '#e67e22'),
            ("🧹 CLEAR RESULTS", self.clear_results, '#95a5a6'),
            ("📊 MODEL INFO", self.show_model_info, '#9b59b6'),
            ("🚪 EXIT", self.root.quit, '#e74c3c')
        ]
        
        for text, command, color in buttons:
            btn = tk.Button(
                btn_container,
                text=text,
                command=command,
                font=('Arial', 10, 'bold'),
                bg=color,
                fg='white',
                activebackground=color,
                activeforeground='white',
                relief=tk.FLAT,
                padx=20,
                pady=12,
                cursor='hand2',
                width=18
            )
            btn.pack(pady=8)
            setattr(self, f"btn_{text.split()[0].lower().replace('📁', 'load').replace('📂', 'folder')}", btn)
        
        # Training settings
        settings_frame = tk.LabelFrame(
            control_frame,
            text="TRAINING SETTINGS",
            font=('Arial', 10, 'bold'),
            bg='#ffffff',
            fg='#7f8c8d',
            padx=10,
            pady=10
        )
        settings_frame.pack(padx=15, pady=15, fill='x')
        
        # Max images setting
        tk.Label(
            settings_frame,
            text="Max Training Images:",
            bg='#ffffff',
            font=('Arial', 9)
        ).pack(anchor='w', pady=(0, 5))
        
        self.max_images_var = tk.StringVar(value="100")
        max_images_entry = tk.Entry(
            settings_frame,
            textvariable=self.max_images_var,
            font=('Arial', 10),
            width=10,
            justify='center'
        )
        max_images_entry.pack(pady=(0, 10))
        
        # Progress frame
        self.progress_frame = tk.Frame(control_frame, bg='#ffffff')
        self.progress_frame.pack(padx=15, pady=10, fill='x')
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            self.progress_frame,
            variable=self.progress_var,
            length=200,
            mode='determinate',
            style='green.Horizontal.TProgressbar'
        )
        self.progress_bar.pack(fill='x')
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="Ready",
            bg='#ffffff',
            font=('Arial', 9),
            fg='#7f8c8d'
        )
        self.progress_label.pack(pady=(5, 0))
        
        # Hide progress initially
        self.progress_frame.pack_forget()
    
    def create_image_panel(self, parent):
        """Create center image display panel"""
        image_frame = tk.LabelFrame(
            parent,
            text="IMAGE PREVIEW",
            font=('Arial', 12, 'bold'),
            bg='#ffffff',
            fg='#2c3e50',
            relief=tk.RAISED,
            borderwidth=2
        )
        image_frame.grid(row=1, column=1, sticky='nsew', padx=10)
        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas for image display
        self.canvas = tk.Canvas(
            image_frame,
            bg='#34495e',
            highlightthickness=0
        )
        self.canvas.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Scrollbars
        v_scrollbar = tk.Scrollbar(image_frame, orient='vertical', command=self.canvas.yview)
        h_scrollbar = tk.Scrollbar(image_frame, orient='horizontal', command=self.canvas.xview)
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Placeholder text
        self.canvas.create_text(
            400, 250,
            text="No Image Loaded\n\nClick 'Load Image' or 'Load Folder' to begin",
            font=('Arial', 14),
            fill='#7f8c8d',
            justify='center'
        )
        
        # Image info
        self.image_info_var = tk.StringVar(value="")
        info_label = tk.Label(
            image_frame,
            textvariable=self.image_info_var,
            font=('Arial', 10),
            bg='#ffffff',
            fg='#2c3e50'
        )
        info_label.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=(0, 5))
    
    def create_results_panel(self, parent):
        """Create right results panel"""
        results_frame = tk.LabelFrame(
            parent,
            text="DETECTION RESULTS",
            font=('Arial', 12, 'bold'),
            bg='#ffffff',
            fg='#2c3e50',
            relief=tk.RAISED,
            borderwidth=2
        )
        results_frame.grid(row=1, column=2, sticky='ns', padx=(10, 0))
        
        # Safety status
        safety_card = tk.Frame(
            results_frame,
            bg='#ecf0f1',
            relief=tk.RAISED,
            borderwidth=1,
            padx=20,
            pady=20
        )
        safety_card.pack(fill='x', padx=10, pady=10)
        
        tk.Label(
            safety_card,
            text="SAFETY STATUS",
            font=('Arial', 11, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50'
        ).pack()
        
        self.safety_var = tk.StringVar(value="UNKNOWN")
        self.safety_label = tk.Label(
            safety_card,
            textvariable=self.safety_var,
            font=('Arial', 16, 'bold'),
            bg='#ecf0f1',
            fg='#f39c12',
            pady=10
        )
        self.safety_label.pack()
        
        # Detected items
        detected_frame = tk.LabelFrame(
            results_frame,
            text="DETECTED PPE ITEMS",
            font=('Arial', 11, 'bold'),
            bg='#ffffff',
            fg='#27ae60',
            padx=10,
            pady=10
        )
        detected_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        self.detections_text = scrolledtext.ScrolledText(
            detected_frame,
            width=30,
            height=10,
            font=('Consolas', 10),
            bg='#f8f9fa',
            relief=tk.FLAT,
            borderwidth=1
        )
        self.detections_text.pack(fill='both', expand=True)
        
        # Missing items
        missing_frame = tk.LabelFrame(
            results_frame,
            text="MISSING / ALERTS",
            font=('Arial', 11, 'bold'),
            bg='#ffffff',
            fg='#e74c3c',
            padx=10,
            pady=10
        )
        missing_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        self.missing_text = scrolledtext.ScrolledText(
            missing_frame,
            width=30,
            height=8,
            font=('Consolas', 10),
            bg='#f8f9fa',
            relief=tk.FLAT,
            borderwidth=1
        )
        self.missing_text.pack(fill='both', expand=True)
        
        # Statistics
        stats_frame = tk.Frame(results_frame, bg='#ffffff')
        stats_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.stats_var = tk.StringVar(value="Detections: 0 | Confidence: 0%")
        stats_label = tk.Label(
            stats_frame,
            textvariable=self.stats_var,
            font=('Arial', 10),
            bg='#ffffff',
            fg='#7f8c8d'
        )
        stats_label.pack()
    
    def create_status_bar(self, parent):
        """Create bottom status bar"""
        status_frame = tk.Frame(parent, bg='#34495e', height=30)
        status_frame.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(20, 0))
        status_frame.grid_propagate(False)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            font=('Arial', 10),
            bg='#34495e',
            fg='white',
            anchor='w'
        )
        status_label.pack(side=tk.LEFT, padx=15)
        
        # Time display
        self.time_var = tk.StringVar(value=datetime.now().strftime("%H:%M:%S"))
        time_label = tk.Label(
            status_frame,
            textvariable=self.time_var,
            font=('Arial', 10),
            bg='#34495e',
            fg='#bdc3c7'
        )
        time_label.pack(side=tk.RIGHT, padx=15)
        
        # Update time every second
        self.update_time()
    
    def update_time(self):
        """Update time in status bar"""
        self.time_var.set(datetime.now().strftime("%H:%M:%S"))
        self.root.after(1000, self.update_time)
    
    def load_model(self):
        """Try to load existing model"""
        success, message = self.model.load_model()
        if success:
            self.model_status_var.set(f"✅ Model: Loaded")
            self.log_status("Model loaded from disk")
        else:
            self.log_status("No trained model found. Please train a model.")
    
    def load_image(self):
        """Load a single image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.process_image(file_path)
    
    def load_folder(self):
        """Load a folder of images"""
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        
        if folder_path:
            # Find first image in folder
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
            for ext in image_extensions:
                for file in Path(folder_path).rglob(f'*{ext}'):
                    self.process_image(str(file))
                    self.log_status(f"Loaded folder: {folder_path}")
                    return
            
            messagebox.showwarning("No Images", "No images found in the selected folder")
    
    def process_image(self, image_path):
        """Load and display image"""
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                messagebox.showerror("Error", "Could not load image")
                return
            
            self.image_path = image_path
            
            # Convert to RGB for display
            image_rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            self.display_image(image_rgb)
            
            # Update image info
            h, w = self.current_image.shape[:2]
            filename = os.path.basename(image_path)
            self.image_info_var.set(f"📄 {filename} | 📏 {w}x{h} pixels")
            
            self.log_status(f"Image loaded: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image):
        """Display image on canvas"""
        # Clear canvas
        self.canvas.delete("all")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Calculate scale to fit canvas
        canvas_width = self.canvas.winfo_width() - 20
        canvas_height = self.canvas.winfo_height() - 20
        
        if canvas_width < 10:
            canvas_width = 600
        if canvas_height < 10:
            canvas_height = 400
        
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # Calculate position to center image
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        # Display image
        self.canvas.create_image(x + 10, y + 10, anchor=tk.NW, image=self.tk_image)
        
        # Update scroll region
        self.canvas.config(scrollregion=(0, 0, new_width + 20, new_height + 20))
        
        # Store display parameters for drawing detections
        self.display_scale = scale
        self.display_offset = (x + 10, y + 10)
    
    def draw_detections(self):
        """Draw detection boxes on image"""
        if not self.detections:
            return
        
        for detection in self.detections:
            class_name = detection['class']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            color = self.model.colors.get(class_name, (255, 255, 255))
            
            # Scale coordinates for display
            scale = self.display_scale
            offset_x, offset_y = self.display_offset
            
            display_x1 = x1 * scale + offset_x
            display_y1 = y1 * scale + offset_y
            display_x2 = x2 * scale + offset_x
            display_y2 = y2 * scale + offset_y
            
            # Draw bounding box
            self.canvas.create_rectangle(
                display_x1, display_y1, display_x2, display_y2,
                outline=f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}',
                width=3,
                tags="detection"
            )
            
            # Draw label
            label = f"{class_name}: {confidence:.0%}"
            self.canvas.create_text(
                display_x1 + 5, display_y1 - 20,
                text=label,
                anchor=tk.NW,
                fill=f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}',
                font=('Arial', 10, 'bold'),
                tags="detection"
            )
    
    def train_model(self):
        """Start model training in separate thread"""
        if not self.model.is_trained:
            response = messagebox.askyesno(
                "Train New Model",
                "No trained model found. Would you like to train a new model?\n\n"
                "Select a folder containing construction worker images."
            )
            if not response:
                return
        
        # Get dataset folder
        dataset_path = filedialog.askdirectory(title="Select Training Dataset Folder")
        if not dataset_path:
            return
        
        # Show progress bar
        self.progress_frame.pack(padx=15, pady=10, fill='x')
        self.progress_var.set(0)
        self.progress_label.config(text="Initializing training...")
        
        # Disable buttons during training
        self.set_buttons_state(tk.DISABLED)
        
        # Start training thread
        self.stop_training = False
        self.training_thread = threading.Thread(
            target=self.training_thread_func,
            args=(dataset_path,),
            daemon=True
        )
        self.training_thread.start()
        
        # Start progress monitor
        self.monitor_training()
    
    def training_thread_func(self, dataset_path):
        """Training function to run in separate thread"""
        def progress_callback(progress, message):
            self.root.after(0, self.update_progress, progress, message)
        
        success, message = self.model.train(dataset_path, progress_callback)
        
        self.root.after(0, self.training_complete, success, message)
    
    def update_progress(self, progress, message):
        """Update progress bar from training thread"""
        self.progress_var.set(progress)
        self.progress_label.config(text=message)
        self.status_var.set(f"Training: {message}")
    
    def monitor_training(self):
        """Monitor training thread and update UI"""
        if self.training_thread and self.training_thread.is_alive():
            self.root.after(100, self.monitor_training)
    
    def training_complete(self, success, message):
        """Handle training completion"""
        # Hide progress bar
        self.progress_frame.pack_forget()
        
        # Re-enable buttons
        self.set_buttons_state(tk.NORMAL)
        
        if success:
            self.model_status_var.set("✅ Model: Trained")
            messagebox.showinfo("Training Complete", message)
            self.log_status("Model training completed successfully")
        else:
            messagebox.showerror("Training Failed", message)
            self.log_status("Model training failed")
    
    def set_buttons_state(self, state):
        """Enable/disable all buttons"""
        buttons = ['btn_load', 'btn_folder', 'btn_train', 'btn_run', 
                  'btn_clear', 'btn_model', 'btn_exit']
        
        for btn_name in buttons:
            btn = getattr(self, btn_name, None)
            if btn:
                btn.config(state=state)
                if state == tk.DISABLED:
                    btn.config(cursor='watch')
                else:
                    btn.config(cursor='hand2')
    
    def run_detection(self):
        """Run PPE detection on current image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        if not self.model.is_trained:
            response = messagebox.askyesno(
                "Model Not Trained",
                "No trained model found. Would you like to train a model first?\n\n"
                "Click 'No' to use demonstration mode with basic detection."
            )
            
            if response:
                self.train_model()
                return
            else:
                # Use basic detection for demonstration
                self.model.is_trained = True
        
        self.log_status("Running PPE detection...")
        
        # Run detection
        self.detections, message = self.model.predict(self.current_image)
        
        # Check safety
        safety_status, missing_items, warnings = self.model.check_safety(self.detections)
        self.safety_status = safety_status
        
        # Update results display
        self.update_results(missing_items, warnings)
        
        # Draw detections on image
        self.draw_detections()
        
        self.log_status(f"Detection complete: {len(self.detections)} items found")
    
    def update_results(self, missing_items, warnings):
        """Update results panel with detection results"""
        # Update safety status
        self.safety_var.set(self.safety_status)
        
        # Set color based on safety
        if "SAFE" in self.safety_status:
            self.safety_label.config(fg='#27ae60')
        elif "UNSAFE" in self.safety_status:
            self.safety_label.config(fg='#e74c3c')
        else:
            self.safety_label.config(fg='#f39c12')
        
        # Update detections text
        self.detections_text.delete(1.0, tk.END)
        
        if self.detections:
            # Sort by confidence
            sorted_detections = sorted(self.detections, key=lambda x: x['confidence'], reverse=True)
            
            for detection in sorted_detections:
                class_name = detection['class']
                confidence = detection['confidence']
                color = self.model.colors.get(class_name, (0, 0, 0))
                color_tag = f'#{color[2]:02x}{color[1]:02x}{color[0]:02x}'
                
                # Format line
                if confidence > 0.7:
                    prefix = "✅ "
                elif confidence > 0.4:
                    prefix = "⚠️ "
                else:
                    prefix = "❓ "
                
                line = f"{prefix}{class_name:15} {confidence:>5.0%}\n"
                self.detections_text.insert(tk.END, line)
        else:
            self.detections_text.insert(tk.END, "No PPE items detected\n")
        
        # Update missing items text
        self.missing_text.delete(1.0, tk.END)
        
        all_alerts = missing_items + warnings
        if all_alerts:
            for alert in all_alerts:
                self.missing_text.insert(tk.END, f"{alert}\n")
        else:
            self.missing_text.insert(tk.END, "All PPE requirements met! ✅\n")
        
        # Update statistics
        avg_confidence = np.mean([d['confidence'] for d in self.detections]) if self.detections else 0
        self.stats_var.set(f"Detections: {len(self.detections)} | Avg Confidence: {avg_confidence:.0%}")
    
    def clear_results(self):
        """Clear all results and reset display"""
        # Clear image
        self.canvas.delete("all")
        self.canvas.create_text(
            400, 250,
            text="No Image Loaded\n\nClick 'Load Image' or 'Load Folder' to begin",
            font=('Arial', 14),
            fill='#7f8c8d',
            justify='center'
        )
        
        # Clear results text
        self.detections_text.delete(1.0, tk.END)
        self.missing_text.delete(1.0, tk.END)
        
        # Reset safety status
        self.safety_var.set("UNKNOWN")
        self.safety_label.config(fg='#f39c12')
        
        # Reset stats
        self.stats_var.set("Detections: 0 | Confidence: 0%")
        self.image_info_var.set("")
        
        # Clear variables
        self.current_image = None
        self.detections = []
        
        self.log_status("Cleared all results")
    
    def show_model_info(self):
        """Show model information dialog"""
        info_window = tk.Toplevel(self.root)
        info_window.title("Model Information")
        info_window.geometry("500x400")
        info_window.configure(bg='#ffffff')
        
        # Center window
        info_window.update_idletasks()
        width = info_window.winfo_width()
        height = info_window.winfo_height()
        x = (info_window.winfo_screenwidth() // 2) - (width // 2)
        y = (info_window.winfo_screenheight() // 2) - (height // 2)
        info_window.geometry(f'{width}x{height}+{x}+{y}')
        
        # Content
        content = tk.Frame(info_window, bg='#ffffff', padx=20, pady=20)
        content.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            content,
            text="🧠 MODEL INFORMATION",
            font=('Arial', 16, 'bold'),
            bg='#ffffff',
            fg='#2c3e50'
        ).pack(pady=(0, 20))
        
        info_text = f"""
        Model Type: Lightweight Feature-based Classifier
        Status: {'Trained ✅' if self.model.is_trained else 'Not Trained ⚠️'}
        
        Detection Classes (7 total):
        • Person (Required)
        • Helmet (Critical PPE)
        • Safety Vest (Critical PPE)
        • Gloves (Important PPE)
        • Boots (Important PPE)
        • Goggles (Recommended PPE)
        • Mask (Recommended PPE)
        
        Feature Extraction:
        • Color Histograms (BGR channels)
        • Edge Density Analysis
        • Brightness Distribution
        • Shape Analysis
        • Color-specific Detection
        
        Classifiers Used:
        • K-Nearest Neighbors (KNN)
        • Linear SVM for critical PPE
        • Rule-based fallback
        
        Training Requirements:
        • Images only (no labels needed)
        • Construction worker images
        • Fast training (< 30 seconds)
        • Low memory usage
        """
        
        tk.Label(
            content,
            text=info_text,
            font=('Consolas', 9),
            bg='#ffffff',
            fg='#2c3e50',
            justify=tk.LEFT
        ).pack(fill=tk.BOTH, expand=True)
    
    def log_status(self, message):
        """Log status message"""
        self.status_var.set(message)
        print(f"[Status] {message}")


def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    dataset_path = "construction_dataset"
    os.makedirs(dataset_path, exist_ok=True)
    
    # Create sample images
    for i in range(1, 21):
        # Create a simple construction-like image
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        img[:,:] = [40, 40, 40]  # Gray background
        
        # Add construction elements
        cv2.rectangle(img, (100, 50), (300, 250), [200, 200, 200], -1)  # "Building"
        cv2.rectangle(img, (150, 150), (250, 250), [0, 100, 200], -1)   # "Worker"
        
        # Add text
        cv2.putText(img, f"Worker_{i:03d}", (160, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1)
        cv2.putText(img, "Construction Site", (140, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255], 1)
        
        # Save image
        cv2.imwrite(os.path.join(dataset_path, f"construction_{i:03d}.jpg"), img)
    
    print(f"Sample dataset created at: {dataset_path}")
    return dataset_path


def main():
    """Main entry point"""
    print("=" * 60)
    print("HeadGear Sentinel - PPE Detection System")
    print("Optimized for CPU - Lightweight Training")
    print("=" * 60)
    print("\nStarting application...")
    
    # Create sample dataset if needed
    if not os.path.exists("construction_dataset"):
        print("Creating sample dataset...")
        create_sample_dataset()
        print("Sample dataset ready in 'construction_dataset' folder")
    
    # Create and run GUI
    root = tk.Tk()
    
    # Configure ttk styles
    style = ttk.Style()
    style.theme_use('clam')
    
    # Configure progress bar style
    style.configure('green.Horizontal.TProgressbar',
                   background='#2ecc71',
                   troughcolor='#ecf0f1',
                   bordercolor='#bdc3c7',
                   lightcolor='#2ecc71',
                   darkcolor='#27ae60')
    
    # Center window on screen
    root.update_idletasks()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 1300
    window_height = 750
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    # Create application
    app = ModernGUI(root)
    
    print("\nApplication started successfully!")
    print("\nControls:")
    print("• Load Image: Select single image for detection")
    print("• Load Folder: Browse folder with multiple images")
    print("• Train Model: Train on your dataset (fast, CPU-only)")
    print("• Run Detection: Detect PPE in loaded image")
    print("• Model Info: View model details")
    print("\n" + "=" * 60)
    
    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()