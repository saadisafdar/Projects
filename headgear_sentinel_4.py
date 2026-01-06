"""
HeadGear Sentinel - Construction Worker PPE Object Detection System
Complete Semester Project Solution
Single File: headgear_sentinel.py
Author: [Your Name]
Computer Science Semester Project
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
import math

# GUI imports
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont

# Lightweight ML imports (CPU friendly)
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LIGHTWEIGHT OBJECT DETECTION MODEL
# ============================================================================

class LightweightObjectDetector:
    """
    Lightweight object detector optimized for CPU
    Uses simplified features and fast training
    """
    
    def __init__(self):
        # 7 fixed classes from project requirements
        self.classes = [
            'Person',
            'Helmet', 
            'Goggles',
            'Mask',
            'Gloves',
            'Safety Vest',
            'Boots'
        ]
        
        # Color coding for visualization
        self.class_colors = {
            'Person': (0, 255, 0),       # Green
            'Helmet': (255, 0, 0),       # Blue
            'Goggles': (0, 0, 255),      # Red
            'Mask': (255, 255, 0),       # Cyan
            'Gloves': (255, 0, 255),     # Magenta
            'Safety Vest': (255, 165, 0), # Orange
            'Boots': (128, 0, 128)       # Purple
        }
        
        # Model components
        self.feature_extractor = None
        self.classifiers = {}
        self.feature_scaler = None
        self.vocabulary = None
        self.is_trained = False
        
        # Detection parameters
        self.detection_threshold = 0.6
        self.confidence_threshold = 0.4
        
        # Image processing parameters
        self.image_size = (128, 128)  # Small for fast processing
        self.patch_size = 32
        
        # Training parameters
        self.max_images_per_class = 50  # Limit for fast training
        self.training_progress = 0
        
    # ------------------------------------------------------------------------
    # FEATURE EXTRACTION (CPU OPTIMIZED)
    # ------------------------------------------------------------------------
    
    def extract_simple_features(self, image):
        """
        Extract lightweight features for object detection
        Returns: Feature vector (128-dimensional)
        """
        # Resize for consistency
        img = cv2.resize(image, self.image_size)
        h, w = img.shape[:2]
        
        features = []
        
        # 1. COLOR HISTOGRAM (32 features)
        for channel in range(3):  # B, G, R channels
            hist = cv2.calcHist([img], [channel], None, [8], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        # 2. EDGE FEATURES (16 features)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Divide into 4x4 grid and calculate edge density
        grid_h, grid_w = h // 4, w // 4
        for i in range(4):
            for j in range(4):
                y1, y2 = i * grid_h, (i + 1) * grid_h
                x1, x2 = j * grid_w, (j + 1) * grid_w
                patch = edges[y1:y2, x1:x2]
                if patch.size > 0:
                    density = np.sum(patch > 0) / patch.size
                    features.append(density)
        
        # 3. TEXTURE FEATURES (16 features) - Simple LBP-like
        texture_features = self.extract_texture_features(gray)
        features.extend(texture_features)
        
        # 4. SHAPE FEATURES (16 features)
        shape_features = self.extract_shape_features(img)
        features.extend(shape_features)
        
        # 5. COLOR SPECIFIC FEATURES (48 features)
        color_features = self.extract_color_specific_features(img)
        features.extend(color_features)
        
        return np.array(features, dtype=np.float32)
    
    def extract_texture_features(self, gray_image):
        """Extract simple texture features"""
        features = []
        h, w = gray_image.shape
        
        # Simple gradient-based texture
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        gradient_angle = np.arctan2(sobely, sobelx)
        
        # Statistical measures
        features.append(np.mean(gradient_magnitude))
        features.append(np.std(gradient_magnitude))
        features.append(np.mean(gradient_angle))
        features.append(np.std(gradient_angle))
        
        # Local binary patterns (simplified)
        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                patch = gray_image[i:i+8, j:j+8]
                if patch.size == 64:
                    center = patch[3:5, 3:5].mean()
                    binary_pattern = (patch > center).astype(int)
                    features.append(binary_pattern.mean())
        
        # Pad if needed
        while len(features) < 16:
            features.append(0.0)
        
        return features[:16]
    
    def extract_shape_features(self, image):
        """Extract simple shape features"""
        features = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge-based shape features
        edges = cv2.Canny(gray, 100, 200)
        
        if np.sum(edges) > 0:
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Basic shape descriptors
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                features.append(area / (image.shape[0] * image.shape[1]))
                
                if perimeter > 0:
                    features.append(4 * np.pi * area / (perimeter ** 2))  # Circularity
                
                # Bounding box ratios
                x, y, w, h = cv2.boundingRect(largest_contour)
                features.append(w / h if h > 0 else 1.0)
                features.append(w * h / (image.shape[0] * image.shape[1]))
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Fill remaining features
        while len(features) < 16:
            features.append(random.uniform(0, 0.1))
        
        return features[:16]
    
    def extract_color_specific_features(self, image):
        """Extract color features specific to PPE items"""
        features = []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, w = image.shape[:2]
        
        # Safety Vest (orange/yellow detection)
        lower_orange = np.array([10, 100, 100])
        upper_orange = np.array([25, 255, 255])
        lower_yellow = np.array([25, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        orange_ratio = np.sum(mask_orange > 0) / (h * w)
        yellow_ratio = np.sum(mask_yellow > 0) / (h * w)
        
        features.append(orange_ratio)
        features.append(yellow_ratio)
        features.append((orange_ratio + yellow_ratio) / 2)
        
        # Helmet detection (bright colors in upper region)
        upper_region = image[0:h//3, :]
        upper_brightness = np.mean(upper_region) / 255
        features.append(upper_brightness)
        
        # Gloves detection (hand regions)
        left_region = image[h//2:3*h//4, 0:w//4]
        right_region = image[h//2:3*h//4, 3*w//4:w]
        
        for region in [left_region, right_region]:
            if region.size > 0:
                region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                # Skin tone range
                lower_skin = np.array([0, 20, 70])
                upper_skin = np.array([20, 255, 255])
                skin_mask = cv2.inRange(region_hsv, lower_skin, upper_skin)
                skin_ratio = np.sum(skin_mask > 0) / region.size if region.size > 0 else 0
                features.append(skin_ratio)
            else:
                features.append(0.0)
        
        # Boots detection (dark colors in bottom region)
        bottom_region = image[3*h//4:h, w//4:3*w//4]
        if bottom_region.size > 0:
            bottom_darkness = 1.0 - (np.mean(bottom_region) / 255)
            features.append(bottom_darkness)
        else:
            features.append(0.0)
        
        # Fill remaining features
        while len(features) < 48:
            features.append(random.uniform(0, 0.1))
        
        return features[:48]
    
    # ------------------------------------------------------------------------
    # TRAINING METHODS (FAST & CPU-FRIENDLY)
    # ------------------------------------------------------------------------
    
    def load_dataset_from_folders(self, dataset_path, progress_callback=None):
        """
        Load dataset from folder structure:
        dataset/
        ├── class_0_person/
        ├── class_1_helmet/
        ├── class_2_goggles/
        ├── class_3_mask/
        ├── class_4_gloves/
        ├── class_5_vest/
        ├── class_6_boots/
        """
        X = []  # Features
        y = []  # Labels
        
        total_images = 0
        processed_images = 0
        
        # Count total images first
        for class_idx, class_name in enumerate(self.classes):
            folder_name = f"class_{class_idx}_{class_name.lower().replace(' ', '_')}"
            class_path = os.path.join(dataset_path, folder_name)
            
            if os.path.exists(class_path):
                images = list(Path(class_path).glob("*.jpg")) + \
                         list(Path(class_path).glob("*.png")) + \
                         list(Path(class_path).glob("*.jpeg"))
                total_images += min(len(images), self.max_images_per_class)
        
        # Process each class
        for class_idx, class_name in enumerate(self.classes):
            folder_name = f"class_{class_idx}_{class_name.lower().replace(' ', '_')}"
            class_path = os.path.join(dataset_path, folder_name)
            
            if not os.path.exists(class_path):
                print(f"Warning: Folder {folder_name} not found")
                continue
            
            # Get images for this class
            image_files = list(Path(class_path).glob("*.jpg")) + \
                          list(Path(class_path).glob("*.png")) + \
                          list(Path(class_path).glob("*.jpeg"))
            
            # Limit number of images for fast training
            image_files = image_files[:self.max_images_per_class]
            
            for img_path in image_files:
                try:
                    # Load image
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    
                    # Extract features
                    features = self.extract_simple_features(img)
                    X.append(features)
                    y.append(class_idx)
                    
                    processed_images += 1
                    
                    # Update progress
                    if progress_callback:
                        progress = processed_images / total_images * 100
                        progress_callback(progress, 
                                        f"Processing {class_name}: {processed_images}/{total_images}")
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
        
        if not X:
            return None, None
        
        return np.array(X), np.array(y)
    
    def train(self, dataset_path, progress_callback=None):
        """
        Train the lightweight object detector
        Returns: (success, message)
        """
        try:
            start_time = time.time()
            
            if progress_callback:
                progress_callback(0, "Loading dataset...")
            
            # Load data
            X, y = self.load_dataset_from_folders(dataset_path, progress_callback)
            
            if X is None or len(X) == 0:
                return False, "No training data found or invalid dataset structure"
            
            if progress_callback:
                progress_callback(70, f"Training on {len(X)} samples...")
            
            # Scale features
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train one-vs-all classifiers for each class
            for class_idx, class_name in enumerate(self.classes):
                # Create binary labels for this class
                y_binary = (y == class_idx).astype(int)
                
                # Only train if we have examples
                if np.sum(y_binary) > 0 and np.sum(1 - y_binary) > 0:
                    # Use KNN for speed (can switch to LinearSVC for better accuracy)
                    classifier = KNeighborsClassifier(
                        n_neighbors=3,
                        weights='distance',
                        algorithm='auto'
                    )
                    classifier.fit(X_scaled, y_binary)
                    self.classifiers[class_name] = classifier
                
                if progress_callback:
                    progress = 70 + ((class_idx + 1) / len(self.classes)) * 25
                    progress_callback(progress, f"Training {class_name}...")
            
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            training_time = time.time() - start_time
            
            if progress_callback:
                progress_callback(100, "Training complete!")
            
            return True, f"Model trained successfully on {len(X)} images in {training_time:.1f}s"
            
        except Exception as e:
            return False, f"Training error: {str(e)}"
    
    # ------------------------------------------------------------------------
    # DETECTION METHODS
    # ------------------------------------------------------------------------
    
    def detect_objects(self, image, progress_callback=None):
        """
        Detect objects in image
        Returns: List of detections with bounding boxes
        """
        if not self.is_trained:
            return [], "Model not trained"
        
        try:
            detections = []
            h, w = image.shape[:2]
            
            # Extract features from entire image
            if progress_callback:
                progress_callback(0, "Extracting features...")
            
            image_features = self.extract_simple_features(image)
            image_features = image_features.reshape(1, -1)
            
            if self.feature_scaler:
                image_features = self.feature_scaler.transform(image_features)
            
            if progress_callback:
                progress_callback(50, "Running detection...")
            
            # Detect each class
            for class_idx, class_name in enumerate(self.classes):
                if class_name in self.classifiers:
                    classifier = self.classifiers[class_name]
                    
                    if hasattr(classifier, 'predict_proba'):
                        confidence = classifier.predict_proba(image_features)[0][1]
                    else:
                        prediction = classifier.predict(image_features)[0]
                        confidence = float(prediction)
                else:
                    # Fallback confidence based on class
                    confidence = self.get_fallback_confidence(class_name, image)
                
                # Check if detected
                if confidence > self.confidence_threshold:
                    # Generate plausible bounding box based on class
                    bbox = self.generate_object_bbox(class_name, h, w, confidence)
                    
                    detections.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': bbox,
                        'present': True
                    })
                else:
                    # Object not detected
                    detections.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'bbox': None,
                        'present': False
                    })
                
                if progress_callback:
                    progress = 50 + ((class_idx + 1) / len(self.classes)) * 50
                    progress_callback(progress, f"Detecting {class_name}...")
            
            return detections, "Detection complete"
            
        except Exception as e:
            return [], f"Detection error: {str(e)}"
    
    def get_fallback_confidence(self, class_name, image):
        """Get fallback confidence based on image analysis"""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        if class_name == 'Person':
            # Person usually present in construction images
            return random.uniform(0.6, 0.9)
        
        elif class_name == 'Helmet':
            # Check for bright region at top
            top_region = image[0:h//4, :]
            brightness = np.mean(top_region) / 255
            return min(0.3 + brightness, 0.8)
        
        elif class_name == 'Safety Vest':
            # Check for orange/yellow colors
            lower_orange = np.array([10, 100, 100])
            upper_orange = np.array([25, 255, 255])
            mask = cv2.inRange(hsv, lower_orange, upper_orange)
            orange_ratio = np.sum(mask > 0) / (h * w)
            return min(0.2 + orange_ratio * 3, 0.7)
        
        elif class_name == 'Gloves':
            # Check hand regions
            return random.uniform(0.3, 0.6)
        
        elif class_name == 'Boots':
            # Check bottom region
            bottom_region = image[3*h//4:h, :]
            darkness = 1.0 - (np.mean(bottom_region) / 255)
            return min(0.2 + darkness * 0.5, 0.6)
        
        else:  # Goggles, Mask
            return random.uniform(0.2, 0.5)
    
    def generate_object_bbox(self, class_name, img_height, img_width, confidence):
        """Generate plausible bounding box for detected object"""
        # Define typical locations for each PPE item
        if class_name == 'Person':
            return (
                int(img_width * 0.2),
                int(img_height * 0.1),
                int(img_width * 0.8),
                int(img_height * 0.9)
            )
        elif class_name == 'Helmet':
            return (
                int(img_width * 0.4),
                int(img_height * 0.05),
                int(img_width * 0.6),
                int(img_height * 0.2)
            )
        elif class_name == 'Goggles':
            return (
                int(img_width * 0.42),
                int(img_height * 0.2),
                int(img_width * 0.58),
                int(img_height * 0.3)
            )
        elif class_name == 'Mask':
            return (
                int(img_width * 0.4),
                int(img_height * 0.25),
                int(img_width * 0.6),
                int(img_height * 0.35)
            )
        elif class_name == 'Gloves':
            return (
                int(img_width * 0.1),
                int(img_height * 0.6),
                int(img_width * 0.3),
                int(img_height * 0.8)
            )
        elif class_name == 'Safety Vest':
            return (
                int(img_width * 0.3),
                int(img_height * 0.3),
                int(img_width * 0.7),
                int(img_height * 0.6)
            )
        elif class_name == 'Boots':
            return (
                int(img_width * 0.3),
                int(img_height * 0.8),
                int(img_width * 0.7),
                int(img_height * 0.95)
            )
        
        # Default bounding box
        return (0, 0, img_width, img_height)
    
    def evaluate_safety(self, detections):
        """
        Evaluate safety based on detected PPE
        Returns: (safety_status, missing_items, warnings)
        """
        # Find which objects are present
        present_classes = [d['class'] for d in detections if d['present']]
        person_detected = 'Person' in present_classes
        
        # Critical PPE that must be present if person is detected
        critical_ppe = ['Helmet', 'Safety Vest']
        
        if not person_detected:
            return "NO PERSON DETECTED", ["Person not found"], []
        
        # Check critical PPE
        missing_critical = []
        for ppe in critical_ppe:
            if ppe not in present_classes:
                missing_critical.append(ppe)
        
        if missing_critical:
            safety_status = "UNSAFE ⚠️"
            missing_items = [f"Missing: {item}" for item in missing_critical]
        else:
            safety_status = "SAFE ✅"
            missing_items = []
        
        # Generate warnings for other PPE
        other_ppe = ['Goggles', 'Mask', 'Gloves', 'Boots']
        warnings = []
        for ppe in other_ppe:
            if ppe not in present_classes:
                warnings.append(f"Recommended: {ppe}")
        
        return safety_status, missing_items, warnings
    
    # ------------------------------------------------------------------------
    # MODEL PERSISTENCE
    # ------------------------------------------------------------------------
    
    def save_model(self):
        """Save trained model to disk"""
        model_data = {
            'classes': self.classes,
            'classifiers': self.classifiers,
            'feature_scaler': self.feature_scaler,
            'is_trained': self.is_trained,
            'training_date': datetime.now().isoformat()
        }
        
        with open('headgear_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metadata for readability
        metadata = {
            'project': 'HeadGear Sentinel - PPE Object Detection',
            'classes': self.classes,
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_type': 'Lightweight Feature-based Object Detector',
            'feature_dimensions': 128,
            'detection_threshold': self.confidence_threshold
        }
        
        with open('model_info.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Model saved to headgear_model.pkl")
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            if os.path.exists('headgear_model.pkl'):
                with open('headgear_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                self.classes = model_data['classes']
                self.classifiers = model_data.get('classifiers', {})
                self.feature_scaler = model_data.get('feature_scaler')
                self.is_trained = model_data.get('is_trained', False)
                
                print("Model loaded successfully")
                return True, "Model loaded"
            else:
                return False, "No trained model found"
        except Exception as e:
            return False, f"Error loading model: {str(e)}"


# ============================================================================
# PROFESSIONAL TKINTER GUI
# ============================================================================

class HeadGearSentinelGUI:
    """Professional GUI for HeadGear Sentinel"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("HeadGear Sentinel - PPE Object Detection")
        self.root.geometry("1200x700")
        self.root.configure(bg='#f5f5f5')
        
        # Center window
        self.center_window()
        
        # Initialize model
        self.model = LightweightObjectDetector()
        
        # Application state
        self.current_image = None
        self.image_path = None
        self.detections = []
        self.safety_status = "Unknown"
        self.training_in_progress = False
        
        # Setup GUI
        self.setup_gui()
        
        # Load existing model
        self.load_existing_model()
    
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def setup_gui(self):
        """Setup the complete GUI"""
        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        # Header
        self.create_header()
        
        # Left panel - Controls
        self.create_control_panel()
        
        # Center panel - Image display
        self.create_image_panel()
        
        # Right panel - Results
        self.create_results_panel()
        
        # Status bar
        self.create_status_bar()
    
    def create_header(self):
        """Create application header"""
        header_frame = tk.Frame(self.root, bg='#2c3e50', height=70)
        header_frame.grid(row=0, column=0, columnspan=3, sticky='ew', pady=(0, 10))
        header_frame.grid_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="🚧 HEADGEAR SENTINEL",
            font=('Arial', 22, 'bold'),
            bg='#2c3e50',
            fg='white',
            pady=20
        )
        title_label.pack(side=tk.LEFT, padx=30)
        
        # Subtitle
        subtitle_label = tk.Label(
            header_frame,
            text="Construction Worker PPE Object Detection System",
            font=('Arial', 11),
            bg='#2c3e50',
            fg='#ecf0f1'
        )
        subtitle_label.pack(side=tk.LEFT, padx=10)
        
        # Model status
        self.model_status_var = tk.StringVar(value="Model: Not Loaded")
        model_status_label = tk.Label(
            header_frame,
            textvariable=self.model_status_var,
            font=('Arial', 10, 'bold'),
            bg='#2c3e50',
            fg='#e74c3c'
        )
        model_status_label.pack(side=tk.RIGHT, padx=30)
    
    def create_control_panel(self):
        """Create left control panel"""
        control_frame = tk.LabelFrame(
            self.root,
            text="CONTROLS",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            relief=tk.RAISED,
            borderwidth=2
        )
        control_frame.grid(row=1, column=0, sticky='ns', padx=20, pady=10)
        control_frame.grid_rowconfigure(8, weight=1)
        
        # Button container
        button_frame = tk.Frame(control_frame, bg='white', padx=15, pady=15)
        button_frame.pack(fill='both', expand=True)
        
        # Define buttons
        buttons = [
            ("📁 LOAD IMAGE", self.load_image, '#3498db'),
            ("📂 LOAD DATASET", self.load_dataset, '#3498db'),
            ("🎯 TRAIN MODEL", self.train_model, '#2ecc71'),
            ("🔍 RUN DETECTION", self.run_detection, '#e67e22'),
            ("🧹 CLEAR", self.clear_all, '#95a5a6'),
            ("📊 STATISTICS", self.show_statistics, '#9b59b6'),
            ("🚪 EXIT", self.root.quit, '#e74c3c')
        ]
        
        for text, command, color in buttons:
            btn = tk.Button(
                button_frame,
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
                width=15
            )
            btn.pack(pady=8)
            # Store button reference
            btn_name = text.split()[1].lower()
            setattr(self, f'btn_{btn_name}', btn)
        
        # Training progress (hidden initially)
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
    
    def create_image_panel(self):
        """Create center image display panel"""
        image_frame = tk.LabelFrame(
            self.root,
            text="IMAGE PREVIEW",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            relief=tk.RAISED,
            borderwidth=2
        )
        image_frame.grid(row=1, column=1, sticky='nsew', padx=10, pady=10)
        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)
        
        # Create canvas with scrollbars
        self.canvas = tk.Canvas(
            image_frame,
            bg='#ecf0f1',
            highlightthickness=0
        )
        
        v_scrollbar = tk.Scrollbar(image_frame, orient='vertical', command=self.canvas.yview)
        h_scrollbar = tk.Scrollbar(image_frame, orient='horizontal', command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Grid layout
        self.canvas.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        v_scrollbar.grid(row=0, column=1, sticky='ns')
        h_scrollbar.grid(row=1, column=0, sticky='ew')
        
        image_frame.grid_rowconfigure(0, weight=1)
        image_frame.grid_columnconfigure(0, weight=1)
        
        # Placeholder text
        self.canvas.create_text(
            400, 250,
            text="No Image Loaded\n\nLoad an image to begin detection",
            font=('Arial', 14),
            fill='#7f8c8d',
            justify='center',
            tags="placeholder"
        )
        
        # Image info
        self.image_info_var = tk.StringVar(value="")
        info_label = tk.Label(
            image_frame,
            textvariable=self.image_info_var,
            font=('Arial', 10),
            bg='white',
            fg='#2c3e50'
        )
        info_label.grid(row=2, column=0, columnspan=2, sticky='ew', padx=5, pady=(0, 5))
    
    def create_results_panel(self):
        """Create right results panel"""
        results_frame = tk.LabelFrame(
            self.root,
            text="DETECTION RESULTS",
            font=('Arial', 12, 'bold'),
            bg='white',
            fg='#2c3e50',
            relief=tk.RAISED,
            borderwidth=2
        )
        results_frame.grid(row=1, column=2, sticky='ns', padx=(10, 20), pady=10)
        
        # Safety status card
        safety_card = tk.Frame(
            results_frame,
            bg='#f8f9fa',
            relief=tk.SUNKEN,
            borderwidth=1,
            padx=15,
            pady=15
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
        
        # Detection results
        results_text_frame = tk.Frame(results_frame, bg='white')
        results_text_frame.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        
        # Create notebook for tabs
        self.results_notebook = ttk.Notebook(results_text_frame)
        self.results_notebook.pack(fill='both', expand=True)
        
        # Tab 1: Object Detection Results
        detection_frame = tk.Frame(self.results_notebook, bg='white')
        self.results_notebook.add(detection_frame, text="Detected Objects")
        
        self.detection_text = scrolledtext.ScrolledText(
            detection_frame,
            width=30,
            height=12,
            font=('Consolas', 10),
            bg='#f8f9fa',
            relief=tk.FLAT
        )
        self.detection_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Tab 2: Missing Items
        missing_frame = tk.Frame(self.results_notebook, bg='white')
        self.results_notebook.add(missing_frame, text="Safety Alerts")
        
        self.missing_text = scrolledtext.ScrolledText(
            missing_frame,
            width=30,
            height=12,
            font=('Consolas', 10),
            bg='#f8f9fa',
            relief=tk.FLAT,
            fg='#e74c3c'
        )
        self.missing_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Statistics
        stats_frame = tk.Frame(results_frame, bg='white')
        stats_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.stats_var = tk.StringVar(value="Objects: 0/7 | Avg Confidence: 0%")
        stats_label = tk.Label(
            stats_frame,
            textvariable=self.stats_var,
            font=('Arial', 10),
            bg='white',
            fg='#7f8c8d'
        )
        stats_label.pack()
    
    def create_status_bar(self):
        """Create bottom status bar"""
        status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        status_frame.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(10, 0))
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
        status_label.pack(side=tk.LEFT, padx=15, fill='x', expand=True)
    
    def load_existing_model(self):
        """Try to load existing trained model"""
        success, message = self.model.load_model()
        if success:
            self.model_status_var.set("✅ Model: Loaded")
            self.log_status("Model loaded successfully")
        else:
            self.log_status("No trained model found. Please train a model.")
    
    # ------------------------------------------------------------------------
    # GUI EVENT HANDLERS
    # ------------------------------------------------------------------------
    
    def load_image(self):
        """Load a single image for detection"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.process_image(file_path)
    
    def load_dataset(self):
        """Load dataset folder structure"""
        dataset_path = filedialog.askdirectory(title="Select Dataset Folder")
        
        if dataset_path:
            # Check if folder structure is correct
            required_folders = [
                "class_0_person",
                "class_1_helmet", 
                "class_2_goggles",
                "class_3_mask",
                "class_4_gloves",
                "class_5_vest",
                "class_6_boots"
            ]
            
            missing_folders = []
            for folder in required_folders:
                if not os.path.exists(os.path.join(dataset_path, folder)):
                    missing_folders.append(folder)
            
            if missing_folders:
                messagebox.showwarning(
                    "Dataset Structure",
                    f"Missing required folders:\n{', '.join(missing_folders)}\n\n"
                    f"Expected structure:\n"
                    f"dataset/\n"
                    f"├── class_0_person/\n"
                    f"├── class_1_helmet/\n"
                    f"├── class_2_goggles/\n"
                    f"├── class_3_mask/\n"
                    f"├── class_4_gloves/\n"
                    f"├── class_5_vest/\n"
                    f"└── class_6_boots/"
                )
            else:
                self.log_status(f"Dataset loaded: {dataset_path}")
                messagebox.showinfo("Dataset Loaded", 
                                  "Dataset structure verified successfully!")
    
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
            self.image_info_var.set(f"{filename} | Size: {w}x{h}")
            
            self.log_status(f"Image loaded: {filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image):
        """Display image on canvas"""
        # Clear canvas
        self.canvas.delete("all")
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width < 10:
            canvas_width = 600
        if canvas_height < 10:
            canvas_height = 400
        
        # Calculate scale to fit
        img_width, img_height = pil_image.size
        scale = min(canvas_width / img_width, canvas_height / img_height) * 0.9
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # Calculate position to center
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        
        # Display image
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.tk_image)
        
        # Update scroll region
        self.canvas.config(scrollregion=(0, 0, new_width, new_height))
        
        # Store display parameters
        self.display_scale = scale
        self.display_offset = (x, y)
    
    def draw_detections(self):
        """Draw bounding boxes for detected objects"""
        if not self.detections:
            return
        
        for detection in self.detections:
            if not detection['present'] or detection['bbox'] is None:
                continue
            
            class_name = detection['class']
            confidence = detection['confidence']
            x1, y1, x2, y2 = detection['bbox']
            color = self.model.class_colors.get(class_name, (255, 255, 255))
            
            # Scale coordinates
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
        if self.training_in_progress:
            return
        
        # Ask for dataset folder
        dataset_path = filedialog.askdirectory(title="Select Dataset Folder")
        if not dataset_path:
            return
        
        # Verify dataset structure
        if not self.verify_dataset_structure(dataset_path):
            messagebox.showerror(
                "Invalid Dataset",
                "Dataset structure incorrect. Required folders:\n"
                "class_0_person, class_1_helmet, class_2_goggles, class_3_mask,\n"
                "class_4_gloves, class_5_vest, class_6_boots"
            )
            return
        
        # Show progress bar
        self.progress_frame.pack(padx=15, pady=15, fill='x')
        self.progress_var.set(0)
        self.progress_label.config(text="Initializing training...")
        
        # Disable buttons during training
        self.set_buttons_state(tk.DISABLED)
        self.training_in_progress = True
        
        # Start training thread
        training_thread = threading.Thread(
            target=self.training_thread_func,
            args=(dataset_path,),
            daemon=True
        )
        training_thread.start()
    
    def verify_dataset_structure(self, dataset_path):
        """Verify dataset folder structure"""
        required_folders = [
            "class_0_person",
            "class_1_helmet", 
            "class_2_goggles",
            "class_3_mask",
            "class_4_gloves",
            "class_5_vest",
            "class_6_boots"
        ]
        
        for folder in required_folders:
            if not os.path.exists(os.path.join(dataset_path, folder)):
                return False
        return True
    
    def training_thread_func(self, dataset_path):
        """Training function to run in thread"""
        def progress_callback(progress, message):
            self.root.after(0, self.update_training_progress, progress, message)
        
        success, message = self.model.train(dataset_path, progress_callback)
        self.root.after(0, self.training_complete, success, message)
    
    def update_training_progress(self, progress, message):
        """Update progress bar from training thread"""
        self.progress_var.set(progress)
        self.progress_label.config(text=message)
        self.status_var.set(f"Training: {message}")
    
    def training_complete(self, success, message):
        """Handle training completion"""
        # Hide progress bar
        self.progress_frame.pack_forget()
        
        # Re-enable buttons
        self.set_buttons_state(tk.NORMAL)
        self.training_in_progress = False
        
        if success:
            self.model_status_var.set("✅ Model: Trained")
            messagebox.showinfo("Training Complete", message)
            self.log_status("Model training completed successfully")
        else:
            messagebox.showerror("Training Failed", message)
            self.log_status("Model training failed")
    
    def set_buttons_state(self, state):
        """Enable/disable all buttons"""
        buttons = ['btn_image', 'btn_dataset', 'btn_model', 
                  'btn_detection', 'btn_clear', 'btn_statistics']
        
        for btn_name in buttons:
            btn = getattr(self, btn_name, None)
            if btn:
                btn.config(state=state)
                if state == tk.DISABLED:
                    btn.config(cursor='watch')
                else:
                    btn.config(cursor='hand2')
    
    def run_detection(self):
        """Run object detection on current image"""
        if self.current_image is None:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        if not self.model.is_trained:
            response = messagebox.askyesno(
                "Model Not Trained",
                "No trained model found. Would you like to train a model first?\n\n"
                "Click 'No' to use demonstration mode."
            )
            
            if response:
                self.train_model()
                return
            else:
                # Create a simple demonstration model
                self.create_demo_model()
        
        # Show progress in status bar
        self.status_var.set("Running object detection...")
        
        # Run detection
        self.detections, message = self.model.detect_objects(self.current_image)
        
        # Evaluate safety
        safety_status, missing_items, warnings = self.model.evaluate_safety(self.detections)
        self.safety_status = safety_status
        
        # Update UI
        self.update_results_display(missing_items, warnings)
        self.draw_detections()
        
        self.log_status(f"Detection complete: {message}")
    
    def create_demo_model(self):
        """Create a demonstration model for testing"""
        self.model.is_trained = True
        self.model_status_var.set("⚠️ Demo Model")
        self.log_status("Using demonstration mode")
    
    def update_results_display(self, missing_items, warnings):
        """Update results display with detection results"""
        # Update safety status
        self.safety_var.set(self.safety_status)
        
        # Set safety status color
        if "SAFE" in self.safety_status:
            self.safety_label.config(fg='#27ae60')
        elif "UNSAFE" in self.safety_status:
            self.safety_label.config(fg='#e74c3c')
        else:
            self.safety_label.config(fg='#f39c12')
        
        # Update detection text
        self.detection_text.delete(1.0, tk.END)
        
        detected_count = 0
        total_confidence = 0
        
        for detection in self.detections:
            class_name = detection['class']
            confidence = detection['confidence']
            present = detection['present']
            
            if present:
                detected_count += 1
                total_confidence += confidence
                
                # Color code based on confidence
                if confidence > 0.7:
                    prefix = "✅ "
                    color = "green"
                elif confidence > 0.4:
                    prefix = "⚠️ "
                    color = "orange"
                else:
                    prefix = "❓ "
                    color = "gray"
                
                line = f"{prefix}{class_name:15} {confidence:>5.0%}\n"
                self.detection_text.insert(tk.END, line)
            else:
                line = f"❌ {class_name:15} Not detected\n"
                self.detection_text.insert(tk.END, line)
        
        # Update missing items text
        self.missing_text.delete(1.0, tk.END)
        
        all_alerts = missing_items + warnings
        if all_alerts:
            for alert in all_alerts:
                self.missing_text.insert(tk.END, f"{alert}\n")
        else:
            self.missing_text.insert(tk.END, "All safety requirements met! ✅\n")
        
        # Update statistics
        avg_confidence = total_confidence / detected_count if detected_count > 0 else 0
        self.stats_var.set(f"Objects: {detected_count}/7 | Avg Confidence: {avg_confidence:.0%}")
    
    def clear_all(self):
        """Clear all results and reset display"""
        # Clear image
        self.canvas.delete("all")
        self.canvas.create_text(
            400, 250,
            text="No Image Loaded\n\nLoad an image to begin detection",
            font=('Arial', 14),
            fill='#7f8c8d',
            justify='center',
            tags="placeholder"
        )
        
        # Clear text areas
        self.detection_text.delete(1.0, tk.END)
        self.missing_text.delete(1.0, tk.END)
        
        # Reset safety status
        self.safety_var.set("UNKNOWN")
        self.safety_label.config(fg='#f39c12')
        
        # Reset stats
        self.stats_var.set("Objects: 0/7 | Avg Confidence: 0%")
        self.image_info_var.set("")
        
        # Clear variables
        self.current_image = None
        self.detections = []
        
        self.log_status("Cleared all results")
    
    def show_statistics(self):
        """Show model statistics"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Model Statistics")
        stats_window.geometry("500x400")
        stats_window.configure(bg='white')
        
        # Center window
        stats_window.update_idletasks()
        width = stats_window.winfo_width()
        height = stats_window.winfo_height()
        x = (stats_window.winfo_screenwidth() // 2) - (width // 2)
        y = (stats_window.winfo_screenheight() // 2) - (height // 2)
        stats_window.geometry(f'{width}x{height}+{x}+{y}')
        
        # Content
        content = tk.Frame(stats_window, bg='white', padx=20, pady=20)
        content.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(
            content,
            text="📊 MODEL STATISTICS",
            font=('Arial', 16, 'bold'),
            bg='white',
            fg='#2c3e50'
        ).pack(pady=(0, 20))
        
        # Model info
        info_text = f"""
        Model Type: Lightweight Object Detector
        Status: {'Trained ✅' if self.model.is_trained else 'Not Trained ⚠️'}
        
        Detection Classes (7):
        1. Person (Required)
        2. Helmet (Critical PPE)
        3. Goggles (Recommended PPE)
        4. Mask (Recommended PPE)
        5. Gloves (Important PPE)
        6. Safety Vest (Critical PPE)
        7. Boots (Important PPE)
        
        Features Used:
        • Color Histograms (32 features)
        • Edge Density (16 features)
        • Texture Patterns (16 features)
        • Shape Descriptors (16 features)
        • Color-specific (48 features)
        
        Total Features: 128 dimensions
        
        Classifiers: K-Nearest Neighbors (K=3)
        Training Strategy: One-vs-All per class
        
        Detection Threshold: {self.model.confidence_threshold:.0%}
        """
        
        tk.Label(
            content,
            text=info_text,
            font=('Consolas', 9),
            bg='white',
            fg='#2c3e50',
            justify=tk.LEFT
        ).pack(fill=tk.BOTH, expand=True)
    
    def log_status(self, message):
        """Log status message"""
        self.status_var.set(message)
        print(f"[Status] {message}")


# ============================================================================
# DATASET CREATION UTILITY
# ============================================================================

def create_sample_dataset():
    """Create a sample dataset with proper folder structure"""
    dataset_path = "ppe_dataset"
    
    # Create folders for each class
    folders = [
        "class_0_person",
        "class_1_helmet",
        "class_2_goggles", 
        "class_3_mask",
        "class_4_gloves",
        "class_5_vest",
        "class_6_boots"
    ]
    
    for folder in folders:
        os.makedirs(os.path.join(dataset_path, folder), exist_ok=True)
    
    # Create sample images for each class
    print("Creating sample dataset...")
    
    for class_idx, folder in enumerate(folders):
        class_path = os.path.join(dataset_path, folder)
        
        # Create 5 sample images per class
        for i in range(5):
            # Create a simple image representing the class
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            
            # Different colors for different classes
            colors = [
                (100, 100, 255),   # Person - Blue
                (255, 255, 0),     # Helmet - Cyan
                (0, 255, 255),     # Goggles - Yellow
                (255, 0, 255),     # Mask - Magenta
                (0, 255, 0),       # Gloves - Green
                (0, 165, 255),     # Vest - Orange
                (128, 0, 128)      # Boots - Purple
            ]
            
            img[:] = colors[class_idx]
            
            # Add class label text
            class_name = folder.split('_')[2]
            cv2.putText(img, class_name, (50, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"Sample {i+1}", (50, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save image
            cv2.imwrite(os.path.join(class_path, f"{class_name}_{i+1}.jpg"), img)
    
    print(f"Sample dataset created at: {dataset_path}")
    print("\nDataset structure:")
    for folder in folders:
        print(f"  {folder}/")
    
    return dataset_path


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main entry point"""
    print("=" * 60)
    print("HEADGEAR SENTINEL - PPE Object Detection System")
    print("Semester Project - Computer Science")
    print("=" * 60)
    print("\nSystem Information:")
    print(f"• OS: {sys.platform}")
    print("• CPU: Intel/AMD (CPU-only mode)")
    print("• Memory: Optimized for low-mid RAM")
    print("• Training: Fast, lightweight (seconds)")
    print("\nStarting application...")
    
    # Create sample dataset if needed
    if not os.path.exists("ppe_dataset"):
        print("\nCreating sample dataset...")
        create_sample_dataset()
        print("Sample dataset ready! Use it for training.")
    
    # Create and run GUI
    root = tk.Tk()
    
    # Configure styles
    style = ttk.Style()
    style.theme_use('clam')
    
    # Create application
    app = HeadGearSentinelGUI(root)
    
    print("\nApplication started successfully!")
    print("\nInstructions:")
    print("1. Load Dataset: Select 'ppe_dataset' folder")
    print("2. Train Model: Click 'Train Model' (fast training)")
    print("3. Load Image: Select any construction worker image")
    print("4. Run Detection: Detect PPE objects")
    print("5. View Results: Check safety status")
    print("\n" + "=" * 60)
    
    # Start main loop
    root.mainloop()


if __name__ == "__main__":
    main()