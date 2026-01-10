#!/usr/bin/env python3
"""
HeadGear Sentinel - PPE Object Detection for Construction Workers
A complete semester project with dataset loading, model training, and GUI detection.
Single-file Python application using PyQt5 and TensorFlow/Keras.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import threading

import cv2
from PIL import Image
import numpy as np

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QTabWidget, QLabel, QPushButton, 
                             QLineEdit, QSpinBox, QDoubleSpinBox, QFileDialog,
                             QProgressBar, QGridLayout, QFrame, QScrollArea,
                             QComboBox, QMessageBox)
from PyQt5.QtGui import QPixmap, QFont, QColor, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QRect
from PyQt5.QtCore import Qt, QTimer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# ============================================================================
# Global Configuration
# ============================================================================

CLASSES = ['Person', 'Helmet', 'Goggles', 'Mask', 'Gloves', 'Safety Vest', 'Boots']
REQUIRED_PPE = ['Helmet', 'Gloves', 'Safety Vest', 'Boots']
IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5


# ============================================================================
# Dataset Manager
# ============================================================================

class DatasetManager:
    """Handles dataset loading and preparation."""
    
    def __init__(self):
        self.dataset_path = None
        self.class_images = defaultdict(list)
        self.total_images = 0
        
    def load_dataset(self, folder_path):
        """Load dataset from folder structure: dataset/class_name/images."""
        self.class_images.clear()
        self.dataset_path = folder_path
        
        for class_name in CLASSES:
            class_path = os.path.join(folder_path, class_name)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
                self.class_images[class_name] = [os.path.join(class_path, img) for img in images]
                self.total_images += len(images)
        
        return self.class_images, self.total_images
    
    def get_statistics(self):
        """Return dataset statistics."""
        stats = {
            'total_classes': len(self.class_images),
            'total_images': self.total_images,
            'classes': {}
        }
        for class_name, images in self.class_images.items():
            stats['classes'][class_name] = len(images)
        return stats
    
    def prepare_training_data(self):
        """Prepare images and labels for training."""
        X = []
        y = []
        
        for class_idx, class_name in enumerate(CLASSES):
            if class_name in self.class_images:
                for img_path in self.class_images[class_name]:
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                            img = img.astype('float32') / 255.0
                            X.append(img)
                            y.append(class_idx)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        X = np.array(X)
        y = keras.utils.to_categorical(y, num_classes=len(CLASSES))
        return X, y


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Handles model creation, training, and inference."""
    
    def __init__(self):
        self.model = None
        self.training_history = None
        
    def build_model(self):
        """Build CNN model for PPE classification."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(CLASSES), activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model with specified learning rate."""
        if self.model is None:
            self.model = self.build_model()
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=16):
        """Train the model."""
        if self.model is None:
            self.compile_model()
        
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return self.training_history
    
    def predict(self, image_path):
        """Predict PPE classes in an image."""
        if self.model is None:
            return None, None
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, None
            
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_normalized = img_resized.astype('float32') / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            predictions = self.model.predict(img_batch, verbose=0)
            confidences = predictions[0]
            
            detected = {}
            for class_idx, class_name in enumerate(CLASSES):
                detected[class_name] = confidences[class_idx] >= CONFIDENCE_THRESHOLD
            
            avg_confidence = np.max(confidences)
            return detected, avg_confidence
        
        except Exception as e:
            print(f"Error predicting: {e}")
            return None, None
    
    def save_model(self, path):
        """Save trained model."""
        if self.model:
            self.model.save(path)
    
    def load_model(self, path):
        """Load trained model."""
        try:
            self.model = keras.models.load_model(path)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


# ============================================================================
# Worker Threads
# ============================================================================

class TrainingWorker(QThread):
    """Background thread for model training."""
    
    progress_update = pyqtSignal(int, str)
    training_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, dataset_manager, model_manager, epochs, batch_size, lr, train_split):
        super().__init__()
        self.dataset_manager = dataset_manager
        self.model_manager = model_manager
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.train_split = train_split / 100.0
    
    def run(self):
        try:
            self.progress_update.emit(10, "Preparing training data...")
            X, y = self.dataset_manager.prepare_training_data()
            
            if len(X) == 0:
                self.error_occurred.emit("No images found in dataset")
                return
            
            self.progress_update.emit(20, f"Splitting data ({int(self.train_split*100)}% train)...")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=1-self.train_split, random_state=42
            )
            
            self.progress_update.emit(30, "Building model...")
            self.model_manager.compile_model(learning_rate=self.lr)
            
            self.progress_update.emit(40, f"Training for {self.epochs} epochs...")
            history = self.model_manager.train(
                X_train, y_train, X_val, y_val,
                epochs=self.epochs,
                batch_size=self.batch_size
            )
            
            self.progress_update.emit(95, "Finalizing training...")
            
            metrics = {
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.lr,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'final_accuracy': float(history.history['accuracy'][-1]),
                'final_val_accuracy': float(history.history['val_accuracy'][-1]),
                'final_loss': float(history.history['loss'][-1])
            }
            
            self.progress_update.emit(100, "Training complete!")
            self.training_complete.emit(metrics)
        
        except Exception as e:
            self.error_occurred.emit(f"Training error: {str(e)}")


class DetectionWorker(QThread):
    """Background thread for image detection."""
    
    progress_update = pyqtSignal(int)
    detection_complete = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model_manager, image_paths):
        super().__init__()
        self.model_manager = model_manager
        self.image_paths = image_paths
    
    def run(self):
        try:
            results = []
            total = len(self.image_paths)
            
            for idx, img_path in enumerate(self.image_paths):
                detected, confidence = self.model_manager.predict(img_path)
                
                if detected:
                    safety_status = self._check_safety(detected)
                    results.append({
                        'image': os.path.basename(img_path),
                        'path': img_path,
                        'detected': detected,
                        'safety_status': safety_status,
                        'confidence': float(confidence)
                    })
                
                progress = int((idx + 1) / total * 100)
                self.progress_update.emit(progress)
            
            self.detection_complete.emit(results)
        
        except Exception as e:
            self.error_occurred.emit(f"Detection error: {str(e)}")
    
    def _check_safety(self, detected):
        """Determine safety status based on detected PPE."""
        missing_ppe = [ppe for ppe in REQUIRED_PPE if not detected.get(ppe, False)]
        if len(missing_ppe) == 0:
            return 'SAFE'
        elif len(missing_ppe) == 1:
            return 'WARNING'
        else:
            return 'UNSAFE'


# ============================================================================
# Main GUI Application
# ============================================================================

class HeadGearSentinel(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle('HeadGear Sentinel - PPE Detection System')
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(self._get_stylesheet())
        
        # Managers
        self.dataset_manager = DatasetManager()
        self.model_manager = ModelManager()
        
        # State
        self.dataset_loaded = False
        self.model_trained = False
        self.current_images = []
        self.training_worker = None
        self.detection_worker = None
        
        # UI
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup main UI components."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        
        # Header
        header = self._create_header()
        main_layout.addWidget(header)
        
        # Tabs
        tabs = QTabWidget()
        tabs.addTab(self._create_training_tab(), "Dataset & Training")
        tabs.addTab(self._create_detection_tab(), "Detection & Results")
        tabs.addTab(self._create_info_tab(), "Information")
        main_layout.addWidget(tabs)
        
        central_widget.setLayout(main_layout)
    
    def _create_header(self):
        """Create header section."""
        header = QFrame()
        header.setStyleSheet("background: qlineargradient(x1:0, y1:0, x2:1, y2:1, "
                           "stop:0 #217D8D, stop:1 #1D748D); color: white; padding: 20px; border-radius: 8px;")
        layout = QVBoxLayout()
        
        title = QLabel("🛡️ HeadGear Sentinel")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setStyleSheet("color: white;")
        
        subtitle = QLabel("PPE Object Detection System for Construction Workers")
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.9);")
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        header.setLayout(layout)
        return header
    
    def _create_training_tab(self):
        """Create training/dataset tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Dataset Configuration
        layout.addWidget(self._create_section("📁 Dataset Configuration"))
        
        dataset_layout = QHBoxLayout()
        dataset_btn = QPushButton("📂 Load Dataset Folder")
        dataset_btn.clicked.connect(self._load_dataset)
        dataset_layout.addWidget(dataset_btn)
        self.dataset_status = QLabel("No dataset loaded")
        dataset_layout.addWidget(self.dataset_status)
        layout.addLayout(dataset_layout)
        
        # Training Configuration
        layout.addWidget(self._create_section("⚙️ Training Configuration"))
        
        config_layout = QGridLayout()
        
        config_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setValue(10)
        self.epochs_spin.setRange(1, 100)
        config_layout.addWidget(self.epochs_spin, 0, 1)
        
        config_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_spin = QSpinBox()
        self.batch_spin.setValue(16)
        self.batch_spin.setRange(1, 128)
        config_layout.addWidget(self.batch_spin, 0, 3)
        
        config_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setSingleStep(0.0001)
        config_layout.addWidget(self.lr_spin, 1, 1)
        
        config_layout.addWidget(QLabel("Train Split (%):"), 1, 2)
        self.split_spin = QSpinBox()
        self.split_spin.setValue(80)
        self.split_spin.setRange(50, 95)
        config_layout.addWidget(self.split_spin, 1, 3)
        
        layout.addLayout(config_layout)
        
        # Train Button
        self.train_btn = QPushButton("🚀 Start Training")
        self.train_btn.setStyleSheet("background: #217D8D; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
        self.train_btn.clicked.connect(self._start_training)
        layout.addWidget(self.train_btn)
        
        # Progress
        layout.addWidget(self._create_section("Training Progress"))
        self.train_progress = QProgressBar()
        self.train_progress.setVisible(False)
        layout.addWidget(self.train_progress)
        self.train_status = QLabel("")
        layout.addWidget(self.train_status)
        
        # Statistics
        layout.addWidget(self._create_section("📊 Dataset Statistics"))
        self.dataset_stats = QLabel("No dataset loaded")
        layout.addWidget(self.dataset_stats)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def _create_detection_tab(self):
        """Create detection tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Image Input
        layout.addWidget(self._create_section("🖼️ Image Input"))
        
        input_layout = QHBoxLayout()
        single_btn = QPushButton("📷 Load Single Image")
        single_btn.clicked.connect(self._load_single_image)
        input_layout.addWidget(single_btn)
        
        folder_btn = QPushButton("📁 Load Image Folder")
        folder_btn.clicked.connect(self._load_image_folder)
        input_layout.addWidget(folder_btn)
        
        self.images_status = QLabel("No images loaded")
        input_layout.addWidget(self.images_status)
        layout.addLayout(input_layout)
        
        # Detection Button
        layout.addWidget(self._create_section("🔍 Run Detection"))
        self.detect_btn = QPushButton("🚀 Run Detection")
        self.detect_btn.setStyleSheet("background: #217D8D; color: white; padding: 10px; font-weight: bold; border-radius: 5px;")
        self.detect_btn.clicked.connect(self._run_detection)
        layout.addWidget(self.detect_btn)
        
        # Progress
        self.detect_progress = QProgressBar()
        self.detect_progress.setVisible(False)
        layout.addWidget(self.detect_progress)
        
        # Results
        layout.addWidget(self._create_section("🛡️ Safety Status"))
        self.safety_status = QLabel("Run detection to see results")
        layout.addWidget(self.safety_status)
        
        layout.addWidget(self._create_section("📋 Detection Results"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout()
        self.results_container.setLayout(self.results_layout)
        scroll.setWidget(self.results_container)
        layout.addWidget(scroll)
        
        layout.addWidget(self._create_section("📊 Statistics"))
        self.detection_stats = QLabel("No results yet")
        layout.addWidget(self.detection_stats)
        
        widget.setLayout(layout)
        return widget
    
    def _create_info_tab(self):
        """Create information tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        layout.addWidget(self._create_section("ℹ️ About HeadGear Sentinel"))
        
        info = QLabel(
            "<b>A comprehensive PPE object detection system for construction worker safety.</b><br><br>"
            "<b>Features:</b><br>"
            "• Real dataset training with customizable parameters<br>"
            "• Multi-class PPE detection (7 classes)<br>"
            "• Single image and batch processing<br>"
            "• Safety status reporting<br>"
            "• Detection confidence scoring<br><br>"
            "<b>Supported PPE Classes:</b><br>"
            "👤 Person | 🧢 Helmet | 👓 Goggles | 😷 Mask | 🧤 Gloves | 🦺 Safety Vest | 👢 Boots<br><br>"
            "<b>Dataset Structure:</b><br>"
            "<code>dataset/<br>"
            "├── Person/<br>"
            "├── Helmet/<br>"
            "├── Goggles/<br>"
            "├── Mask/<br>"
            "├── Gloves/<br>"
            "├── Safety Vest/<br>"
            "└── Boots/</code><br><br>"
            "<b>Safety Rules:</b><br>"
            "• SAFE: All required PPE present (Helmet, Gloves, Safety Vest, Boots)<br>"
            "• WARNING: Missing 1 required item<br>"
            "• UNSAFE: Missing 2+ required items"
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def _create_section(self, title):
        """Create section header."""
        label = QLabel(title)
        label.setFont(QFont("Arial", 12, QFont.Bold))
        label.setStyleSheet("color: #217D8D; margin-top: 12px; margin-bottom: 8px;")
        return label
    
    def _load_dataset(self):
        """Load dataset folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            try:
                classes, total = self.dataset_manager.load_dataset(folder)
                self.dataset_loaded = True
                self.dataset_status.setText(f"✓ Dataset loaded: {len(classes)} classes, {total} images")
                self.dataset_status.setStyleSheet("color: green;")
                
                stats = self.dataset_manager.get_statistics()
                stats_text = f"<b>Total:</b> {stats['total_classes']} classes, {stats['total_images']} images<br>"
                for cls_name, count in stats['classes'].items():
                    stats_text += f"{cls_name}: {count} images<br>"
                self.dataset_stats.setText(stats_text)
                
                self._show_info("Dataset loaded successfully!")
            except Exception as e:
                self._show_error(f"Error loading dataset: {str(e)}")
    
    def _start_training(self):
        """Start model training."""
        if not self.dataset_loaded:
            self._show_error("Please load a dataset first")
            return
        
        self.train_btn.setEnabled(False)
        self.train_progress.setVisible(True)
        self.train_progress.setValue(0)
        self.train_status.setText("Initializing training...")
        
        self.training_worker = TrainingWorker(
            self.dataset_manager,
            self.model_manager,
            self.epochs_spin.value(),
            self.batch_spin.value(),
            self.lr_spin.value(),
            self.split_spin.value()
        )
        self.training_worker.progress_update.connect(self._update_training_progress)
        self.training_worker.training_complete.connect(self._training_completed)
        self.training_worker.error_occurred.connect(self._training_error)
        self.training_worker.start()
    
    def _update_training_progress(self, progress, status):
        """Update training progress."""
        self.train_progress.setValue(progress)
        self.train_status.setText(status)
    
    def _training_completed(self, metrics):
        """Handle training completion."""
        self.model_trained = True
        self.train_btn.setEnabled(True)
        self.train_progress.setValue(100)
        
        metrics_text = (
            f"<b>Training Complete!</b><br>"
            f"Epochs: {metrics['epochs']}<br>"
            f"Final Accuracy: {metrics['final_accuracy']:.4f}<br>"
            f"Final Val Accuracy: {metrics['final_val_accuracy']:.4f}<br>"
            f"Train Samples: {metrics['train_samples']}<br>"
            f"Val Samples: {metrics['val_samples']}"
        )
        self.train_status.setText(metrics_text)
        self._show_info("Model trained successfully! Ready for detection.")
    
    def _training_error(self, error):
        """Handle training error."""
        self.train_btn.setEnabled(True)
        self.train_progress.setVisible(False)
        self._show_error(error)
    
    def _load_single_image(self):
        """Load single image."""
        file, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.jpeg *.png *.bmp *.gif)")
        if file:
            self.current_images = [file]
            self.images_status.setText(f"✓ Loaded: {os.path.basename(file)}")
            self.images_status.setStyleSheet("color: green;")
    
    def _load_image_folder(self):
        """Load image folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder:
            images = [os.path.join(folder, f) for f in os.listdir(folder)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            if images:
                self.current_images = images
                self.images_status.setText(f"✓ Loaded {len(images)} image(s)")
                self.images_status.setStyleSheet("color: green;")
            else:
                self._show_error("No images found in folder")
    
    def _run_detection(self):
        """Run detection on loaded images."""
        if not self.model_trained:
            self._show_error("Please train a model first")
            return
        if not self.current_images:
            self._show_error("Please load images first")
            return
        
        self.detect_btn.setEnabled(False)
        self.detect_progress.setVisible(True)
        self.detect_progress.setValue(0)
        
        self.detection_worker = DetectionWorker(self.model_manager, self.current_images)
        self.detection_worker.progress_update.connect(self.detect_progress.setValue)
        self.detection_worker.detection_complete.connect(self._display_results)
        self.detection_worker.error_occurred.connect(self._detection_error)
        self.detection_worker.start()
    
    def _display_results(self, results):
        """Display detection results."""
        self.detect_btn.setEnabled(True)
        self.detect_progress.setVisible(False)
        
        # Safety status
        all_safe = all(r['safety_status'] == 'SAFE' for r in results)
        any_unsafe = any(r['safety_status'] == 'UNSAFE' for r in results)
        
        if any_unsafe:
            status_text = "⚠️ <b style='color: red;'>UNSAFE - Workers missing critical PPE</b>"
        elif all_safe:
            status_text = "✅ <b style='color: green;'>ALL SAFE - All workers properly equipped</b>"
        else:
            status_text = "⚡ <b style='color: orange;'>WARNING - Some workers missing PPE items</b>"
        
        self.safety_status.setText(status_text)
        
        # Clear results layout
        for i in reversed(range(self.results_layout.count())):
            self.results_layout.itemAt(i).widget().setParent(None)
        
        # Display each result
        for result in results:
            result_frame = QFrame()
            result_frame.setStyleSheet("border: 1px solid #ddd; border-radius: 5px; padding: 10px; margin: 5px 0;")
            result_layout = QVBoxLayout()
            
            img_name = QLabel(f"<b>{result['image']}</b> - Status: {result['safety_status']}")
            result_layout.addWidget(img_name)
            
            detection_grid = QGridLayout()
            for idx, class_name in enumerate(CLASSES):
                detected = result['detected'].get(class_name, False)
                icon = "✓" if detected else "✗"
                color = "green" if detected else "red"
                label = QLabel(f"<span style='color: {color};'>{icon} {class_name}</span>")
                detection_grid.addWidget(label, idx // 4, idx % 4)
            
            result_layout.addLayout(detection_grid)
            conf_label = QLabel(f"Confidence: {result['confidence']:.2%}")
            result_layout.addWidget(conf_label)
            
            result_frame.setLayout(result_layout)
            self.results_layout.addWidget(result_frame)
        
        # Statistics
        helmet_count = sum(1 for r in results if r['detected'].get('Helmet', False))
        safe_count = sum(1 for r in results if r['safety_status'] == 'SAFE')
        avg_conf = np.mean([r['confidence'] for r in results]) * 100
        
        stats_text = (
            f"<b>Total Images:</b> {len(results)}<br>"
            f"<b>Workers with Helmets:</b> {helmet_count}/{len(results)}<br>"
            f"<b>Safe Workers:</b> {safe_count}/{len(results)}<br>"
            f"<b>Average Confidence:</b> {avg_conf:.1f}%"
        )
        self.detection_stats.setText(stats_text)
        self._show_info("Detection complete!")
    
    def _detection_error(self, error):
        """Handle detection error."""
        self.detect_btn.setEnabled(True)
        self.detect_progress.setVisible(False)
        self._show_error(error)
    
    def _show_info(self, message):
        """Show info message."""
        QMessageBox.information(self, "HeadGear Sentinel", message)
    
    def _show_error(self, message):
        """Show error message."""
        QMessageBox.critical(self, "Error", message)
    
    def _get_stylesheet(self):
        """Return application stylesheet."""
        return """
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
            }
            QTabBar::tab {
                padding: 8px 20px;
                background: #e8e8e8;
                border: 1px solid #ddd;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 3px solid #217D8D;
            }
            QPushButton {
                padding: 8px 16px;
                border-radius: 5px;
                border: 1px solid #ddd;
                background: #f0f0f0;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #e0e0e0;
            }
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 5px;
                background: #f0f0f0;
                height: 24px;
            }
            QProgressBar::chunk {
                background: #217D8D;
                border-radius: 3px;
            }
            QSpinBox, QDoubleSpinBox, QLineEdit {
                padding: 6px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background: white;
            }
            QLabel {
                color: #333;
            }
        """


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    window = HeadGearSentinel()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
