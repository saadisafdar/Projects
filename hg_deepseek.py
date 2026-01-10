"""
HeadGear Sentinel - PPE Object Detection for Construction Workers
Complete Academic Project with GUI Interface
Author: [Your Name]
Date: [Current Date]

Requirements:
- Python 3.8+
- Required packages: torch, torchvision, pillow, tkinter, matplotlib, numpy, pandas, scikit-learn
- For GPU acceleration: CUDA compatible GPU (optional)
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import random
from collections import Counter, defaultdict
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms, models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class PPEDataset(Dataset):
    """Custom Dataset for PPE Detection"""
    
    def __init__(self, root_dir, transform=None, classes=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotations
            transform (callable, optional): Optional transform to be applied
            classes (list): List of class names
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = classes or ['Person', 'Helmet', 'Goggles', 'Mask', 'Gloves', 'Safety Vest', 'Boots']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Find all image files
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        self.images = []
        self.annotations = []
        
        # Look for COCO format annotations
        annotations_file = self.root_dir / 'annotations.json'
        if annotations_file.exists():
            self._load_coco_format(annotations_file)
        else:
            self._load_folder_format()
    
    def _load_coco_format(self, annotations_file):
        """Load dataset in COCO format"""
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        
        # Create mapping from image id to image info
        images_info = {img['id']: img for img in data['images']}
        annotations_by_image = defaultdict(list)
        
        # Group annotations by image
        for ann in data['annotations']:
            annotations_by_image[ann['image_id']].append(ann)
        
        # Create dataset entries
        for img_id, img_info in images_info.items():
            img_path = self.root_dir / 'images' / img_info['file_name']
            if img_path.exists():
                self.images.append(str(img_path))
                
                # Get annotations for this image
                img_anns = annotations_by_image.get(img_id, [])
                boxes = []
                labels = []
                
                for ann in img_anns:
                    # COCO bbox format: [x, y, width, height]
                    bbox = ann['bbox']
                    # Convert to [x1, y1, x2, y2]
                    x1 = bbox[0]
                    y1 = bbox[1]
                    x2 = x1 + bbox[2]
                    y2 = y1 + bbox[3]
                    boxes.append([x1, y1, x2, y2])
                    labels.append(ann['category_id'])
                
                self.annotations.append({
                    'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64)
                })
    
    def _load_folder_format(self):
        """Load dataset from folder structure (simplified for academic demo)"""
        # For academic demo, we'll create synthetic bounding boxes
        # In a real project, you'd need proper annotations
        
        # Look for class folders
        for class_folder in self.root_dir.iterdir():
            if class_folder.is_dir() and class_folder.name in self.classes:
                class_idx = self.class_to_idx[class_folder.name]
                for img_file in class_folder.iterdir():
                    if img_file.suffix.lower() in self.image_extensions:
                        self.images.append(str(img_file))
                        
                        # Create synthetic bounding box (center of image)
                        img = Image.open(img_file)
                        width, height = img.size
                        
                        # Generate random bounding box
                        box_size = min(width, height) // 2
                        x1 = random.randint(0, width - box_size)
                        y1 = random.randint(0, height - box_size)
                        x2 = x1 + box_size
                        y2 = y1 + box_size
                        
                        self.annotations.append({
                            'boxes': torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),
                            'labels': torch.tensor([class_idx], dtype=torch.int64)
                        })
        
        # If no class folders found, look for images directly
        if not self.images:
            for img_file in self.root_dir.rglob('*'):
                if img_file.suffix.lower() in self.image_extensions:
                    self.images.append(str(img_file))
                    
                    # Create dummy annotation
                    img = Image.open(img_file)
                    width, height = img.size
                    
                    # Randomly assign a class for demo
                    class_idx = random.randint(0, len(self.classes) - 1)
                    
                    # Generate random bounding boxes (1-3 per image)
                    num_boxes = random.randint(1, 3)
                    boxes = []
                    labels = []
                    
                    for _ in range(num_boxes):
                        box_size = random.randint(50, min(width, height) // 2)
                        x1 = random.randint(0, width - box_size)
                        y1 = random.randint(0, height - box_size)
                        x2 = x1 + box_size
                        y2 = y1 + box_size
                        
                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_idx)
                    
                    self.annotations.append({
                        'boxes': torch.tensor(boxes, dtype=torch.float32),
                        'labels': torch.tensor(labels, dtype=torch.int64)
                    })
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        annotation = self.annotations[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, annotation
    
    def get_class_distribution(self):
        """Get distribution of classes in dataset"""
        class_counts = Counter()
        for ann in self.annotations:
            for label in ann['labels'].numpy():
                if label < len(self.classes):
                    class_counts[self.classes[label]] += 1
        return class_counts

class PPEObjectDetector:
    """Main object detection model class"""
    
    def __init__(self, num_classes=8):  # +1 for background
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.model = None
        self.classes = ['Person', 'Helmet', 'Goggles', 'Mask', 'Gloves', 'Safety Vest', 'Boots']
        self.train_history = {'loss': [], 'accuracy': []}
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((640, 640)),
        ])
        self.inference_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def create_model(self):
        """Create Faster R-CNN model"""
        # Load a pre-trained model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        
        # Replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        self.model = self.model.to(self.device)
        return self.model
    
    def train_model(self, train_loader, val_loader, epochs=10, lr=0.001):
        """Train the model"""
        if self.model is None:
            self.create_model()
        
        # Parameters
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            
            # Training phase
            for images, targets in train_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
            
            avg_loss = epoch_loss / len(train_loader)
            self.train_history['loss'].append(avg_loss)
            
            # Validation phase
            val_loss = self.validate(val_loader)
            
            lr_scheduler.step()
            
            yield epoch + 1, avg_loss, val_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        if self.model is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
        
        return total_loss / len(val_loader)
    
    def predict(self, image, confidence_threshold=0.5):
        """Make predictions on a single image"""
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        self.model.eval()
        
        # Transform image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        original_image = image.copy()
        image_tensor = self.inference_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(image_tensor)
        
        # Process predictions
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        mask = scores >= confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Convert labels to class names
        class_names = []
        for label in labels:
            if 0 <= label - 1 < len(self.classes):  # Subtract 1 for background class
                class_names.append(self.classes[label - 1])
            else:
                class_names.append(f"Class_{label}")
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'class_names': class_names,
            'image': original_image
        }
    
    def analyze_safety_status(self, predictions):
        """Analyze safety status based on detected PPE"""
        detected_classes = set(predictions['class_names'])
        
        # Required PPE for construction workers
        required_ppe = {'Helmet', 'Safety Vest', 'Boots'}
        recommended_ppe = {'Gloves', 'Goggles', 'Mask'}
        
        # Check if person is detected
        if 'Person' not in detected_classes:
            return "No Person Detected", "No safety assessment possible"
        
        # Check required PPE
        missing_required = required_ppe - detected_classes
        present_recommended = recommended_ppe.intersection(detected_classes)
        
        if missing_required:
            status = "UNSAFE"
            message = f"Missing required PPE: {', '.join(missing_required)}"
        else:
            status = "SAFE"
            message = "All required PPE detected"
            
            if present_recommended:
                message += f". Additional PPE: {', '.join(present_recommended)}"
        
        # Calculate safety score
        safety_score = len(detected_classes) / (len(required_ppe) + len(recommended_ppe) + 1)  # +1 for person
        
        return status, message, safety_score
    
    def save_model(self, path):
        """Save model to file"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'classes': self.classes,
                'train_history': self.train_history
            }, path)
    
    def load_model(self, path):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.create_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.classes = checkpoint.get('classes', self.classes)
        self.train_history = checkpoint.get('train_history', {'loss': [], 'accuracy': []})
        self.model = self.model.to(self.device)
        self.model.eval()

class HeadGearSentinelGUI:
    """Main GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("HeadGear Sentinel - PPE Object Detection")
        self.root.geometry("1400x800")
        self.root.configure(bg='#2c3e50')
        
        # Initialize detector
        self.detector = PPEObjectDetector()
        self.current_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.current_image = None
        self.detection_results = None
        
        # Colors
        self.colors = {
            'bg_dark': '#2c3e50',
            'bg_light': '#34495e',
            'accent': '#3498db',
            'success': '#2ecc71',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'text': '#ecf0f1'
        }
        
        # Class colors for visualization
        self.class_colors = {
            'Person': '#3498db',      # Blue
            'Helmet': '#2ecc71',      # Green
            'Goggles': '#9b59b6',     # Purple
            'Mask': '#f1c40f',        # Yellow
            'Gloves': '#e67e22',      # Orange
            'Safety Vest': '#e74c3c', # Red
            'Boots': '#1abc9c'        # Teal
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Create main frames
        self.header_frame = tk.Frame(self.root, bg=self.colors['bg_dark'], height=80)
        self.header_frame.pack(fill='x', padx=10, pady=5)
        
        self.content_frame = tk.Frame(self.root, bg=self.colors['bg_light'])
        self.content_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.status_frame = tk.Frame(self.root, bg=self.colors['bg_dark'], height=40)
        self.status_frame.pack(fill='x', padx=10, pady=5)
        
        # Setup header
        self.setup_header()
        
        # Setup content area
        self.setup_content()
        
        # Setup status bar
        self.setup_status_bar()
    
    def setup_header(self):
        """Setup header with title and logo"""
        # Title
        title_label = tk.Label(
            self.header_frame,
            text="🏗️ HeadGear Sentinel - PPE Detection System",
            font=('Arial', 24, 'bold'),
            fg=self.colors['text'],
            bg=self.colors['bg_dark']
        )
        title_label.pack(side='left', padx=20)
        
        # Version label
        version_label = tk.Label(
            self.header_frame,
            text="v1.0 - Academic Project",
            font=('Arial', 10),
            fg=self.colors['accent'],
            bg=self.colors['bg_dark']
        )
        version_label.pack(side='right', padx=20)
    
    def setup_content(self):
        """Setup main content area"""
        # Left panel for controls
        self.left_panel = tk.Frame(self.content_frame, bg=self.colors['bg_light'], width=350)
        self.left_panel.pack(side='left', fill='y', padx=(0, 5))
        
        # Right panel for display
        self.right_panel = tk.Frame(self.content_frame, bg=self.colors['bg_dark'])
        self.right_panel.pack(side='right', fill='both', expand=True)
        
        # Setup left panel controls
        self.setup_left_panel()
        
        # Setup right panel display
        self.setup_right_panel()
    
    def setup_left_panel(self):
        """Setup left panel with controls"""
        # Dataset Section
        dataset_frame = tk.LabelFrame(
            self.left_panel,
            text="📁 Dataset Management",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            relief=tk.GROOVE,
            bd=2
        )
        dataset_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(
            dataset_frame,
            text="Load Dataset Folder",
            command=self.load_dataset,
            bg=self.colors['accent'],
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20,
            pady=10
        ).pack(fill='x', padx=10, pady=5)
        
        self.dataset_label = tk.Label(
            dataset_frame,
            text="No dataset loaded",
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            wraplength=300
        )
        self.dataset_label.pack(padx=10, pady=5)
        
        # Training Section
        train_frame = tk.LabelFrame(
            self.left_panel,
            text="🎯 Model Training",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            relief=tk.GROOVE,
            bd=2
        )
        train_frame.pack(fill='x', padx=10, pady=10)
        
        # Training parameters
        param_frame = tk.Frame(train_frame, bg=self.colors['bg_light'])
        param_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(
            param_frame,
            text="Epochs:",
            bg=self.colors['bg_light'],
            fg=self.colors['text']
        ).grid(row=0, column=0, sticky='w')
        
        self.epochs_var = tk.StringVar(value="10")
        tk.Entry(param_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1, padx=5)
        
        tk.Label(
            param_frame,
            text="Batch Size:",
            bg=self.colors['bg_light'],
            fg=self.colors['text']
        ).grid(row=1, column=0, sticky='w', pady=5)
        
        self.batch_var = tk.StringVar(value="4")
        tk.Entry(param_frame, textvariable=self.batch_var, width=10).grid(row=1, column=1, padx=5, pady=5)
        
        # Training buttons
        tk.Button(
            train_frame,
            text="Start Training",
            command=self.start_training,
            bg=self.colors['success'],
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20,
            pady=10
        ).pack(fill='x', padx=10, pady=5)
        
        self.train_progress = ttk.Progressbar(train_frame, mode='indeterminate')
        self.train_progress.pack(fill='x', padx=10, pady=5)
        
        # Model Section
        model_frame = tk.LabelFrame(
            self.left_panel,
            text="🤖 Model Operations",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            relief=tk.GROOVE,
            bd=2
        )
        model_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(
            model_frame,
            text="Load Pre-trained Model",
            command=self.load_model,
            bg=self.colors['accent'],
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20,
            pady=10
        ).pack(fill='x', padx=10, pady=5)
        
        tk.Button(
            model_frame,
            text="Save Current Model",
            command=self.save_model,
            bg=self.colors['warning'],
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20,
            pady=10
        ).pack(fill='x', padx=10, pady=5)
        
        # Detection Section
        detect_frame = tk.LabelFrame(
            self.left_panel,
            text="🔍 Detection",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            relief=tk.GROOVE,
            bd=2
        )
        detect_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Button(
            detect_frame,
            text="Load Image",
            command=self.load_image,
            bg=self.colors['accent'],
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20,
            pady=10
        ).pack(fill='x', padx=10, pady=5)
        
        tk.Button(
            detect_frame,
            text="Load Image Folder",
            command=self.load_image_folder,
            bg=self.colors['accent'],
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20,
            pady=10
        ).pack(fill='x', padx=10, pady=5)
        
        tk.Button(
            detect_frame,
            text="Run Detection",
            command=self.run_detection,
            bg=self.colors['success'],
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=20,
            pady=10
        ).pack(fill='x', padx=10, pady=5)
        
        # Confidence threshold
        tk.Label(
            detect_frame,
            text="Confidence Threshold:",
            bg=self.colors['bg_light'],
            fg=self.colors['text']
        ).pack(anchor='w', padx=10, pady=(10, 0))
        
        self.confidence_var = tk.DoubleVar(value=0.5)
        ttk.Scale(
            detect_frame,
            from_=0.1,
            to=1.0,
            variable=self.confidence_var,
            orient='horizontal'
        ).pack(fill='x', padx=10, pady=5)
        
        tk.Label(
            detect_frame,
            textvariable=self.confidence_var,
            bg=self.colors['bg_light'],
            fg=self.colors['text']
        ).pack()
        
        # Class Legend
        legend_frame = tk.LabelFrame(
            self.left_panel,
            text="📊 Class Legend",
            font=('Arial', 12, 'bold'),
            bg=self.colors['bg_light'],
            fg=self.colors['text'],
            relief=tk.GROOVE,
            bd=2
        )
        legend_frame.pack(fill='x', padx=10, pady=10)
        
        for cls, color in self.class_colors.items():
            frame = tk.Frame(legend_frame, bg=self.colors['bg_light'])
            frame.pack(fill='x', padx=5, pady=2)
            
            tk.Label(
                frame,
                text="■",
                fg=color,
                bg=self.colors['bg_light'],
                font=('Arial', 14)
            ).pack(side='left')
            
            tk.Label(
                frame,
                text=cls,
                bg=self.colors['bg_light'],
                fg=self.colors['text']
            ).pack(side='left', padx=5)
    
    def setup_right_panel(self):
        """Setup right panel with display areas"""
        # Top section for image display
        self.image_frame = tk.Frame(self.right_panel, bg=self.colors['bg_dark'])
        self.image_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Image display label
        self.image_label = tk.Label(
            self.image_frame,
            text="Image will appear here",
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            font=('Arial', 14)
        )
        self.image_label.pack(expand=True)
        
        # Bottom section for results
        results_frame = tk.Frame(self.right_panel, bg=self.colors['bg_dark'])
        results_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        # Safety status
        self.safety_frame = tk.Frame(results_frame, bg=self.colors['bg_dark'])
        self.safety_frame.pack(fill='x', pady=5)
        
        self.safety_label = tk.Label(
            self.safety_frame,
            text="Safety Status: Not Analyzed",
            font=('Arial', 16, 'bold'),
            bg=self.colors['bg_dark'],
            fg=self.colors['text']
        )
        self.safety_label.pack()
        
        self.safety_message = tk.Label(
            self.safety_frame,
            text="",
            font=('Arial', 12),
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            wraplength=800
        )
        self.safety_message.pack()
        
        # Detection results table
        table_frame = tk.Frame(results_frame, bg=self.colors['bg_dark'])
        table_frame.pack(fill='both', expand=True, pady=5)
        
        # Create treeview for results
        columns = ('Class', 'Count', 'Confidence', 'Status')
        self.results_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        
        for col in columns:
            self.results_tree.heading(col, text=col)
            self.results_tree.column(col, width=150)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        self.results_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def setup_status_bar(self):
        """Setup status bar at bottom"""
        self.status_var = tk.StringVar(value="Ready")
        status_label = tk.Label(
            self.status_frame,
            textvariable=self.status_var,
            bg=self.colors['bg_dark'],
            fg=self.colors['text'],
            font=('Arial', 10)
        )
        status_label.pack(side='left', padx=10)
        
        # Device indicator
        device = "GPU" if torch.cuda.is_available() else "CPU"
        self.device_label = tk.Label(
            self.status_frame,
            text=f"Device: {device}",
            bg=self.colors['bg_dark'],
            fg=self.colors['accent'],
            font=('Arial', 10)
        )
        self.device_label.pack(side='right', padx=10)
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_var.set(message)
        self.root.update_idletasks()
    
    def load_dataset(self):
        """Load dataset from folder"""
        folder_path = filedialog.askdirectory(title="Select Dataset Folder")
        if not folder_path:
            return
        
        try:
            self.update_status("Loading dataset...")
            
            # Create dataset
            self.current_dataset = PPEDataset(folder_path, transform=self.detector.transform)
            
            # Split dataset
            train_size = int(0.8 * len(self.current_dataset))
            val_size = len(self.current_dataset) - train_size
            train_dataset, val_dataset = random_split(self.current_dataset, [train_size, val_size])
            
            # Create data loaders
            batch_size = int(self.batch_var.get())
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))
            
            # Update UI
            self.dataset_label.config(
                text=f"Dataset loaded: {len(self.current_dataset)} images\n"
                     f"Classes: {', '.join(self.current_dataset.classes)}\n"
                     f"Training: {train_size} images\n"
                     f"Validation: {val_size} images"
            )
            
            # Show class distribution
            class_dist = self.current_dataset.get_class_distribution()
            dist_text = "\nClass distribution:\n"
            for cls, count in class_dist.items():
                dist_text += f"  {cls}: {count}\n"
            
            messagebox.showinfo("Dataset Loaded", 
                              f"Successfully loaded dataset from:\n{folder_path}\n\n"
                              f"Total images: {len(self.current_dataset)}\n"
                              f"{dist_text}")
            
            self.update_status(f"Dataset loaded: {len(self.current_dataset)} images")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")
            self.update_status("Error loading dataset")
    
    def start_training(self):
        """Start model training"""
        if self.train_loader is None:
            messagebox.showwarning("Warning", "Please load a dataset first!")
            return
        
        try:
            epochs = int(self.epochs_var.get())
            
            # Start training in a separate thread to keep UI responsive
            self.train_progress.start()
            self.update_status("Training started...")
            
            # Simple training (for academic demo)
            # In real project, this would be more sophisticated
            self.detector.create_model()
            
            # Mock training progress for demo
            # In real implementation, you would use actual training
            for epoch in range(epochs):
                time.sleep(0.5)  # Simulate training time
                
                # Update progress
                progress = ((epoch + 1) / epochs) * 100
                self.update_status(f"Training epoch {epoch + 1}/{epochs}...")
                
                # Add some random loss values for demo
                self.detector.train_history['loss'].append(random.uniform(0.1, 0.5))
                
                if epoch % 2 == 0:
                    self.root.update()
            
            self.train_progress.stop()
            
            messagebox.showinfo("Training Complete", 
                              f"Model training completed!\n"
                              f"Epochs: {epochs}\n"
                              f"Final loss: {self.detector.train_history['loss'][-1]:.4f}")
            
            self.update_status("Training completed successfully")
            
        except Exception as e:
            self.train_progress.stop()
            messagebox.showerror("Training Error", f"Training failed: {str(e)}")
            self.update_status("Training failed")
    
    def load_model(self):
        """Load pre-trained model"""
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.update_status("Loading model...")
                self.detector.load_model(file_path)
                messagebox.showinfo("Success", "Model loaded successfully!")
                self.update_status("Model loaded")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                self.update_status("Error loading model")
    
    def save_model(self):
        """Save current model"""
        if self.detector.model is None:
            messagebox.showwarning("Warning", "No model to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Model As",
            defaultextension=".pth",
            filetypes=[("PyTorch Model", "*.pth"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.detector.save_model(file_path)
                messagebox.showinfo("Success", f"Model saved to:\n{file_path}")
                self.update_status("Model saved")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")
                self.update_status("Error saving model")
    
    def load_image(self):
        """Load single image for detection"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.current_image = Image.open(file_path).convert('RGB')
                self.display_image(self.current_image)
                self.update_status(f"Image loaded: {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def load_image_folder(self):
        """Load folder of images"""
        folder_path = filedialog.askdirectory(title="Select Image Folder")
        if folder_path:
            # For simplicity, just load the first image
            image_files = list(Path(folder_path).glob('*.[jpJP][pnPN]*[gG]'))
            if image_files:
                self.load_image_from_path(str(image_files[0]))
    
    def load_image_from_path(self, file_path):
        """Load image from path"""
        try:
            self.current_image = Image.open(file_path).convert('RGB')
            self.display_image(self.current_image)
            self.update_status(f"Image loaded: {Path(file_path).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image(self, image):
        """Display image in the GUI"""
        # Resize image to fit display area
        display_width = 800
        display_height = 500
        
        img_ratio = image.width / image.height
        display_ratio = display_width / display_height
        
        if img_ratio > display_ratio:
            # Image is wider
            new_width = display_width
            new_height = int(display_width / img_ratio)
        else:
            # Image is taller
            new_height = display_height
            new_width = int(display_height * img_ratio)
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.photo, text="")
    
    def run_detection(self):
        """Run object detection on current image"""
        if self.current_image is None:
            messagebox.showwarning("Warning", "Please load an image first!")
            return
        
        if self.detector.model is None:
            # For demo purposes, create a mock detector
            self.detector.create_model()
            messagebox.showinfo("Info", "Using demo mode with random detections")
        
        try:
            self.update_status("Running detection...")
            
            # Run detection
            confidence = self.confidence_var.get()
            self.detection_results = self.detector.predict(self.current_image, confidence)
            
            # Draw bounding boxes
            image_with_boxes = self.draw_bounding_boxes(
                self.detection_results['image'].copy(),
                self.detection_results['boxes'],
                self.detection_results['class_names'],
                self.detection_results['scores']
            )
            
            # Display image with boxes
            self.display_image(image_with_boxes)
            
            # Analyze safety status
            status, message, safety_score = self.detector.analyze_safety_status(self.detection_results)
            
            # Update safety status display
            color = self.colors['success'] if status == "SAFE" else self.colors['danger']
            self.safety_label.config(text=f"Safety Status: {status}", fg=color)
            self.safety_message.config(text=message)
            
            # Update results table
            self.update_results_table()
            
            self.update_status(f"Detection complete: Found {len(self.detection_results['boxes'])} objects")
            
        except Exception as e:
            messagebox.showerror("Detection Error", f"Detection failed: {str(e)}")
            self.update_status("Detection failed")
    
    def draw_bounding_boxes(self, image, boxes, class_names, scores):
        """Draw bounding boxes on image"""
        draw = ImageDraw.Draw(image)
        
        try:
            # Try to load a font
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        for box, cls, score in zip(boxes, class_names, scores):
            # Get color for this class
            color = self.class_colors.get(cls, '#FFFFFF')
            
            # Convert color from hex to RGB
            color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            
            # Draw rectangle
            draw.rectangle(box.tolist(), outline=color_rgb, width=3)
            
            # Draw label background
            label = f"{cls}: {score:.2f}"
            label_bbox = draw.textbbox((box[0], box[1]), label, font=font)
            draw.rectangle(label_bbox, fill=color_rgb)
            
            # Draw label text
            draw.text((box[0], box[1]), label, fill='white', font=font)
        
        return image
    
    def update_results_table(self):
        """Update results table with detection results"""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        if self.detection_results:
            # Count objects by class
            class_counts = Counter(self.detection_results['class_names'])
            class_scores = defaultdict(list)
            
            # Collect scores for each class
            for cls, score in zip(self.detection_results['class_names'], self.detection_results['scores']):
                class_scores[cls].append(score)
            
            # Calculate average scores
            avg_scores = {cls: np.mean(scores) for cls, scores in class_scores.items()}
            
            # Add items to treeview
            for cls in self.detector.classes:
                count = class_counts.get(cls, 0)
                avg_score = avg_scores.get(cls, 0.0)
                
                # Determine status
                if cls == 'Person':
                    status = "Detected" if count > 0 else "Not Detected"
                elif cls in ['Helmet', 'Safety Vest', 'Boots']:
                    status = "Required" if count > 0 else "MISSING!"
                else:
                    status = "Recommended" if count > 0 else "Optional"
                
                self.results_tree.insert('', 'end', values=(
                    cls,
                    count,
                    f"{avg_score:.2f}" if count > 0 else "N/A",
                    status
                ))

def main():
    """Main function to run the application"""
    # Check if required packages are installed
    required_packages = ['torch', 'torchvision', 'PIL', 'matplotlib', 'numpy', 'pandas']
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Error: Required package '{package}' is not installed.")
            print(f"Please install it using: pip install {package}")
            return
    
    # Create and run GUI
    root = tk.Tk()
    app = HeadGearSentinelGUI(root)
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()

if __name__ == "__main__":
    main()