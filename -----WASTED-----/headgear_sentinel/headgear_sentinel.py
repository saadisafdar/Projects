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
    # HOG + sliding window
    'window_size': (64, 64),
    'step_size': 24,          # coarser step to reduce windows
    'scale_factor': 1.4,      # fewer pyramid levels

    # Classes
    'class_names': [
        'Person',
        'Helmet',
        'Goggles',
        'Mask',
        'Gloves',
        'Vest',
        'Boots'
    ],
    'safe_classes': [1, 2, 3, 4, 5, 6],

    # Detection post-processing
    # These are deliberately strict to reduce false positives.
    'confidence_threshold': 1.5,     # SVM margin threshold
    'nms_threshold': 0.30,           # IoU for NMS
    'max_detections_per_class': 4    # show at most 4 boxes per class
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

    def image_pyramid(self, image, scale=1.4, min_size=(64, 64)):
        """
        Generate image pyramid for multi-scale detection

        Yields: (resized_image, scale_factor_relative_to_original)
        """
        current = image.copy()
        base_h, base_w = current.shape[:2]
        scale_factor = 1.0

        while True:
            h, w = current.shape[:2]
            if min(h, w) < min_size[0]:
                break
            yield current, scale_factor

            new_h = int(h / scale)
            new_w = int(w / scale)
            if min(new_h, new_w) < min_size[0]:
                break
            current = cv2.resize(current, (new_w, new_h))
            scale_factor = current.shape[0] / float(base_h)

    def sliding_window(self, image, step_size=24, window_size=(64, 64)):
        """
        Slide window across image

        Yields: (x, y, window_patch)
        """
        h, w = image.shape[:2]
        win_h, win_w = window_size
        for y in range(0, h - win_h + 1, step_size):
            for x in range(0, w - win_w + 1, step_size):
                window = image[y:y + win_h, x:x + win_w]
                if window.shape == window_size:
                    yield (x, y, window)

    def detect_single_class(self, image, classifier, scaler, class_name):
        """
        Detect single class in image using sliding window

        Returns:
            list of dicts: {x,y,w,h,confidence}
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        detections = []
        conf_thr = CONFIG['confidence_threshold']

        for scaled_img, scale in self.image_pyramid(
            gray,
            CONFIG['scale_factor'],
            CONFIG['window_size']
        ):
            h, w = scaled_img.shape[:2]
            if min(h, w) < CONFIG['window_size'][0]:
                continue

            for (x, y, window) in self.sliding_window(
                scaled_img,
                CONFIG['step_size'],
                CONFIG['window_size']
            ):
                features = self.hog_extractor.extract(window)
                features = features.reshape(1, -1)
                features = scaler.transform(features)

                # Use real class prediction plus margin
                prediction = classifier.predict(features)[0]
                confidence = classifier.decision_function(features)[0]

                # Only accept if SVM says "positive" AND margin is strong
                if prediction == 1 and confidence >= conf_thr:
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

        # sort by confidence and keep top-N for this class
        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        max_n = CONFIG['max_detections_per_class']
        if max_n is not None and len(detections) > max_n:
            detections = detections[:max_n]

        return detections

    def non_maximum_suppression(self, detections, threshold=0.30):
        """
        Remove overlapping detections using IoU threshold
        """
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
        keep = []

        for det in detections:
            overlaps = False
            for kept in keep:
                x1_min, x1_max = det['x'], det['x'] + det['w']
                y1_min, y1_max = det['y'], det['y'] + det['h']
                x2_min, x2_max = kept['x'], kept['x'] + kept['w']
                y2_min, y2_max = kept['y'], kept['y'] + kept['h']

                inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
                inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
                inter = inter_x * inter_y
                union = (det['w'] * det['h']) + (kept['w'] * kept['h']) - inter
                iou = inter / union if union > 0 else 0.0

                if iou > threshold:
                    overlaps = True
                    break

            if not overlaps:
                keep.append(det)

        return keep


# ============================================================================
# TRAINER - BATCH TRAINING
# ============================================================================

class ModelTrainer:
    """Train one-vs-rest binary classifiers for each PPE class"""

    def __init__(self):
        self.hog_extractor = HOGExtractor()
        self.classifiers = {}
        self.scalers = {}
        self.training_stats = {}

    def load_dataset(self, dataset_path):
        dataset = {}
        dataset_path = Path(dataset_path)

        for class_id in range(len(CONFIG['class_names'])):
            folder_name = f"class_{class_id}_{CONFIG['class_names'][class_id].lower()}"
            class_folder = dataset_path / folder_name
            if not class_folder.exists():
                print(f"WARNING: {class_folder} not found")
                dataset[class_id] = []
                continue

            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images.extend(class_folder.glob(ext))

            dataset[class_id] = [str(img) for img in images]
            print(f"Class {class_id} ({CONFIG['class_names'][class_id]}): {len(images)} images")
        return dataset

    def extract_features_batch(self, image_paths, class_label):
        features_list = []
        labels_list = []

        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, CONFIG['window_size'])
                feat = self.hog_extractor.extract(gray)
                features_list.append(feat)
                labels_list.append(class_label)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        if not features_list:
            return np.array([]), np.array([])
        return np.array(features_list), np.array(labels_list)

    def train_binary_classifier(self, positive_paths, negative_paths, class_id, callback=None):
        class_name = CONFIG['class_names'][class_id]
        print(f"\n--- Training Class {class_id}: {class_name} ---")

        if callback:
            callback(f"Extracting positives for {class_name}...", 10)

        pos_features, pos_labels = self.extract_features_batch(positive_paths, 1)
        if len(pos_features) == 0:
            print(f"WARNING: No positive samples for class {class_id}")
            return False
        print(f"  Positive samples: {len(pos_features)}")

        if callback:
            callback(f"Extracting negatives for {class_name}...", 30)

        neg_features_list = []
        for other_id, paths in negative_paths.items():
            if other_id == class_id:
                continue
            feats, _ = self.extract_features_batch(paths, 0)
            if len(feats) > 0:
                neg_features_list.append(feats)

        if neg_features_list:
            neg_features = np.vstack(neg_features_list)
        else:
            neg_features = np.array([])

        if len(neg_features) > 0:
            print(f"  Negative samples: {len(neg_features)}")

        if len(neg_features) > 0:
            X = np.vstack([pos_features, neg_features])
            y = np.hstack([pos_labels, np.zeros(len(neg_features), dtype=int)])
        else:
            X = pos_features
            y = pos_labels

        print(f"  Total training samples: {len(X)}")

        if callback:
            callback(f"Scaling features for {class_name}...", 50)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if callback:
            callback(f"Training SVM for {class_name}...", 70)

        clf = LinearSVC(
            C=1.0,
            max_iter=10000,
            dual=False,
            random_state=42,
            verbose=0
        )
        clf.fit(X_scaled, y)
        train_score = clf.score(X_scaled, y)
        print(f"  Training accuracy: {train_score:.2%}")

        self.classifiers[class_id] = clf
        self.scalers[class_id] = scaler
        self.training_stats[class_id] = {
            'positive_samples': int(len(pos_features)),
            'negative_samples': int(len(neg_features)),
            'training_accuracy': float(train_score),
            'timestamp': datetime.now().isoformat()
        }
        return True

    def train_all_classes(self, dataset_path, callback=None):
        print("\n" + "=" * 60)
        print("HEADGEAR SENTINEL - TRAINING STARTED")
        print("=" * 60)

        if callback:
            callback("Loading dataset...", 0)

        dataset = self.load_dataset(dataset_path)
        total_classes = len(CONFIG['class_names'])
        trained = 0

        for class_id in range(total_classes):
            if not dataset.get(class_id):
                print(f"Skipping class {class_id} (no samples)")
                continue

            progress = int((class_id / total_classes) * 100)
            if callback:
                callback(f"Training {class_id + 1}/{total_classes}...", progress)

            positive = dataset[class_id]
            negative = {cid: paths for cid, paths in dataset.items() if cid != class_id}

            success = self.train_binary_classifier(positive, negative, class_id, callback)
            if success:
                trained += 1

        print(f"\nTraining complete: {trained}/{total_classes} classifiers trained")
        if callback:
            callback("Training complete!", 100)

        return trained > 0

    def save_models(self, model_path):
        model_path = Path(model_path)
        model_path.mkdir(parents=True, exist_ok=True)

        for class_id, clf in self.classifiers.items():
            joblib.dump(clf, model_path / f'clf_{class_id}.pkl')
            joblib.dump(self.scalers[class_id], model_path / f'scaler_{class_id}.pkl')

        with open(model_path / 'training_stats.json', 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        print(f"Models saved to {model_path}")

    def load_models(self, model_path):
        model_path = Path(model_path)
        self.classifiers.clear()
        self.scalers.clear()

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
        self.root.geometry("1250x900")
        self.root.resizable(False, False)

        self.center_window()

        self.trainer = ModelTrainer()
        self.detector = SlidingWindowDetector()
        self.current_image = None
        self.current_image_path = None
        self.dataset_path = None
        self.model_path = "./models"

        self.training_thread = None
        self.is_training = False

        self.build_ui()
        self.log("HeadGear Sentinel initialized")

    def center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f'{w}x{h}+{x}+{y}')

    def build_ui(self):
        style = ttk.Style()
        style.configure('TLabel', font=('Segoe UI', 9))
        style.configure('Header.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        title_label = ttk.Label(
            main_frame,
            text="HeadGear Sentinel - Construction PPE Detection",
            style='Title.TLabel',
            anchor='center'
        )
        title_label.pack(fill='x', pady=(0, 10))

        control_frame = ttk.LabelFrame(main_frame, text='Control Panel', padding=10)
        control_frame.pack(fill='x', pady=(0, 10))

        ttk.Label(control_frame, text='Dataset Path:').grid(row=0, column=0, sticky='w')
        self.dataset_entry = ttk.Entry(control_frame, width=60)
        self.dataset_entry.grid(row=0, column=1, padx=5)
        self.dataset_entry.insert(0, './dataset')
        ttk.Button(control_frame, text='Browse...', width=12,
                   command=self.browse_dataset).grid(row=0, column=2, padx=5)

        ttk.Button(control_frame, text='Load Dataset', width=18,
                   command=self.load_dataset).grid(row=1, column=0, sticky='ew', padx=5, pady=5)
        ttk.Button(control_frame, text='Train Model', width=18,
                   command=self.start_training).grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        ttk.Button(control_frame, text='Load Trained Model', width=18,
                   command=self.load_model).grid(row=1, column=2, sticky='ew', padx=5, pady=5)

        ttk.Button(control_frame, text='Load Image', width=18,
                   command=self.load_image).grid(row=2, column=0, sticky='ew', padx=5, pady=5)
        ttk.Button(control_frame, text='Run Detection', width=18,
                   command=self.run_detection).grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        ttk.Button(control_frame, text='Clear', width=18,
                   command=self.clear_results).grid(row=2, column=2, sticky='ew', padx=5, pady=5)

        progress_frame = ttk.LabelFrame(main_frame, text='Training Progress', padding=10)
        progress_frame.pack(fill='x', pady=(0, 10))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill='x', pady=5)

        self.status_label = ttk.Label(progress_frame, text='Ready', foreground='blue')
        self.status_label.pack(anchor='w')

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill='both', expand=True, pady=(0, 10))

        left_frame = ttk.LabelFrame(content_frame, text='Image Viewer', padding=10)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))

        self.image_label = ttk.Label(left_frame, background='lightgray', text='No image loaded')
        self.image_label.pack(fill='both', expand=True)

        right_frame = ttk.LabelFrame(content_frame, text='Detection Results', padding=10)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))

        header_frame = ttk.Frame(right_frame)
        header_frame.pack(fill='x')

        ttk.Label(header_frame, text='Detection Summary', style='Header.TLabel').pack(side='left')
        self.safety_label = ttk.Label(header_frame, text='Status: -',
                                      font=('Segoe UI', 12, 'bold'), foreground='gray')
        self.safety_label.pack(side='right')

        summary_details_frame = ttk.Frame(right_frame)
        summary_details_frame.pack(fill='both', expand=True, pady=(10, 0))

        summary_frame = ttk.Frame(summary_details_frame)
        summary_frame.pack(side='top', fill='x', expand=False)

        self.summary_text = tk.Text(summary_frame, height=7, width=60,
                                    font=('Courier', 9), state='disabled', wrap='word')
        self.summary_text.pack(fill='x', expand=False)

        ttk.Label(summary_details_frame, text='Detailed Detections:').pack(anchor='w', pady=(5, 2))

        details_frame = ttk.Frame(summary_details_frame)
        details_frame.pack(fill='both', expand=True)

        details_scroll = ttk.Scrollbar(details_frame)
        details_scroll.pack(side='right', fill='y')

        self.results_text = tk.Text(
            details_frame,
            height=15,
            width=60,
            yscrollcommand=details_scroll.set,
            font=('Courier', 9)
        )
        self.results_text.pack(side='left', fill='both', expand=True)
        details_scroll.config(command=self.results_text.yview)

        log_frame = ttk.LabelFrame(main_frame, text='System Log', padding=10, height=100)
        log_frame.pack(fill='both', expand=False)

        log_scroll = ttk.Scrollbar(log_frame)
        log_scroll.pack(side='right', fill='y')

        self.log_text = tk.Text(log_frame, height=5, yscrollcommand=log_scroll.set, font=('Courier', 8))
        self.log_text.pack(fill='both', expand=True)
        log_scroll.config(command=self.log_text.yview)

    def log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        line = f"[{timestamp}] {message}\n"
        try:
            if not hasattr(self, "log_text") or self.log_text is None:
                print(line, end='')
                return
            if not self.root.winfo_exists():
                print(line, end='')
                return
            self.log_text.insert('end', line)
            self.log_text.see('end')
            self.root.update_idletasks()
        except tk.TclError:
            print(line, end='')

    def update_progress(self, message, progress):
        try:
            if not self.root.winfo_exists():
                print(f"[PROGRESS] {message} ({progress}%)")
                return
            self.status_label.config(text=message)
            self.progress_var.set(progress)
            self.root.update_idletasks()
        except tk.TclError:
            print(f"[PROGRESS] {message} ({progress}%)")

    def browse_dataset(self):
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            self.dataset_entry.delete(0, 'end')
            self.dataset_entry.insert(0, folder)

    def load_dataset(self):
        dataset_path = self.dataset_entry.get()
        if not Path(dataset_path).exists():
            messagebox.showerror("Error", f"Dataset path not found: {dataset_path}")
            return
        dataset = self.trainer.load_dataset(dataset_path)
        total = sum(len(paths) for paths in dataset.values())
        self.log(f"Dataset loaded: {total} total images")
        self.dataset_path = dataset_path

    def start_training(self):
        if not self.dataset_path:
            messagebox.showwarning("Warning", "Please load dataset first")
            return
        if self.is_training:
            messagebox.showwarning("Warning", "Training already in progress")
            return

        self.is_training = True
        self.training_thread = threading.Thread(
            target=self._train_worker,
            daemon=True
        )
        self.training_thread.start()

    def _train_worker(self):
        try:
            self.log("Training started...")
            success = self.trainer.train_all_classes(
                self.dataset_path,
                callback=self.update_progress
            )
            if success:
                self.trainer.save_models(self.model_path)
                self.log(f"Training complete! Models saved to {self.model_path}")
                self.update_progress("Training complete!", 100)
                try:
                    messagebox.showinfo("Success", "Model training complete!")
                except tk.TclError:
                    pass
            else:
                self.log("Training failed")
                try:
                    messagebox.showerror("Error", "Training failed")
                except tk.TclError:
                    pass
        except Exception as e:
            self.log(f"Training error: {e}")
            try:
                messagebox.showerror("Error", f"Training error: {e}")
            except tk.TclError:
                pass
        finally:
            self.is_training = False

    def load_model(self):
        if not Path(self.model_path).exists():
            messagebox.showerror("Error", f"Model path not found: {self.model_path}")
            return
        if self.trainer.load_models(self.model_path):
            self.log(f"Loaded {len(self.trainer.classifiers)} class models")
            messagebox.showinfo("Success", f"Loaded {len(self.trainer.classifiers)} class models")
        else:
            messagebox.showerror("Error", "No trained models found")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Error", "Failed to load image")
            return

        self.current_image = img
        self.current_image_path = file_path
        self.display_image(self.current_image)
        self.log(f"Image loaded: {Path(file_path).name}")
        self.clear_results_text()

    def display_image(self, image):
        h, w = image.shape[:2]
        max_w, max_h = 500, 550
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            w = int(w * scale)
            h = int(h * scale)
            image = cv2.resize(image, (w, h))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        self.image_label.config(image=image_tk, text='')
        self.image_label.image = image_tk

    def run_detection(self):
        if not self.root.winfo_exists():
            print("Root window destroyed, aborting detection")
            return

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

            for class_id in range(len(CONFIG['class_names'])):
                if class_id not in self.trainer.classifiers:
                    continue

                class_name = CONFIG['class_names'][class_id]
                clf = self.trainer.classifiers[class_id]
                scaler = self.trainer.scalers[class_id]

                self.log(f"Detecting {class_name}...")

                dets = self.detector.detect_single_class(
                    self.current_image,
                    clf,
                    scaler,
                    class_name
                )
                dets = self.detector.non_maximum_suppression(
                    dets, threshold=CONFIG['nms_threshold']
                )
                detections_by_class[class_id] = dets

                for det in dets:
                    det['class_id'] = class_id
                    det['class_name'] = class_name
                    all_detections.append(det)

            self.display_detection_results(detections_by_class, all_detections)

            result_image = self.current_image.copy()
            for det in all_detections:
                x, y, w, h = det['x'], det['y'], det['w'], det['h']
                color = (0, 255, 0)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                label = f"{det['class_name']} ({det['confidence']:.2f})"
                cv2.putText(result_image, label, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            self.display_image(result_image)
            self.log("Detection complete")
        except Exception as e:
            self.log(f"Detection error: {e}")
            try:
                messagebox.showerror("Error", f"Detection error: {e}")
            except tk.TclError:
                pass

    def display_detection_results(self, by_class, all_dets):
        is_safe = all(len(by_class.get(c, [])) > 0 for c in CONFIG['safe_classes'])
        if len(all_dets) == 0:
            is_safe = False

        status_text = "SAFE" if is_safe else "UNSAFE"
        status_color = 'green' if is_safe else 'red'
        self.safety_label.config(text=f"Status: {status_text}", foreground=status_color)

        summary_lines = []
        summary_lines.append("==============================================")
        summary_lines.append(f" SAFETY STATUS: {status_text}")
        summary_lines.append("==============================================")
        summary_lines.append("")
        summary_lines.append(f"Total detections (all classes): {len(all_dets)}")
        summary_lines.append("")
        summary_lines.append("PPE checklist (per image):")
        for cid in CONFIG['safe_classes']:
            name = CONFIG['class_names'][cid]
            count = len(by_class.get(cid, []))
            mark = "✓" if count > 0 else "✗"
            summary_lines.append(f"  {name:8s}: {mark}  (count: {count})")

        summary_text = "\n".join(summary_lines)

        self.summary_text.config(state='normal')
        self.summary_text.delete('1.0', 'end')
        self.summary_text.insert('end', summary_text)
        self.summary_text.config(state='disabled')

        details_lines = []
        if not all_dets:
            details_lines.append("No objects detected.")
        else:
            for cid, dets in by_class.items():
                if not dets:
                    continue
                cname = CONFIG['class_names'][cid]
                details_lines.append("-" * 40)
                details_lines.append(f"{cname} (up to {len(dets)} detections):")
                for idx, det in enumerate(dets, start=1):
                    details_lines.append(
                        f"  #{idx:02d}  pos=({det['x']},{det['y']}), "
                        f"size={det['w']}x{det['h']}, conf={det['confidence']:.2f}"
                    )
                details_lines.append("")

        self.results_text.insert('end', "\n".join(details_lines))

    def clear_results(self):
        self.current_image = None
        self.current_image_path = None
        self.image_label.config(image='', text='No image loaded')
        self.image_label.image = None
        self.clear_results_text()
        self.safety_label.config(text='Status: -', foreground='gray')

    def clear_results_text(self):
        self.results_text.delete('1.0', 'end')
        self.summary_text.config(state='normal')
        self.summary_text.delete('1.0', 'end')
        self.summary_text.insert('end', "Run detection to see summary here.")
        self.summary_text.config(state='disabled')


# ============================================================================
# MAIN
# ============================================================================

def main():
    root = tk.Tk()
    app = HeadGearSentinelGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
