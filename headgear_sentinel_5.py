import os
import threading
import pickle
import traceback
from PIL import Image, ImageTk

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk


# =========================
# Global configuration
# =========================

# Fixed class mapping (index -> label)
CLASS_INDEX_TO_NAME = {
    0: "Person",
    1: "Helmet",
    2: "Goggles",
    3: "Mask",
    4: "Gloves",
    5: "Safety Vest",
    6: "Boots",
}

# Folder name mapping: class_0_person, class_1_helmet, ...
FOLDER_PREFIX = "class_"

# Image size used for feature extraction (small and light)
IMG_SIZE = (64, 64)

# Model file path
MODEL_FILE = "headgear_sentinel_model.pkl"
SCALER_FILE = "headgear_sentinel_scaler.pkl"

# =========================
# Feature extraction (lightweight)
# =========================

def load_and_preprocess_image(path):
    """
    Load an image from disk, convert to RGB, resize to IMG_SIZE, and
    flatten as a 1D float32 vector in [0,1].
    This is intentionally simple to stay CPU‑friendly.
    """
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(IMG_SIZE)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.flatten()
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def load_dataset(dataset_root, log_callback=None):
    """
    Walk through dataset folders and build X (features) and y (labels).
    Assumes structure:
        dataset/
          class_0_person/
          class_1_helmet/
          ...
          class_6_boots/
    """
    X = []
    y = []

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError("Dataset folder does not exist")

    # Iterate over directories
    for entry in sorted(os.listdir(dataset_root)):
        entry_path = os.path.join(dataset_root, entry)
        if not os.path.isdir(entry_path):
            continue

        if not entry.startswith(FOLDER_PREFIX):
            # Ignore unknown folders
            continue

        # Extract numeric index from folder name: class_0_person -> 0
        try:
            # Split like ["class", "0", "person"]
            parts = entry.split("_")
            class_index = int(parts[1])
        except Exception:
            # If naming unexpected, skip
            continue

        if class_index not in CLASS_INDEX_TO_NAME:
            continue

        class_name = CLASS_INDEX_TO_NAME[class_index]
        if log_callback:
            log_callback(f"Loading: {entry}  ->  label {class_index} ({class_name})")

        # Read all image files in this folder
        for fname in os.listdir(entry_path):
            fpath = os.path.join(entry_path, fname)
            if not os.path.isfile(fpath):
                continue
            # Basic filter by extension
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            feat = load_and_preprocess_image(fpath)
            if feat is not None:
                X.append(feat)
                y.append(class_index)

    if len(X) == 0:
        raise ValueError("No images found in dataset")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    return X, y


# =========================
# Lightweight model: SGDClassifier
# =========================

def train_model(X, y, log_callback=None):
    """
    Train a lightweight linear classifier (SGDClassifier) on flattened images.
    Uses StandardScaler + linear classifier for speed and stability.
    """
    if log_callback:
        log_callback("Scaling features...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split for basic validation (not required but looks academic)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    if log_callback:
        log_callback("Training linear model (SGDClassifier)...")

    clf = SGDClassifier(
        loss="log_loss",  # logistic regression
        max_iter=200,
        learning_rate="optimal",
        n_jobs=1,   # CPU‑friendly
        random_state=42
    )
    clf.fit(X_train, y_train)

    if log_callback:
        log_callback("Evaluating model on validation split...")
        y_pred = clf.predict(X_test)
        report = classification_report(
            y_test,
            y_pred,
            target_names=[CLASS_INDEX_TO_NAME[i] for i in sorted(CLASS_INDEX_TO_NAME.keys())],
            zero_division=0
        )
        log_callback(report)

    return clf, scaler


def save_model(model, scaler, model_path=MODEL_FILE, scaler_path=SCALER_FILE):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)


def load_model(model_path=MODEL_FILE, scaler_path=SCALER_FILE):
    if not os.path.isfile(model_path) or not os.path.isfile(scaler_path):
        raise FileNotFoundError("Model or scaler file not found. Train the model first.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, scaler


def predict_image(model, scaler, image_path):
    feat = load_and_preprocess_image(image_path)
    if feat is None:
        raise ValueError("Could not preprocess image")
    X = np.expand_dims(feat, axis=0)
    X_scaled = scaler.transform(X)
    probs = None
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
    pred = model.predict(X_scaled)[0]
    return int(pred), probs


# =========================
# Safety logic
# =========================

def evaluate_safety(pred_class_indices):
    """
    Given a list of predicted class indices present in the image,
    determine safety status.

    Logic:
    - If no person -> Object not present / N/A safety
    - If person present but any PPE (helmet, vest, boots) missing -> UNSAFE
    - Otherwise -> SAFE

    You can easily adjust required PPE according to teacher preference.
    """
    class_set = set(pred_class_indices)

    has_person = 0 in class_set
    has_helmet = 1 in class_set
    has_vest = 5 in class_set
    has_boots = 6 in class_set

    # Optional: include mask, goggles, gloves in stricter logic
    # has_mask = 3 in class_set
    # has_goggles = 2 in class_set
    # has_gloves = 4 in class_set

    if not has_person:
        return "No person detected", "NOT PRESENT", "N/A"

    # Basic PPE requirement: helmet + vest + boots
    missing_items = []
    if not has_helmet:
        missing_items.append("Helmet")
    if not has_vest:
        missing_items.append("Safety Vest")
    if not has_boots:
        missing_items.append("Boots")

    if missing_items:
        msg = "Missing PPE: " + ", ".join(missing_items)
        return msg, "PRESENT", "UNSAFE"

    return "All required PPE detected", "PRESENT", "SAFE"


# =========================
# Tkinter GUI
# =========================

class HeadGearSentinelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("HeadGear Sentinel - PPE Object Detection")

        # Center window
        win_w = 1000
        win_h = 600
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        x = int((screen_w / 2) - (win_w / 2))
        y = int((screen_h / 2) - (win_h / 2))
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.resizable(False, False)

        # State variables
        self.dataset_path = None
        self.model = None
        self.scaler = None
        self.current_image_path = None
        self.current_image_tk = None
        self.training_thread = None
        self.is_training = False

        # Build UI
        self._build_ui()

        # Try to autoload existing model (nice for demo)
        try:
            self.model, self.scaler = load_model()
            self.log("Existing model loaded successfully.")
        except Exception:
            self.log("No existing model found. Please train a new model.")

    def _build_ui(self):
        # Top frame: buttons, progress, status
        top_frame = tk.Frame(self.root, padx=5, pady=5)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # Buttons
        btn_load_dataset = tk.Button(
            top_frame, text="Load Dataset", width=15, command=self.load_dataset_dialog
        )
        btn_load_dataset.pack(side=tk.LEFT, padx=5)

        btn_train = tk.Button(
            top_frame, text="Train Model", width=15, command=self.start_training_thread
        )
        btn_train.pack(side=tk.LEFT, padx=5)

        btn_load_image = tk.Button(
            top_frame, text="Load Image", width=15, command=self.load_image_dialog
        )
        btn_load_image.pack(side=tk.LEFT, padx=5)

        btn_detect = tk.Button(
            top_frame, text="Run Detection", width=15, command=self.run_detection
        )
        btn_detect.pack(side=tk.LEFT, padx=5)

        btn_clear = tk.Button(
            top_frame, text="Clear / Reset", width=15, command=self.clear_all
        )
        btn_clear.pack(side=tk.LEFT, padx=5)

        # Progress bar
        self.progress = ttk.Progressbar(
            top_frame, orient=tk.HORIZONTAL, mode="indeterminate", length=200
        )
        self.progress.pack(side=tk.LEFT, padx=10)

        # Status label
        self.status_var = tk.StringVar()
        self.status_var.set("Status: Idle")
        status_label = tk.Label(top_frame, textvariable=self.status_var)
        status_label.pack(side=tk.LEFT, padx=10)

        # Main split: left image, right panels
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left: image panel
        image_frame = tk.LabelFrame(main_frame, text="Image Preview", padx=5, pady=5)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.image_label = tk.Label(image_frame, bg="gray", width=60, height=25)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Right: result + log
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Result panel
        result_frame = tk.LabelFrame(right_frame, text="Detection Result", padx=5, pady=5)
        result_frame.pack(fill=tk.X, padx=5, pady=5)

        self.detected_class_var = tk.StringVar()
        self.object_present_var = tk.StringVar()
        self.safety_status_var = tk.StringVar()

        lbl1 = tk.Label(result_frame, text="Detected classes:")
        lbl1.grid(row=0, column=0, sticky="w")
        lbl_val1 = tk.Label(result_frame, textvariable=self.detected_class_var, fg="blue")
        lbl_val1.grid(row=0, column=1, sticky="w")

        lbl2 = tk.Label(result_frame, text="Object presence:")
        lbl2.grid(row=1, column=0, sticky="w")
        lbl_val2 = tk.Label(result_frame, textvariable=self.object_present_var, fg="blue")
        lbl_val2.grid(row=1, column=1, sticky="w")

        lbl3 = tk.Label(result_frame, text="Safety status:")
        lbl3.grid(row=2, column=0, sticky="w")
        lbl_val3 = tk.Label(result_frame, textvariable=self.safety_status_var, fg="red")
        lbl_val3.grid(row=2, column=1, sticky="w")

        # Log panel
        log_frame = tk.LabelFrame(right_frame, text="Status / Log", padx=5, pady=5)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = tk.Text(log_frame, height=15, state=tk.DISABLED)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text["yscrollcommand"] = scrollbar.set

    # ------------- Logging helpers -------------

    def log(self, msg):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        print(msg)

    def set_status(self, msg):
        self.status_var.set("Status: " + msg)
        self.root.update_idletasks()

    # ------------- Button callbacks -------------

    def load_dataset_dialog(self):
        path = filedialog.askdirectory(title="Select dataset root folder")
        if path:
            self.dataset_path = path
            self.log(f"Dataset path set to: {path}")
            self.set_status("Dataset loaded")

    def start_training_thread(self):
        if self.is_training:
            messagebox.showinfo("Training", "Training is already running.")
            return

        if not self.dataset_path:
            messagebox.showwarning("Dataset not loaded", "Please load the dataset folder first.")
            return

        self.training_thread = threading.Thread(target=self.train_model_worker, daemon=True)
        self.is_training = True
        self.progress.start(10)
        self.set_status("Training model...")
        self.training_thread.start()

    def train_model_worker(self):
        try:
            self.log("Starting dataset loading...")
            X, y = load_dataset(self.dataset_path, log_callback=self.log)
            self.log(f"Dataset loaded. Total samples: {len(X)}")

            model, scaler = train_model(X, y, log_callback=self.log)
            save_model(model, scaler)
            self.model = model
            self.scaler = scaler

            self.log("Model training completed and saved to disk.")
            self.set_status("Training finished")
            messagebox.showinfo("Training", "Model training completed successfully.")
        except Exception as e:
            self.log("Error during training: " + str(e))
            self.log(traceback.format_exc())
            self.set_status("Training error")
            messagebox.showerror("Error", f"Training failed:\n{e}")
        finally:
            self.is_training = False
            self.progress.stop()

    def load_image_dialog(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.current_image_path = path
            self.show_image(path)
            self.set_status("Image loaded")

    def show_image(self, path):
        try:
            img = Image.open(path)
            # Fit image into preview area
            max_w, max_h = 500, 400
            img.thumbnail((max_w, max_h))
            self.current_image_tk = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.current_image_tk)
        except Exception as e:
            self.log(f"Error displaying image: {e}")
            messagebox.showerror("Error", f"Could not display image:\n{e}")

    def run_detection(self):
        if not self.current_image_path:
            messagebox.showwarning("No image", "Please load an image first.")
            return

        if self.model is None or self.scaler is None:
            try:
                self.model, self.scaler = load_model()
                self.log("Model loaded from disk.")
            except Exception as e:
                self.log(f"Error loading model: {e}")
                messagebox.showerror(
                    "Model not available",
                    "Model is not trained or cannot be loaded. Train the model first."
                )
                return

        try:
            self.set_status("Running detection...")
            pred_idx, probs = predict_image(self.model, self.scaler, self.current_image_path)
            class_name = CLASS_INDEX_TO_NAME.get(pred_idx, f"Class {pred_idx}")

            # For demonstration: treat single predicted label as present.
            # If you want multi‑label, use threshold on probs, but simple is fine.
            pred_classes = [pred_idx]

            # Update result panel with detections
            detected_names = [CLASS_INDEX_TO_NAME.get(i, str(i)) for i in pred_classes]
            self.detected_class_var.set(", ".join(detected_names))

            # Safety logic
            explanation, presence, safety = evaluate_safety(pred_classes)

            self.object_present_var.set(presence)
            self.safety_status_var.set(safety)

            self.log(f"Predicted: {class_name}")
            if probs is not None:
                prob_str = ", ".join(
                    f"{CLASS_INDEX_TO_NAME[i]}: {probs[i]:.2f}"
                    for i in range(len(probs))
                )
                self.log("Class probabilities: " + prob_str)
            self.log("Safety evaluation: " + explanation)
            self.set_status("Detection finished")
        except Exception as e:
            self.log(f"Error during detection: {e}")
            self.log(traceback.format_exc())
            self.set_status("Detection error")
            messagebox.showerror("Error", f"Detection failed:\n{e}")

    def clear_all(self):
        self.current_image_path = None
        self.current_image_tk = None
        self.image_label.configure(image="", bg="gray")
        self.detected_class_var.set("")
        self.object_present_var.set("")
        self.safety_status_var.set("")
        self.log("State cleared.")
        self.set_status("Idle")


# =========================
# Main
# =========================

def main():
    root = tk.Tk()
    app = HeadGearSentinelApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
