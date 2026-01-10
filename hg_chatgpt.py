import os
import shutil
import random
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import yaml
from ultralytics import YOLO

# ---------------- CONFIG ----------------

CLASSES = [
    "Person",
    "Helmet",
    "Goggles",
    "Mask",
    "Gloves",
    "Safety Vest",
    "Boots"
]

MODEL_NAME = "yolov8n.pt"
DATASET_YOLO = "dataset_yolo"
MODEL_OUTPUT = "runs/detect/train/weights/best.pt"

# ---------------- DATASET PREP ----------------

def prepare_dataset(raw_path):
    if os.path.exists(DATASET_YOLO):
        shutil.rmtree(DATASET_YOLO)

    for split in ["train", "val"]:
        os.makedirs(f"{DATASET_YOLO}/images/{split}", exist_ok=True)
        os.makedirs(f"{DATASET_YOLO}/labels/{split}", exist_ok=True)

    image_entries = []

    for class_id, class_name in enumerate(CLASSES):
        class_dir = os.path.join(raw_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file in os.listdir(class_dir):
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_entries.append((class_id, class_name, os.path.join(class_dir, file)))

    random.shuffle(image_entries)
    split_idx = int(0.8 * len(image_entries))

    for idx, (class_id, class_name, img_path) in enumerate(image_entries):
        split = "train" if idx < split_idx else "val"
        img_name = f"{class_name}_{idx}.jpg"

        shutil.copy(img_path, f"{DATASET_YOLO}/images/{split}/{img_name}")

        # YOLO label: one centered bounding box
        label_path = f"{DATASET_YOLO}/labels/{split}/{img_name.replace('.jpg', '.txt')}"
        with open(label_path, "w") as f:
            f.write(f"{class_id} 0.5 0.5 0.8 0.8\n")

    data_yaml = {
        "path": os.path.abspath(DATASET_YOLO),
        "train": "images/train",
        "val": "images/val",
        "names": CLASSES
    }

    with open(f"{DATASET_YOLO}/data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

# ---------------- TRAINING ----------------

def train_model(status_label):
    try:
        status_label.config(text="Training in progress...")
        model = YOLO(MODEL_NAME)
        model.train(
            data=f"{DATASET_YOLO}/data.yaml",
            epochs=30,
            imgsz=640,
            batch=8
        )
        status_label.config(text="Training completed")
    except Exception as e:
        messagebox.showerror("Training Error", str(e))

# ---------------- DETECTION ----------------

def run_detection(image_path, image_label, result_text):
    model = YOLO(MODEL_OUTPUT)

    img = cv2.imread(image_path)
    results = model(img)[0]

    detected = {cls: 0 for cls in CLASSES}

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            detected[CLASSES[cls_id]] += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                CLASSES[cls_id],
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Safety logic
    status = "SAFE"
    if detected["Person"] > 0:
        for ppe in CLASSES[1:]:
            if detected[ppe] == 0:
                status = "UNSAFE"
                break

    text = "Detection Results:\n\n"
    for cls in CLASSES:
        text += f"{cls}: {detected[cls]}\n"
    text += f"\nOverall Status: {status}"

    result_text.set(text)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((500, 500))
    tk_img = ImageTk.PhotoImage(img_pil)

    image_label.config(image=tk_img)
    image_label.image = tk_img

# ---------------- GUI ----------------

root = tk.Tk()
root.title("HeadGear Sentinel - PPE Detection")
root.geometry("900x600")

dataset_path = tk.StringVar()
result_text = tk.StringVar()

def load_dataset():
    path = filedialog.askdirectory()
    if path:
        dataset_path.set(path)
        prepare_dataset(path)
        messagebox.showinfo("Dataset Loaded", "Dataset prepared successfully")

def start_training():
    t = threading.Thread(target=train_model, args=(status_label,))
    t.start()

def load_image():
    img_path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.png *.jpeg")]
    )
    if img_path:
        run_detection(img_path, image_label, result_text)

# Layout
control_frame = tk.Frame(root)
control_frame.pack(side=tk.LEFT, padx=10, pady=10)

tk.Button(control_frame, text="Load Dataset", width=20, command=load_dataset).pack(pady=5)
tk.Button(control_frame, text="Train Model", width=20, command=start_training).pack(pady=5)
tk.Button(control_frame, text="Load Image & Detect", width=20, command=load_image).pack(pady=5)

status_label = tk.Label(control_frame, text="Idle", fg="blue")
status_label.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(side=tk.RIGHT, padx=10, pady=10)

result_label = tk.Label(root, textvariable=result_text, justify=tk.LEFT, font=("Arial", 11))
result_label.pack(side=tk.BOTTOM, pady=10)

root.mainloop()
