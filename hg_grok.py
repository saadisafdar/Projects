import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
# from tensorflow.keras import layers, models
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras import layers, models
from keras.applications import MobileNetV2
from keras.preprocessing.image import load_img, img_to_array


classes = ["Person", "Helmet", "Goggles", "Mask", "Gloves", "Safety Vest", "Boots"]
num_classes = len(classes)
model = None
dataset_dir = None
current_image = None
image_label = None
result_text = None

def build_model():
    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base.trainable = False
    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_dataset(dir):
    image_paths = []
    labels = []
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    for cls in classes:
        cls_dir = os.path.join(dir, cls)
        if os.path.isdir(cls_dir):
            for img_file in os.listdir(cls_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(cls_dir, img_file))
                    label = np.zeros(num_classes)
                    label[class_to_idx[cls]] = 1
                    labels.append(label)
    return image_paths, np.array(labels)

def train_model():
    global model, dataset_dir
    if not dataset_dir:
        messagebox.showerror("Error", "Load dataset first")
        return
    paths, labels = load_dataset(dataset_dir)
    if len(paths) == 0:
        messagebox.showerror("Error", "No images found")
        return
    def preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img /= 255.0
        return img, label
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels)).map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    model = build_model()
    model.fit(dataset, epochs=10)
    model.save('ppe_model.h5')
    messagebox.showinfo("Success", "Model trained")

def load_dataset_gui():
    global dataset_dir
    dataset_dir = filedialog.askdirectory(title="Select Dataset Folder")
    if dataset_dir:
        messagebox.showinfo("Success", "Dataset loaded")

def load_image_gui():
    global current_image
    file = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file:
        display_image(file)
        current_image = file

def load_folder_gui():
    folder = filedialog.askdirectory(title="Select Image Folder")
    if folder:
        process_folder(folder)

def run_detection():
    global model, current_image
    if not model:
        try:
            model = models.load_model('ppe_model.h5')
        except:
            messagebox.showerror("Error", "Train model first")
            return
    if current_image:
        present, status = predict_image(model, current_image)
        result_text.config(state=tk.NORMAL)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Present: {', '.join(present)}\nStatus: {status}")
        result_text.config(state=tk.DISABLED)
    else:
        messagebox.showerror("Error", "Load image first")

def predict_image(model, path):
    img = load_img(path, target_size=(224, 224))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr)[0]
    present = [classes[i] for i in range(num_classes) if preds[i] > 0.5]
    missing = [classes[i] for i in range(1, num_classes) if preds[i] <= 0.5]
    has_person = preds[0] > 0.5
    if not has_person:
        status = "No person detected"
    elif not missing:
        status = "Safe"
    else:
        status = f"Unsafe, missing: {', '.join(missing)}"
    return present, status

def display_image(path):
    img = Image.open(path)
    img = img.resize((300, 300))
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo)
    image_label.image = photo

def process_folder(folder):
    if not model:
        try:
            model = models.load_model('ppe_model.h5')
        except:
            messagebox.showerror("Error", "Train model first")
            return
    results = []
    for file in os.listdir(folder):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, file)
            present, status = predict_image(model, path)
            results.append(f"{file}: Present - {', '.join(present)}; Status - {status}")
    result_text.config(state=tk.NORMAL)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "\n".join(results))
    result_text.config(state=tk.DISABLED)

root = tk.Tk()
root.title("HeadGear Sentinel")

frame = tk.Frame(root)
frame.pack()

tk.Button(frame, text="Load Dataset", command=load_dataset_gui).pack()
tk.Button(frame, text="Train Model", command=train_model).pack()
tk.Button(frame, text="Load Image", command=load_image_gui).pack()
tk.Button(frame, text="Load Folder", command=load_folder_gui).pack()
tk.Button(frame, text="Run Detection", command=run_detection).pack()

image_label = tk.Label(root)
image_label.pack()

result_text = scrolledtext.ScrolledText(root, width=50, height=10, state=tk.DISABLED)
result_text.pack()

root.mainloop()