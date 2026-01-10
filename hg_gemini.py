import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
# from tensorflow.keras import layers, models
from keras import layers, models
from keras.applications import MobileNetV2
from keras.preprocessing.image import load_img, img_to_array


class HeadGearSentinel:
    def __init__(self, root):
        self.root = root
        self.root.title("HeadGear Sentinel - PPE Object Detection")
        self.root.geometry("900x700")
        
        # Project Variables
        self.classes = ['Person', 'Helmet', 'Goggles', 'Mask', 'Gloves', 'Safety Vest', 'Boots']
        self.model = None
        self.img_size = (128, 128)
        
        self.setup_gui()

    def setup_gui(self):
        # Header
        tk.Label(self.root, text="HeadGear Sentinel", font=("Helvetica", 24, "bold")).pack(pady=10)
        
        # Control Panel
        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(pady=10)
        
        tk.Button(ctrl_frame, text="Load Dataset & Train", command=self.train_model, bg="#4CAF50", fg="white").grid(row=0, column=0, padx=5)
        tk.Button(ctrl_frame, text="Load Single Image", command=self.load_and_detect, bg="#2196F3", fg="white").grid(row=0, column=1, padx=5)
        
        # Image Display
        self.img_label = tk.Label(self.root, text="No Image Loaded", bg="gray", width=50, height=20)
        self.img_label.pack(pady=20)
        
        # Results Display
        self.res_label = tk.Label(self.root, text="Detection Results: N/A", font=("Helvetica", 14), justify="left")
        self.res_label.pack(pady=10)
        
        self.status_label = tk.Label(self.root, text="Safety Status: Unknown", font=("Helvetica", 16, "bold"))
        self.status_label.pack(pady=5)

    def build_model(self):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.classes), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        data_path = filedialog.askdirectory(title="Select Dataset Folder (containing subfolders for classes)")
        if not data_path: return

        try:
            # Loading dataset from structured folders
            train_ds = tf.keras.utils.image_dataset_from_directory(
                data_path,
                image_size=self.img_size,
                batch_size=32,
                label_mode='int'
            )
            
            messagebox.showinfo("Training", "Starting real training. The GUI might freeze briefly...")
            self.model = self.build_model()
            self.model.fit(train_ds, epochs=5) # 5 epochs for demonstration
            messagebox.showinfo("Success", "Model Trained Successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to train: {str(e)}")

    def load_and_detect(self):
        if self.model is None:
            messagebox.showwarning("Warning", "Please train the model first!")
            return
            
        file_path = filedialog.askopenfilename()
        if not file_path: return

        # Process Image for Display
        img = Image.open(file_path).resize((400, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk, text="")
        self.img_label.image = img_tk

        # Inference
        test_img = Image.open(file_path).resize(self.img_size)
        img_array = tf.keras.utils.img_to_array(test_img)
        img_array = tf.expand_dims(img_array, 0)
        
        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        detected_idx = np.argmax(score)
        detected_class = self.classes[detected_idx]
        
        # Display Results
        results_text = f"Detected: {detected_class} ({100 * np.max(score):.2f}% confidence)"
        self.res_label.config(text=results_text)
        
        # Logic for Safety Status
        if detected_class in ['Helmet', 'Safety Vest']:
            self.status_label.config(text="Status: SAFE", fg="green")
        else:
            self.status_label.config(text="Status: UNSAFE / COMPLIANCE NEEDED", fg="red")

if __name__ == "__main__":
    root = tk.Tk()
    app = HeadGearSentinel(root)
    root.mainloop()