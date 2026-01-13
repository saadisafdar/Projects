"""
PPE Image Detection using YOLOv8
--------------------------------
A clean, beginner-friendly Tkinter desktop application for *image-only* inference
that loads a pre-trained YOLOv8 model (trained_model.pt) and performs PPE detection.

✅ Single .py file
✅ Tkinter GUI (clean & organized)
✅ Local inference only (no webcam/video/cloud)
✅ Clear ML workflow explanation for viva
✅ Uses only: ultralytics, opencv-python, Pillow, tkinter, numpy

Notes for Viva / Project Explanation:
- **Training**: Done in Google Colab (recommended due to hardware limits).
- **Deployment**: This local app demonstrates loading the trained model and running
  inference on static images.
- **Dataset Format**: YOLO-style folder layout is shown in the GUI for explanation.

Author: Your Name
Date: 2026-01-13
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import scrolledtext
from PIL import Image, ImageTk
import numpy as np
import cv2
from ultralytics import YOLO


class PPEDetectorApp(tk.Tk):
    """Main application class for PPE Image Detection using YOLOv8."""

    def __init__(self):
        super().__init__()
        self.title("PPE Image Detection using YOLOv8")
        self.geometry("1100x720")
        self.minsize(1000, 650)

        # ----------------------------
        # Internal state variables
        # ----------------------------
        self.model = None
        self.model_path = None
        self.image_path = None
        self.original_image = None  # PIL Image for preview
        self.output_image = None    # PIL Image with detections
        self.status_var = tk.StringVar(value="Ready")
        self.model_status_var = tk.StringVar(value="Not Loaded ❌")
        self.conf_threshold = 0.25
        self.img_size = 640  # Standard YOLOv8 inference size

        # Build the GUI layout
        self._build_gui()

    # ------------------------------------------------------------------
    # GUI Construction
    # ------------------------------------------------------------------
    def _build_gui(self):
        # Top Header Section
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X, padx=12, pady=(12, 6))

        title_lbl = ttk.Label(
            header_frame,
            text="PPE Image Detection using YOLOv8",
            font=("Segoe UI", 18, "bold")
        )
        title_lbl.pack(anchor=tk.W)

        subtitle_lbl = ttk.Label(
            header_frame,
            text="AI Lab / University Project",
            font=("Segoe UI", 11)
        )
        subtitle_lbl.pack(anchor=tk.W, pady=(2, 0))

        # Main content area with two columns
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        left_col = ttk.Frame(main_frame)
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_col = ttk.Frame(main_frame)
        right_col.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ---------------- Model Section ----------------
        model_lf = ttk.LabelFrame(left_col, text="Model Section")
        model_lf.pack(fill=tk.X, pady=(0, 8))

        model_btn = ttk.Button(model_lf, text="Load trained_model.pt", command=self.load_model)
        model_btn.grid(row=0, column=0, padx=8, pady=8, sticky=tk.W)

        self.model_status_lbl = ttk.Label(model_lf, textvariable=self.model_status_var)
        self.model_status_lbl.grid(row=0, column=1, padx=8, pady=8, sticky=tk.W)

        # Show selected model path
        self.model_path_var = tk.StringVar(value="Model Path: (not selected)")
        model_path_lbl = ttk.Label(model_lf, textvariable=self.model_path_var)
        model_path_lbl.grid(row=1, column=0, columnspan=2, padx=8, pady=(0, 8), sticky=tk.W)

        # ---------------- Image Selection Section ----------------
        image_lf = ttk.LabelFrame(left_col, text="Image Selection")
        image_lf.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        select_img_btn = ttk.Button(image_lf, text="Select Image", command=self.select_image)
        select_img_btn.grid(row=0, column=0, padx=8, pady=8, sticky=tk.W)

        self.image_path_var = tk.StringVar(value="Image Path: (not selected)")
        image_path_lbl = ttk.Label(image_lf, textvariable=self.image_path_var)
        image_path_lbl.grid(row=0, column=1, padx=8, pady=8, sticky=tk.W)

        # Original image preview panel
        preview_title = ttk.Label(image_lf, text="Original Image Preview", font=("Segoe UI", 10, "bold"))
        preview_title.grid(row=1, column=0, columnspan=2, padx=8, pady=(4, 0), sticky=tk.W)

        self.preview_canvas = tk.Label(image_lf, bg="#f5f5f5", relief=tk.SOLID, bd=1)
        self.preview_canvas.grid(row=2, column=0, columnspan=2, padx=8, pady=8, sticky=tk.NSEW)

        image_lf.columnconfigure(1, weight=1)
        image_lf.rowconfigure(2, weight=1)

        # ---------------- Detection Section ----------------
        detect_lf = ttk.LabelFrame(left_col, text="Detection")
        detect_lf.pack(fill=tk.X, pady=(0, 8))

        run_btn = ttk.Button(detect_lf, text="Run Detection", command=self.run_detection)
        run_btn.grid(row=0, column=0, padx=8, pady=8, sticky=tk.W)

        # ---------------- Results Panel ----------------
        results_lf = ttk.LabelFrame(right_col, text="Results Panel")
        results_lf.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        results_title = ttk.Label(results_lf, text="Detected classes and confidence scores", font=("Segoe UI", 10, "bold"))
        results_title.pack(anchor=tk.W, padx=8, pady=(8, 0))

        self.results_text = scrolledtext.ScrolledText(results_lf, height=12, wrap=tk.WORD, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ---------------- Output Image Panel ----------------
        output_lf = ttk.LabelFrame(right_col, text="Detection Output")
        output_lf.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        output_title = ttk.Label(output_lf, text="Image with bounding boxes, class names, and confidence (%)", font=("Segoe UI", 10, "bold"))
        output_title.pack(anchor=tk.W, padx=8, pady=(8, 0))

        self.output_canvas = tk.Label(output_lf, bg="#f5f5f5", relief=tk.SOLID, bd=1)
        self.output_canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ---------------- Training Workflow (Explanation Only) ----------------
        training_lf = ttk.LabelFrame(self, text="Training Workflow (Explanation)")
        training_lf.pack(fill=tk.X, padx=12, pady=(0, 8))

        training_text = (
            "Training is recommended on Google Colab due to hardware limitations.\n\n"
            "Typical YOLO-format dataset layout (for training/validation/test):\n"
            "dataset/\n"
            "├── data.yaml\n"
            "├── train/images\n"
            "├── train/labels\n"
            "├── valid/images\n"
            "├── valid/labels\n"
            "├── test/images\n"
            "├── test/labels\n\n"
            "Workflow Summary:\n"
            "1) Upload dataset to Colab or mount from Google Drive.\n"
            "2) Train YOLOv8 model (e.g., ultralytics) on GPU.\n"
            "3) Export / download the trained weights as 'trained_model.pt'.\n"
            "4) Place 'trained_model.pt' in the same folder as this app.\n"
            "5) Use this app to perform local image inference (deployment phase)."
        )
        training_lbl = ttk.Label(training_lf, text=training_text, justify=tk.LEFT)
        training_lbl.pack(anchor=tk.W, padx=8, pady=8)

        # ---------------- Status Bar ----------------
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_sep = ttk.Separator(self)
        status_sep.pack(fill=tk.X, side=tk.BOTTOM)
        status_lbl = ttk.Label(status_frame, textvariable=self.status_var, anchor=tk.W)
        status_lbl.pack(fill=tk.X, padx=12, pady=4)

        # Set a modern-looking style where available
        self._configure_style()

    def _configure_style(self):
        try:
            style = ttk.Style()
            # Prefer 'vista' or 'clam' if available for a modern look
            available = style.theme_names()
            for candidate in ("vista", "clam", "default"):
                if candidate in available:
                    style.theme_use(candidate)
                    break
            style.configure("TLabel", font=("Segoe UI", 10))
            style.configure("TButton", font=("Segoe UI", 10))
            style.configure("TLabelframe.Label", font=("Segoe UI", 11, "bold"))
        except Exception:
            # Fallback silently if styling fails
            pass

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------
    def set_status(self, text: str):
        """Update the status bar text."""
        self.status_var.set(text)
        self.update_idletasks()

    def _resize_for_preview(self, pil_image: Image.Image, max_w: int = 450, max_h: int = 320) -> Image.Image:
        """Resize image to fit preview panels while maintaining aspect ratio."""
        w, h = pil_image.size
        scale = min(max_w / float(w), max_h / float(h))
        scale = min(scale, 1.0)  # Never upscale
        new_size = (int(w * scale), int(h * scale))
        return pil_image.resize(new_size, Image.LANCZOS)

    def _update_preview_canvas(self, pil_image: Image.Image):
        """Display a PIL image on the preview canvas."""
        if pil_image is None:
            self.preview_canvas.configure(image="", text="No image selected")
            return
        resized = self._resize_for_preview(pil_image)
        self.preview_photo = ImageTk.PhotoImage(resized)
        self.preview_canvas.configure(image=self.preview_photo)

    def _update_output_canvas(self, pil_image: Image.Image):
        """Display a PIL image on the output canvas."""
        if pil_image is None:
            self.output_canvas.configure(image="", text="No output yet")
            return
        resized = self._resize_for_preview(pil_image, max_w=520, max_h=360)
        self.output_photo = ImageTk.PhotoImage(resized)
        self.output_canvas.configure(image=self.output_photo)

    def _append_results_text(self, lines):
        self.results_text.configure(state=tk.NORMAL)
        self.results_text.delete("1.0", tk.END)
        for ln in lines:
            self.results_text.insert(tk.END, ln + "\n")
        self.results_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def load_model(self):
        """Load the YOLOv8 trained model (trained_model.pt by default)."""
        self.set_status("Loading model…")
        default_path = os.path.join(os.getcwd(), "trained_model.pt")

        selected_path = None
        if os.path.isfile(default_path):
            selected_path = default_path
        else:
            # If default not found, let user choose a .pt file
            messagebox.showinfo(
                "Model not found",
                "Could not find 'trained_model.pt' in the current folder. Please select the trained model file (.pt)."
            )
            selected_path = filedialog.askopenfilename(
                title="Select trained YOLOv8 model (.pt)",
                filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
            )

        if not selected_path:
            self.set_status("Ready")
            return

        try:
            self.model = YOLO(selected_path)
            self.model_path = selected_path
            self.model_status_var.set("Loaded ✅")
            self.model_path_var.set(f"Model Path: {os.path.basename(selected_path)}")
            self.set_status("Model loaded successfully")
        except Exception as e:
            self.model = None
            self.model_path = None
            self.model_status_var.set("Not Loaded ❌")
            self.model_path_var.set("Model Path: (not selected)")
            self.set_status("Error: Failed to load model")
            messagebox.showerror("Model Load Error", str(e))

    def select_image(self):
        """Open a file dialog to select an image and show preview."""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image Files", "*.jpg;*.jpeg;*.png;*.bmp"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("PNG", "*.png"),
                ("Bitmap", "*.bmp"),
                ("All Files", "*.*")
            ]
        )
        if not file_path:
            return
        try:
            pil_img = Image.open(file_path).convert("RGB")
            self.original_image = pil_img
            self.image_path = file_path
            self.image_path_var.set(f"Image Path: {os.path.basename(file_path)}")
            self._update_preview_canvas(self.original_image)
            self.set_status("Ready")
        except Exception as e:
            self.set_status("Error: Could not open image")
            messagebox.showerror("Image Open Error", str(e))

    def run_detection(self):
        """Run YOLOv8 inference on the selected image and display results."""
        if self.model is None:
            messagebox.showwarning("Model not loaded", "Please load the trained YOLOv8 model first.")
            return
        if self.image_path is None:
            messagebox.showwarning("Image not selected", "Please select an image for detection.")
            return

        self.set_status("Running detection…")
        self._append_results_text(["Running detection…"])  # temporary feedback
        try:
            # Perform inference (CPU by default); conf threshold and image size are set
            results = self.model.predict(
                source=self.image_path,
                conf=self.conf_threshold,
                imgsz=self.img_size,
                verbose=False
            )

            if not results:
                self.set_status("Completed: No results returned")
                self._append_results_text(["No results returned by the model."])
                return

            r = results[0]
            names = r.names if hasattr(r, "names") else self.model.names

            # Read image with OpenCV for drawing
            bgr = cv2.imread(self.image_path)
            if bgr is None:
                raise RuntimeError("OpenCV could not read the image.")

            # Extract detections
            lines = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()  # (N, 4)
                conf = r.boxes.conf.cpu().numpy()  # (N,)
                cls = r.boxes.cls.cpu().numpy().astype(int)  # (N,)

                for i in range(xyxy.shape[0]):
                    x1, y1, x2, y2 = map(int, xyxy[i])
                    score = float(conf[i])
                    cls_id = int(cls[i])
                    cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
                    label_text = f"{cls_name} {score*100:.1f}%"

                    # Draw bounding box
                    color = (0, 180, 0)  # green in BGR
                    cv2.rectangle(bgr, (x1, y1), (x2, y2), color, thickness=2)

                    # Draw label background for readability
                    (tw, th), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(bgr, label_text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                    lines.append(f"Class: {cls_name}    Confidence: {score*100:.2f}%")
            else:
                lines.append("No objects detected.")

            # Convert image to RGB and then to PIL for display
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_out = Image.fromarray(rgb)
            self.output_image = pil_out
            self._update_output_canvas(self.output_image)
            self._append_results_text(lines)

            self.set_status("Completed")
        except Exception as e:
            self.set_status("Error during detection")
            messagebox.showerror("Detection Error", str(e))


def main():
    app = PPEDetectorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
