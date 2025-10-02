import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("🖼️ Tkinter Image Viewer")
        self.root.geometry("900x700")
        self.root.configure(bg="#f4f4f4")

        self.images = []
        self.current_index = 0
        self.zoom_factor = 1.0

        # UI Elements
        self.label = tk.Label(root, text="Open a folder to view images", font=("Arial", 14), bg="#f4f4f4")
        self.label.pack(pady=10)

        self.canvas = tk.Canvas(root, bg="black", width=800, height=500)
        self.canvas.pack(pady=10)

        # Buttons
        button_frame = tk.Frame(root, bg="#f4f4f4")
        button_frame.pack()

        tk.Button(button_frame, text="📂 Open Folder", command=self.open_folder).grid(row=0, column=0, padx=5)
        tk.Button(button_frame, text="⬅ Prev", command=self.prev_image).grid(row=0, column=1, padx=5)
        tk.Button(button_frame, text="Next ➡", command=self.next_image).grid(row=0, column=2, padx=5)
        tk.Button(button_frame, text="🔍 Zoom In", command=self.zoom_in).grid(row=0, column=3, padx=5)
        tk.Button(button_frame, text="🔎 Zoom Out", command=self.zoom_out).grid(row=0, column=4, padx=5)
        tk.Button(button_frame, text="⛶ Fullscreen", command=self.toggle_fullscreen).grid(row=0, column=5, padx=5)

        self.fullscreen = False

    def open_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.images = [os.path.join(folder, f) for f in os.listdir(folder)
                           if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
            if not self.images:
                messagebox.showerror("Error", "No images found in this folder.")
                return
            self.current_index = 0
            self.zoom_factor = 1.0
            self.show_image()

    def show_image(self):
        if not self.images:
            return
        image_path = self.images[self.current_index]
        img = Image.open(image_path)

        # Apply zoom
        w, h = img.size
        img = img.resize((int(w * self.zoom_factor), int(h * self.zoom_factor)), Image.Resampling.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(400, 250, image=self.tk_image, anchor="center")
        self.label.config(text=f"{os.path.basename(image_path)} ({self.current_index+1}/{len(self.images)})")

    def next_image(self):
        if self.images:
            self.current_index = (self.current_index + 1) % len(self.images)
            self.zoom_factor = 1.0
            self.show_image()

    def prev_image(self):
        if self.images:
            self.current_index = (self.current_index - 1) % len(self.images)
            self.zoom_factor = 1.0
            self.show_image()

    def zoom_in(self):
        if self.images:
            self.zoom_factor *= 1.2
            self.show_image()

    def zoom_out(self):
        if self.images:
            self.zoom_factor /= 1.2
            self.show_image()

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

# Run the App
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewer(root)
    root.mainloop()
