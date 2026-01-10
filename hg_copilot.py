#!/usr/bin/env python3
"""
HeadGear Sentinel – PPE Object Detection for Construction Workers (Image-Only)
-----------------------------------------------------------------------------
Single-file academic project that demonstrates:
- Loading an image dataset from a folder-per-class structure
- Training a multi-label CNN (transfer learning with ResNet-18)
- Running image-only inference (single image or folder)
- A Tkinter GUI for end-to-end interaction

Supported classes (up to 7):
  Person, Helmet, Goggles, Mask, Gloves, Safety Vest, Boots

Author: Your Name
License: MIT (for academic use)
"""

import os
import sys
import threading
import time
import traceback
import math
import random
import csv
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms

APP_TITLE = "HeadGear Sentinel – PPE Object Detection (Image-Only)"

CANONICAL_CLASSES = [
    "Person",
    "Helmet",
    "Goggles",
    "Mask",
    "Gloves",
    "Safety Vest",
    "Boots",
]

CLASS_ALIASES = {
    "person": "Person",
    "people": "Person",
    "helmet": "Helmet",
    "hardhat": "Helmet",
    "hard_hat": "Helmet",
    "goggles": "Goggles",
    "goggle": "Goggles",
    "safety_goggles": "Goggles",
    "eye_protection": "Goggles",
    "mask": "Mask",
    "face_mask": "Mask",
    "facemask": "Mask",
    "respirator": "Mask",
    "glove": "Gloves",
    "gloves": "Gloves",
    "hand_gloves": "Gloves",
    "safety_vest": "Safety Vest",
    "safetyvest": "Safety Vest",
    "vest": "Safety Vest",
    "reflective_vest": "Safety Vest",
    "hi_vis_vest": "Safety Vest",
    "boot": "Boots",
    "boots": "Boots",
    "safety_boots": "Boots",
    "work_boots": "Boots",
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")

def map_to_canonical(name: str) -> Optional[str]:
    key = normalize_name(name)
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    if key.endswith("s") and key[:-1] in CLASS_ALIASES:
        return CLASS_ALIASES[key[:-1]]
    if key + "s" in CLASS_ALIASES:
        return CLASS_ALIASES[key + "s"]
    flat = key.replace("_", "")
    if flat in CLASS_ALIASES:
        return CLASS_ALIASES[flat]
    for c in CANONICAL_CLASSES:
        if normalize_name(c) == key:
            return c
    return None

@dataclass
class ImageItem:
    path: str
    label_vec: np.ndarray

class FolderPerClassMultiLabelDataset(Dataset):
    """Dataset that interprets folder-per-class structure for multi-label training.

    Images in a recognized class folder are labeled positive for that class and negative for others.
    Optionally, filename tokens that match other class names also mark those classes as positive.
    """
    def __init__(self, root: str, class_order: List[str], img_size: int = 224, augment: bool = True):
        super().__init__()
        self.root = root
        self.class_order = class_order
        self.class_to_idx = {c: i for i, c in enumerate(class_order)}
        self.img_size = img_size
        self.items: List[ImageItem] = []

        if augment:
            self.tfms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tfms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        self._scan()

    def _scan(self):
        per_class_paths: Dict[str, List[str]] = {c: [] for c in self.class_order}
        for entry in os.scandir(self.root):
            if entry.is_dir():
                canonical = map_to_canonical(entry.name)
                if canonical and canonical in per_class_paths:
                    for dirpath, _, filenames in os.walk(entry.path):
                        for fn in filenames:
                            if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                                per_class_paths[canonical].append(os.path.join(dirpath, fn))
        for canonical, paths in per_class_paths.items():
            for p in paths:
                label = np.zeros(len(self.class_order), dtype=np.float32)
                label[self.class_to_idx[canonical]] = 1.0
                tokens = set(normalize_name(os.path.basename(p)).replace(".", "_").split("_"))
                for token in list(tokens):
                    mapped = map_to_canonical(token)
                    if mapped and mapped in self.class_to_idx:
                        label[self.class_to_idx[mapped]] = 1.0
                self.items.append(ImageItem(path=p, label_vec=label))
        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            img = Image.open(item.path).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        x = self.tfms(img)
        y = torch.from_numpy(item.label_vec.copy())
        return x, y

class MultiLabelResNet(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.backbone_name = "resnet18"
        self.pretrained_ok = False
        try:
            try:
                weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
                backbone = torchvision.models.resnet18(weights=weights)
            except Exception:
                backbone = torchvision.models.resnet18(pretrained=pretrained)
            self.pretrained_ok = pretrained and True
        except Exception:
            backbone = torchvision.models.resnet18(weights=None)
            self.pretrained_ok = False
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(in_features, num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 16
    lr: float = 1e-3
    val_split: float = 0.2
    img_size: int = 224
    use_pretrained: bool = True
    num_workers: int = 0

class Trainer:
    def __init__(self, dataset: FolderPerClassMultiLabelDataset, class_names: List[str], cfg: TrainConfig, device: torch.device):
        self.dataset = dataset
        self.class_names = class_names
        self.cfg = cfg
        self.device = device
        self.model: Optional[MultiLabelResNet] = None
        self.history = []

    def _split(self):
        n = len(self.dataset)
        n_val = max(1, int(n * self.cfg.val_split))
        n_train = max(1, n - n_val)
        train_set, val_set = random_split(self.dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
        return train_set, val_set

    def _compute_pos_weight(self, loader: DataLoader, num_classes: int) -> torch.Tensor:
        pos = torch.zeros(num_classes)
        total = 0
        for _, y in loader:
            pos += y.sum(dim=0)
            total += y.shape[0]
        neg = total - pos
        pos_weight = torch.where(pos > 0, neg / torch.clamp(pos, min=1), torch.ones_like(pos))
        return pos_weight

    def train(self, progress_cb=None, status_cb=None, finish_cb=None, save_path: str = "headgear_sentinel_model.pth"):
        train_set, val_set = self._split()
        train_loader = DataLoader(train_set, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)
        val_loader = DataLoader(val_set, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)
        num_classes = len(self.class_names)
        self.model = MultiLabelResNet(num_classes=num_classes, pretrained=self.cfg.use_pretrained)
        self.model.to(self.device)
        tmp_loader = DataLoader(train_set, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)
        pos_weight = self._compute_pos_weight(tmp_loader, num_classes=num_classes).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        best_val = float('inf')
        best_state = None
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            self.model.train()
            run_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                run_loss += loss.item()
                n_batches += 1
                if progress_cb:
                    progress_cb(epoch, self.cfg.epochs, n_batches, math.ceil(len(train_loader)))
            train_loss = run_loss / max(1, n_batches)
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits = self.model(xb)
                    loss = criterion(logits, yb)
                    val_loss += loss.item()
                    val_batches += 1
            val_loss = val_loss / max(1, val_batches)
            scheduler.step(val_loss)
            epoch_time = time.time() - t0
            info = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': optimizer.param_groups[0]['lr'], 'time_sec': epoch_time}
            self.history.append(info)
            if status_cb:
                status_cb(f"Epoch {epoch}/{self.cfg.epochs} – train_loss={train_loss:.4f} val_loss={val_loss:.4f} (lr {optimizer.param_groups[0]['lr']:.1e})")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        if best_state is not None:
            torch.save({'model_state': best_state, 'class_names': self.class_names, 'img_size': self.cfg.img_size, 'pretrained_backbone': self.model.pretrained_ok, 'backbone': self.model.backbone_name, 'history': self.history}, save_path)
        if finish_cb:
            finish_cb(save_path, best_val)

class InferenceEngine:
    def __init__(self, device: torch.device):
        self.device = device
        self.model: Optional[MultiLabelResNet] = None
        self.class_names: List[str] = []
        self.img_size: int = 224
        self.ready: bool = False
        self.tfms = None
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.class_names = ckpt['class_names']
        self.img_size = ckpt.get('img_size', 224)
        self.model = MultiLabelResNet(num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(ckpt['model_state'])
        self.model.to(self.device)
        self.model.eval()
        self.tfms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.ready = True
    @torch.no_grad()
    def predict_image(self, img_path: str, threshold: float = 0.5) -> Tuple[List[float], List[int]]:
        if not self.ready:
            raise RuntimeError("Model not loaded.")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.fromarray(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        x = self.tfms(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy().tolist()
        pred = [1 if p >= threshold else 0 for p in probs]
        return probs, pred
    def safety_status(self, pred: List[int]) -> str:
        try:
            idx_person = self.class_names.index("Person")
        except ValueError:
            idx_person = None
        if idx_person is not None and pred[idx_person] == 0:
            return "No person"
        ppe_present = True
        for i, c in enumerate(self.class_names):
            if c == "Person":
                continue
            if pred[i] == 0:
                ppe_present = False
        return "SAFE" if ppe_present else "UNSAFE"

class App:
    def __init__(self, root):
        self.root = root
        root.title(APP_TITLE)
        root.geometry("1080x720")
        self.dataset_root: Optional[str] = None
        self.discovered_classes: List[str] = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.trainer: Optional[Trainer] = None
        self.engine = InferenceEngine(self.device)
        self.model_path: str = os.path.abspath("headgear_sentinel_model.pth")
        self.cfg = TrainConfig()
        self._build_ui()
    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True)
        self.tab_data = ttk.Frame(nb)
        self.tab_infer = ttk.Frame(nb)
        self.tab_results = ttk.Frame(nb)
        nb.add(self.tab_data, text="Dataset & Training")
        nb.add(self.tab_infer, text="Inference")
        nb.add(self.tab_results, text="Results")
        frm_ds = ttk.LabelFrame(self.tab_data, text="Dataset")
        frm_ds.pack(fill=tk.X, padx=10, pady=10)
        self.lbl_ds = ttk.Label(frm_ds, text="Dataset root: (not selected)")
        self.lbl_ds.pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(frm_ds, text="Select Folder", command=self.select_dataset).pack(side=tk.LEFT, padx=5)
        ttk.Button(frm_ds, text="Rescan", command=self.rescan_classes).pack(side=tk.LEFT, padx=5)
        self.lst_classes = tk.Listbox(self.tab_data, height=7, selectmode=tk.MULTIPLE)
        self.lst_classes.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(self.tab_data, text="Discovered classes (subset of: Person, Helmet, Goggles, Mask, Gloves, Safety Vest, Boots)").pack(padx=10)
        frm_cfg = ttk.LabelFrame(self.tab_data, text="Training Configuration")
        frm_cfg.pack(fill=tk.X, padx=10, pady=10)
        self.var_epochs = tk.IntVar(value=self.cfg.epochs)
        self.var_bs = tk.IntVar(value=self.cfg.batch_size)
        self.var_lr = tk.DoubleVar(value=self.cfg.lr)
        self.var_val = tk.DoubleVar(value=self.cfg.val_split)
        self.var_img = tk.IntVar(value=self.cfg.img_size)
        self.var_pre = tk.BooleanVar(value=self.cfg.use_pretrained)
        def add_row(parent, r, label, widget):
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky=tk.W, padx=5, pady=2)
            widget.grid(row=r, column=1, sticky=tk.W, padx=5, pady=2)
        add_row(frm_cfg, 0, "Epochs:", ttk.Spinbox(frm_cfg, from_=1, to=200, textvariable=self.var_epochs, width=10))
        add_row(frm_cfg, 1, "Batch size:", ttk.Spinbox(frm_cfg, from_=1, to=256, textvariable=self.var_bs, width=10))
        add_row(frm_cfg, 2, "Learning rate:", ttk.Entry(frm_cfg, textvariable=self.var_lr, width=12))
        add_row(frm_cfg, 3, "Validation split (0-0.9):", ttk.Entry(frm_cfg, textvariable=self.var_val, width=12))
        add_row(frm_cfg, 4, "Image size:", ttk.Spinbox(frm_cfg, from_=64, to=512, increment=32, textvariable=self.var_img, width=10))
        add_row(frm_cfg, 5, "Use pretrained backbone (needs internet if not cached):", ttk.Checkbutton(frm_cfg, variable=self.var_pre))
        self.btn_train = ttk.Button(self.tab_data, text="Train Model", command=self.on_train)
        self.btn_train.pack(padx=10, pady=10)
        self.pb = ttk.Progressbar(self.tab_data, orient=tk.HORIZONTAL, mode='determinate', maximum=100)
        self.pb.pack(fill=tk.X, padx=10, pady=5)
        self.txt_log = tk.Text(self.tab_data, height=10)
        self.txt_log.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
        frm_model = ttk.LabelFrame(self.tab_infer, text="Model")
        frm_model.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(frm_model, text="Load Model (.pth)", command=self.on_load_model).pack(side=tk.LEFT, padx=5, pady=5)
        self.lbl_model = ttk.Label(frm_model, text=f"Current: {self.model_path if os.path.exists(self.model_path) else '(none)'}")
        self.lbl_model.pack(side=tk.LEFT, padx=5)
        frm_thresh = ttk.Frame(self.tab_infer)
        frm_thresh.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(frm_thresh, text="Prediction threshold:").pack(side=tk.LEFT)
        self.var_thresh = tk.DoubleVar(value=0.5)
        self.scl_thresh = ttk.Scale(frm_thresh, from_=0.1, to=0.9, variable=self.var_thresh, orient=tk.HORIZONTAL)
        self.scl_thresh.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.lbl_thresh = ttk.Label(frm_thresh, text="0.50")
        self.lbl_thresh.pack(side=tk.LEFT)
        self.var_thresh.trace_add('write', lambda *args: self.lbl_thresh.config(text=f"{self.var_thresh.get():.2f}"))
        frm_infer = ttk.LabelFrame(self.tab_infer, text="Run Inference (Image-Only)")
        frm_infer.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(frm_infer, text="Open Image", command=self.on_open_image).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(frm_infer, text="Open Folder", command=self.on_open_folder).pack(side=tk.LEFT, padx=5, pady=5)
        self.canvas = tk.Canvas(self.tab_infer, width=512, height=384, bg='gray90')
        self.canvas.pack(padx=10, pady=10)
        self.img_on_canvas = None
        self.frm_results = ttk.Frame(self.tab_infer)
        self.frm_results.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.tree = ttk.Treeview(self.frm_results, columns=["image"] + CANONICAL_CLASSES + ["safety"], show='headings')
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for col in ["image"] + CANONICAL_CLASSES + ["safety"]:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor=tk.CENTER)
        self.scroll_y = ttk.Scrollbar(self.frm_results, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=self.scroll_y.set)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        frm_export = ttk.Frame(self.tab_infer)
        frm_export.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(frm_export, text="Export Results to CSV", command=self.on_export_csv).pack(side=tk.LEFT, padx=5)
        ttk.Label(self.tab_results, text="Training History / Logs").pack(padx=10, pady=10)
        self.txt_hist = tk.Text(self.tab_results, height=30)
        self.txt_hist.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.status = ttk.Label(self.root, text=f"Device: {self.device}")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
    def log(self, msg: str):
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)
        self.status.config(text=msg)
    def select_dataset(self):
        path = filedialog.askdirectory(title="Select dataset root (folders per class)")
        if path:
            self.dataset_root = path
            self.lbl_ds.config(text=f"Dataset root: {path}")
            self.rescan_classes()
    def rescan_classes(self):
        self.lst_classes.delete(0, tk.END)
        self.discovered_classes = []
        if not self.dataset_root:
            return
        seen = set()
        for entry in os.scandir(self.dataset_root):
            if entry.is_dir():
                canonical = map_to_canonical(entry.name)
                if canonical and canonical not in seen and canonical in CANONICAL_CLASSES:
                    seen.add(canonical)
        self.discovered_classes = sorted(list(seen), key=lambda c: CANONICAL_CLASSES.index(c))
        if not self.discovered_classes:
            messagebox.showwarning("No classes found", "No recognizable class folders found under the selected root.\n\nExpected any of: " + ", ".join(CANONICAL_CLASSES))
            return
        for c in self.discovered_classes:
            self.lst_classes.insert(tk.END, c)
            self.lst_classes.selection_set(tk.END)
        self.log(f"Discovered classes: {', '.join(self.discovered_classes)}")
    def on_train(self):
        if not self.dataset_root:
            messagebox.showerror("Dataset not set", "Please select a dataset root folder first.")
            return
        sel = [self.lst_classes.get(i) for i in self.lst_classes.curselection()]
        if not sel:
            messagebox.showerror("No classes selected", "Please select at least one class to train.")
            return
        try:
            cfg = self.cfg
            cfg.epochs = int(self.var_epochs.get())
            cfg.batch_size = int(self.var_bs.get())
            cfg.lr = float(self.var_lr.get())
            cfg.val_split = float(self.var_val.get())
            cfg.img_size = int(self.var_img.get())
            cfg.use_pretrained = bool(self.var_pre.get())
        except Exception:
            messagebox.showerror("Config error", "Please check training parameter values.")
            return
        if not (0.0 < self.cfg.val_split < 0.9):
            messagebox.showerror("Config error", "Validation split must be between 0 and 0.9.")
            return
        try:
            ds = FolderPerClassMultiLabelDataset(self.dataset_root, class_order=sel, img_size=self.cfg.img_size, augment=True)
        except Exception as e:
            messagebox.showerror("Dataset error", f"Failed to load dataset: {e}")
            return
        if len(ds) < 4:
            messagebox.showwarning("Small dataset", "Fewer than 4 images found. Training may not be meaningful.")
        self.trainer = Trainer(ds, class_names=sel, cfg=self.cfg, device=self.device)
        self.btn_train.config(state=tk.DISABLED)
        self.pb['value'] = 0
        def progress_cb(epoch, total_epochs, batch_idx, batch_total):
            if batch_total > 0:
                p_epoch = (batch_idx / batch_total)
                p_total = (epoch - 1 + p_epoch) / max(1, total_epochs)
                self.pb['value'] = int(100 * p_total)
                self.root.update_idletasks()
        def status_cb(msg):
            self.log(msg)
            self.txt_hist.insert(tk.END, msg + "\n")
            self.txt_hist.see(tk.END)
        def finish_cb(path, best_val):
            self.log(f"Training complete. Best val_loss={best_val:.4f}. Model saved to: {path}")
            self.btn_train.config(state=tk.NORMAL)
            self.lbl_model.config(text=f"Current: {path}")
            try:
                self.engine.load(path)
            except Exception as e:
                self.log(f"Auto-load failed: {e}")
        def worker():
            try:
                self.trainer.train(progress_cb=progress_cb, status_cb=status_cb, finish_cb=finish_cb, save_path=self.model_path)
            except Exception as e:
                traceback.print_exc()
                messagebox.showerror("Training error", str(e))
                self.btn_train.config(state=tk.NORMAL)
        threading.Thread(target=worker, daemon=True).start()
    def on_load_model(self):
        path = filedialog.askopenfilename(title="Select model file", filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")])
        if path:
            try:
                self.engine.load(path)
                self.model_path = path
                self.lbl_model.config(text=f"Current: {path}")
                self.log("Model loaded successfully.")
            except Exception as e:
                messagebox.showerror("Load error", f"Failed to load model: {e}")
    def show_image_on_canvas(self, img_path: str):
        try:
            img = Image.open(img_path)
        except Exception:
            img = Image.fromarray(np.zeros((512, 384, 3), dtype=np.uint8))
        cw, ch = 512, 384
        img.thumbnail((cw, ch))
        ph = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.img_on_canvas = ph
        self.canvas.create_image(cw // 2, ch // 2, image=ph)
    def clear_tree(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
    def insert_result_row(self, image_path: str, probs: List[float], pred: List[int], safety: str):
        row = [os.path.basename(image_path)]
        for cname in CANONICAL_CLASSES:
            if self.engine.ready and cname in self.engine.class_names:
                i = self.engine.class_names.index(cname)
                txt = "Yes" if pred[i] == 1 else "No"
            else:
                txt = "-"
            row.append(txt)
        row.append(safety)
        self.tree.insert('', tk.END, values=row)
    def run_infer_on_file(self, img_path: str):
        if not self.engine.ready:
            messagebox.showerror("Model not loaded", "Load a trained model first.")
            return
        thr = float(self.var_thresh.get())
        probs, pred = self.engine.predict_image(img_path, threshold=thr)
        safety = self.engine.safety_status(pred)
        self.show_image_on_canvas(img_path)
        self.insert_result_row(img_path, probs, pred, safety)
    def on_open_image(self):
        path = filedialog.askopenfilename(title="Open image", filetypes=[("Image files", ".jpg .jpeg .png .bmp .tif .tiff .webp"), ("All files", "*.*")])
        if path:
            self.run_infer_on_file(path)
    def on_open_folder(self):
        path = filedialog.askdirectory(title="Select folder with images")
        if not path:
            return
        paths = []
        for dirpath, _, filenames in os.walk(path):
            for fn in filenames:
                if os.path.splitext(fn)[1].lower() in IMG_EXTS:
                    paths.append(os.path.join(dirpath, fn))
        if not paths:
            messagebox.showwarning("No images", "No supported images found in the selected folder.")
            return
        self.clear_tree()
        for p in sorted(paths):
            try:
                self.run_infer_on_file(p)
                self.root.update_idletasks()
            except Exception as e:
                self.log(f"Error on {p}: {e}")
    def on_export_csv(self):
        if not self.tree.get_children():
            messagebox.showinfo("No results", "There are no inference results to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], title="Save results as CSV")
        if not path:
            return
        cols = ["image"] + CANONICAL_CLASSES + ["safety"]
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(cols)
            for iid in self.tree.get_children():
                row = self.tree.item(iid)['values']
                writer.writerow(row)
        messagebox.showinfo("Exported", f"Results exported to: {path}")

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
