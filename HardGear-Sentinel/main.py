#!/usr/bin/env python3
"""
HardGear Sentinel - Offline PPE detection (rule-based) with improved visuals.
"""

import os
import cv2
import numpy as np
import pandas as pd
import argparse
import time
from datetime import datetime
import matplotlib.pyplot as plt

# -------------------------
# Configuration
# -------------------------
CONF_MIN_PERSON_AREA = 4000
RESIZE_WIDTH = 800
VERBOSE = False

# Colors
COLOR_SAFE = (0, 255, 0)
COLOR_WARN = (0, 165, 255)
COLOR_ALERT = (0, 0, 255)
COLOR_TEXT_BG = (30, 30, 30)

# PPE color filters
HELMET_RANGES = [
    ((0, 70, 120), (30, 255, 255)),
    ((5, 70, 120), (20, 255, 255)),
    ((0, 0, 180), (180, 45, 255)),
]
VEST_RANGES = [
    ((5, 120, 120), (15, 255, 255)),
    ((35, 80, 80), (90, 255, 255)),
]
GLOVE_RANGES = [
    ((0, 0, 150), (180, 60, 255)),
]

BOOT_VALUE_THRESH = 70

# -------------------------
# Helper functions
# -------------------------
def resize_keep_aspect(img, width=RESIZE_WIDTH):
    h, w = img.shape[:2]
    if w <= width:
        return img, 1.0
    scale = width / float(w)
    newh = int(h * scale)
    return cv2.resize(img, (width, newh)), scale

def color_mask_from_ranges(hsv, ranges):
    mask = None
    for lower, upper in ranges:
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        m = cv2.inRange(hsv, lower, upper)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask if mask is not None else np.zeros(hsv.shape[:2], dtype=np.uint8)

def draw_text_bg(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    (w, h), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (x, y - h - 6), (x + w + 4, y + 2), COLOR_TEXT_BG, -1)
    cv2.putText(img, text, (x + 2, y - 4), font, scale, color, thick, cv2.LINE_AA)

# -------------------------
# Person Detector
# -------------------------
class PersonDetector:
    def __init__(self, models_dir='models'):
        prototxt = os.path.join(models_dir, "MobileNetSSD_deploy.prototxt")
        caffemodel = os.path.join(models_dir, "MobileNetSSD_deploy.caffemodel")
        if os.path.exists(prototxt) and os.path.exists(caffemodel):
            if VERBOSE: print("Using MobileNetSSD DNN")
            self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
            self.use_dnn = True
        else:
            if VERBOSE: print("Using HOG detector")
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.use_dnn = False

    def detect(self, image):
        if self.use_dnn:
            blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections = self.net.forward()
            h, w = image.shape[:2]
            boxes = []
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                cls = int(detections[0, 0, i, 1])
                if conf > 0.4 and cls == 15:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    boxes.append((x1, y1, x2, y2, conf))
            return boxes
        else:
            rects, weights = self.hog.detectMultiScale(image, winStride=(8,8), padding=(8,8), scale=1.05)
            boxes = [(x, y, x+w, y+h, float(wt)) for (x,y,w,h), wt in zip(rects, weights)]
            return boxes

# -------------------------
# PPE detection logic
# -------------------------
def detect_helmet(r):
    if r.size == 0: return False, 0.0
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    mask = color_mask_from_ranges(hsv, HELMET_RANGES)
    pct = 100.0 * cv2.countNonZero(mask) / (mask.size + 1)
    return pct > 2.0, pct

def detect_vest(r):
    if r.size == 0: return False, 0.0
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    mask = color_mask_from_ranges(hsv, VEST_RANGES)
    pct = 100.0 * cv2.countNonZero(mask) / (mask.size + 1)
    return pct > 3.0, pct

def detect_gloves(r):
    if r.size == 0: return False, 0.0
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    mask = color_mask_from_ranges(hsv, GLOVE_RANGES)
    pct = 100.0 * cv2.countNonZero(mask) / (mask.size + 1)
    return pct > 2.0, pct

def detect_boots(r):
    if r.size == 0: return False, 0.0
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    dark = cv2.threshold(gray, BOOT_VALUE_THRESH, 255, cv2.THRESH_BINARY_INV)[1]
    pct = 100.0 * cv2.countNonZero(dark) / (dark.size + 1)
    return pct > 6.0, pct

# -------------------------
# Region crop
# -------------------------
def crop_regions(img, box):
    x1, y1, x2, y2 = box
    h = y2 - y1
    w = x2 - x1
    head = img[y1:y1+int(0.25*h), x1:x2]
    torso = img[y1+int(0.25*h):y1+int(0.65*h), x1:x2]
    lh = img[y1+int(0.45*h):y1+int(0.63*h), x1:x1+int(0.3*w)]
    rh = img[y1+int(0.45*h):y1+int(0.63*h), x2-int(0.3*w):x2]
    feet = img[y2-int(0.15*h):y2, x1:x2]
    return {"head": head, "torso": torso, "left_hand": lh, "right_hand": rh, "feet": feet}

# -------------------------
# Single image analysis
# -------------------------
def analyze_image(path, detector, output_dir, fps_info=None):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("Cannot read " + path)
    img_resized, scale = resize_keep_aspect(img)
    boxes = detector.detect(img_resized)
    annotated = img_resized.copy()
    results = []

    # Banner
    cv2.rectangle(annotated, (0,0), (annotated.shape[1],32), (25,25,25), -1)
    if fps_info is not None:
        draw_text_bg(annotated, f"FPS: {fps_info}", annotated.shape[1]-120, 28, (255,255,255))

    for (x1, y1, x2, y2, score) in boxes:
        if (x2-x1)*(y2-y1) < CONF_MIN_PERSON_AREA: continue

        regs = crop_regions(img_resized, (x1, y1, x2, y2))
        helmet_ok, h_pct = detect_helmet(regs["head"])
        vest_ok, v_pct = detect_vest(regs["torso"])
        gl_l, glp_l = detect_gloves(regs["left_hand"])
        gl_r, glp_r = detect_gloves(regs["right_hand"])
        boot_ok, b_pct = detect_boots(regs["feet"])

        glove_ok = gl_l or gl_r
        glove_pct = (glp_l + glp_r)/2.0

        if helmet_ok and vest_ok:
            status = "SAFE"
            color = COLOR_SAFE
        elif helmet_ok and not vest_ok:
            status = "WARNING (No Vest)"
            color = COLOR_WARN
        else:
            status = "UNSAFE"
            color = COLOR_ALERT

        label = f"{status} | H:{h_pct:.1f}% V:{v_pct:.1f}% G:{glove_pct:.1f}% B:{b_pct:.1f}%"
        cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
        draw_text_bg(annotated, label, x1, y1, color)

        results.append({"status": status})

    out_path = os.path.join(output_dir, "annotated_" + os.path.basename(path))
    cv2.imwrite(out_path, annotated)

    summary = {
        "image": os.path.basename(path),
        "total": len(results),
        "safe": sum(1 for r in results if r["status"]=="SAFE"),
        "unsafe": sum(1 for r in results if r["status"]!="SAFE"),
    }

    return results, summary, out_path

# -------------------------
# Generate matplotlib visuals
# -------------------------
def generate_visuals(df, output_dir):
    # Bar chart: safe vs unsafe per image
    plt.figure(figsize=(10,5))
    plt.bar(df['image'], df['safe'], color='green', label='Safe')
    plt.bar(df['image'], df['unsafe'], bottom=df['safe'], color='red', label='Unsafe')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Number of Workers")
    plt.title("Safety Status per Image")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "safety_bar_chart.png"))
    plt.close()

    # Pie chart: overall safe vs unsafe
    total_safe = df['safe'].sum()
    total_unsafe = df['unsafe'].sum()
    plt.figure(figsize=(6,6))
    plt.pie([total_safe, total_unsafe], labels=["Safe","Unsafe"], colors=["green","red"], autopct='%1.1f%%', startangle=90)
    plt.title("Overall Safety Distribution")
    plt.savefig(os.path.join(output_dir, "safety_pie_chart.png"))
    plt.close()

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input","-i", required=True)
    parser.add_argument("--output","-o", required=True)
    parser.add_argument("--models","-m", default="models")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    detector = PersonDetector(models_dir=args.models)
    rows = []
    prev = time.time()

    for fname in sorted(os.listdir(args.input)):
        if not fname.lower().endswith((".jpg",".jpeg",".png",".bmp")): continue

        now = time.time()
        fps = int(1/(now-prev)) if (now-prev)>0 else 0
        prev = now

        path = os.path.join(args.input,fname)
        try:
            results, summary, out_img = analyze_image(path, detector, args.output, fps_info=fps)
        except Exception as e:
            print("Error:", fname, e)
            continue

        rows.append({
            "image": summary["image"],
            "total": summary["total"],
            "safe": summary["safe"],
            "unsafe": summary["unsafe"],
            "score": int(0 if summary["total"]==0 else 100*summary["safe"]//summary["total"])
        })

        print(f"[{summary['image']}] Safe: {summary['safe']} Unsafe: {summary['unsafe']} -> saved {out_img}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.output,"safety_report.csv"), index=False)
    print("CSV report saved.")

    # Generate visuals
    generate_visuals(df, args.output)
    print("Matplotlib visuals saved: bar chart & pie chart.")

if __name__=="__main__":
    main()
