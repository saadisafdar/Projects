#!/usr/bin/env python3
"""
HardGear Sentinel - Offline PPE detection (rule-based).
Default person detector: HOG (no external model).
Optional: MobileNet SSD if model files are provided in models/.
"""

import os
import cv2
import numpy as np
import pandas as pd
import argparse
from datetime import datetime

# -------------------------
# Configuration (tweakable)
# -------------------------
CONF_MIN_PERSON_AREA = 4000   # ignore tiny detections
RESIZE_WIDTH = 800           # speed/resolution tradeoff
VERBOSE = False

# HSV ranges for colors (tweak for your dataset lighting)
# helmet: white, yellow, orange
HELMET_RANGES = [
    ((0, 70, 120), (30, 255, 255)),    # yellow-ish
    ((5, 70, 120), (20, 255, 255)),    # orange-ish
    ((0, 0, 180), (180, 45, 255)),     # white-ish (low saturation, high value)
]
# vest: neon orange and neon green (typical safety vests)
VEST_RANGES = [
    ((5, 120, 120), (15, 255, 255)),   # orange neon
    ((35, 80, 80), (90, 255, 255)),    # green neon
]

# gloves: light gray/white near hands (optional)
GLOVE_RANGES = [
    ((0, 0, 150), (180, 60, 255)),     # light colors
]

# boots: dark region at feet (simple darkness test)
BOOT_VALUE_THRESH = 70   # lower value -> dark

# -------------------------
# Utilities
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

# -------------------------
# Person detection
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
            if VERBOSE: print("Using HOG person detector (built-in)")
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
                # class 15 is 'person' in MobileNetSSD
                if conf > 0.4 and cls == 15:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    boxes.append((x1, y1, x2, y2, conf))
            return boxes
        else:
            rects, weights = self.hog.detectMultiScale(image, winStride=(8,8), padding=(8,8), scale=1.05)
            boxes = []
            for (x, y, w, h), wt in zip(rects, weights):
                boxes.append((x, y, x+w, y+h, float(wt)))
            return boxes

# -------------------------
# PPE detection heuristics
# -------------------------
def detect_helmet(head_region):
    if head_region.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
    mask = color_mask_from_ranges(hsv, HELMET_RANGES)
    count = int(cv2.countNonZero(mask))
    total = mask.size
    pct = 100.0 * count / (total + 1)
    return (pct > 2.0), pct  # threshold 2% of head area

def detect_vest(torso_region):
    if torso_region.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(torso_region, cv2.COLOR_BGR2HSV)
    mask = color_mask_from_ranges(hsv, VEST_RANGES)
    count = int(cv2.countNonZero(mask))
    total = mask.size
    pct = 100.0 * count / (total + 1)
    return (pct > 3.0), pct  # threshold 3%

def detect_gloves(hand_region):
    if hand_region.size == 0:
        return False, 0.0
    hsv = cv2.cvtColor(hand_region, cv2.COLOR_BGR2HSV)
    mask = color_mask_from_ranges(hsv, GLOVE_RANGES)
    count = int(cv2.countNonZero(mask))
    total = mask.size
    pct = 100.0 * count / (total + 1)
    return (pct > 2.0), pct

def detect_boots(feet_region):
    if feet_region.size == 0:
        return False, 0.0
    gray = cv2.cvtColor(feet_region, cv2.COLOR_BGR2GRAY)
    # count dark pixels
    dark = cv2.threshold(gray, BOOT_VALUE_THRESH, 255, cv2.THRESH_BINARY_INV)[1]
    count = int(cv2.countNonZero(dark))
    total = dark.size
    pct = 100.0 * count / (total + 1)
    return (pct > 6.0), pct

# -------------------------
# Crop helpers
# -------------------------
def crop_regions(img, box):
    x1, y1, x2, y2 = box
    h = y2 - y1
    w = x2 - x1
    # head: top 25% of box
    head_h = max(5, int(0.25 * h))
    head = img[y1:y1+head_h, x1:x2]
    # torso: middle 40% (below head)
    torso_y1 = y1 + head_h
    torso_h = max(5, int(0.4 * h))
    torso = img[torso_y1:torso_y1+torso_h, x1:x2]
    # hands: left/right quarter height around mid-body
    hand_h = max(5, int(0.18 * h))
    hands_y = y1 + int(0.45 * h)
    left_hand = img[hands_y:hands_y+hand_h, x1:x1+int(0.3*w)]
    right_hand = img[hands_y:hands_y+hand_h, x2-int(0.3*w):x2]
    # feet: bottom 15%
    feet_h = max(5, int(0.15 * h))
    feet = img[y2-feet_h:y2, x1:x2]
    return {'head': head, 'torso': torso, 'left_hand': left_hand, 'right_hand': right_hand, 'feet': feet}

# -------------------------
# Main processing per image
# -------------------------
def analyze_image(path, detector, output_dir):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError("Failed to read " + path)
    img_resized, scale = resize_keep_aspect(img)
    boxes = detector.detect(img_resized)
    results = []
    annotated = img_resized.copy()
    for (x1, y1, x2, y2, score) in boxes:
        # filter tiny
        area = (x2-x1) * (y2-y1)
        if area < CONF_MIN_PERSON_AREA:
            continue
        regs = crop_regions(img_resized, (x1, y1, x2, y2))
        helmet_ok, helmet_pct = detect_helmet(regs['head'])
        vest_ok, vest_pct = detect_vest(regs['torso'])
        glove_ok_l, glove_pct_l = detect_gloves(regs['left_hand'])
        glove_ok_r, glove_pct_r = detect_gloves(regs['right_hand'])
        glove_ok = glove_ok_l or glove_ok_r
        boot_ok, boot_pct = detect_boots(regs['feet'])
        # safety logic
        # simple: helmet + vest -> SAFE; else UNSAFE
        if helmet_ok and vest_ok:
            status = "SAFE"
            color = (0, 255, 0)
        elif helmet_ok and not vest_ok:
            status = "WARNING (No Vest)"
            color = (0, 165, 255)
        else:
            status = "UNSAFE"
            color = (0, 0, 255)
        label = f"{status} | H:{helmet_pct:.1f}% V:{vest_pct:.1f}% G:{(glove_pct_l+glove_pct_r)/2:.1f}% B:{boot_pct:.1f}%"
        # annotate
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, max(y1-8,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        results.append({
            'box': (int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)),
            'helmet_ok': bool(helmet_ok),
            'helmet_pct': float(helmet_pct),
            'vest_ok': bool(vest_ok),
            'vest_pct': float(vest_pct),
            'glove_ok': bool(glove_ok),
            'boot_ok': bool(boot_ok),
            'status': status
        })
    # save annotated
    base = os.path.basename(path)
    out_img = os.path.join(output_dir, "annotated_" + base)
    cv2.imwrite(out_img, annotated)
    summary = {
        'image': base,
        'total': len(results),
        'safe': sum(1 for r in results if r['status'] == "SAFE"),
        'unsafe': sum(1 for r in results if r['status'] != "SAFE"),
        'timestamp': datetime.utcnow().isoformat()
    }
    return results, summary, out_img

# -------------------------
# CLI and batch processing
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="HardGear Sentinel - PPE detection")
    parser.add_argument('--input', '-i', required=True, help='Input folder with images')
    parser.add_argument('--output', '-o', required=True, help='Output folder for annotated images and CSV')
    parser.add_argument('--models', '-m', default='models', help='Models dir (for MobileNet SSD, optional)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    detector = PersonDetector(models_dir=args.models)

    rows = []
    for fname in sorted(os.listdir(args.input)):
        if not fname.lower().endswith(('.jpg','.jpeg','.png','.bmp')):
            continue
        path = os.path.join(args.input, fname)
        try:
            results, summary, out_img = analyze_image(path, detector, args.output)
        except Exception as e:
            print("Error processing", fname, ":", str(e))
            continue
        rows.append({
            'image': summary['image'],
            'total': summary['total'],
            'safe': summary['safe'],
            'unsafe': summary['unsafe'],
            'score': int(0 if summary['total']==0 else (100*summary['safe']//summary['total']))
        })
        print(f"[{summary['image']}] Total: {summary['total']} Safe: {summary['safe']} Unsafe: {summary['unsafe']} -> saved {out_img}")

    # save CSV
    df = pd.DataFrame(rows, columns=['image','total','safe','unsafe','score'])
    csv_path = os.path.join(args.output, 'safety_report.csv')
    df.to_csv(csv_path, index=False)
    print("CSV report saved:", csv_path)

if __name__ == "__main__":
    main()
