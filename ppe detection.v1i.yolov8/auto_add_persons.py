
#!/usr/bin/env python3
from ultralytics import YOLO
import os, glob
from PIL import Image

model = YOLO("yolov8n.pt")

def process(folder):
    img_dir = os.path.join(folder, "images")
    lbl_dir = os.path.join(folder, "labels")
    os.makedirs(lbl_dir, exist_ok=True)

    for img_path in glob.glob(os.path.join(img_dir, "*.*")):
        name = os.path.splitext(os.path.basename(img_path))[0]
        lbl_path = os.path.join(lbl_dir, name + ".txt")

        has_person = False
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    if line.strip() and int(line.split()[0]) == 0:
                        has_person = True
                        break

        if has_person:
            continue

        res = model.predict(
            source=img_path,
            conf=0.5,
            classes=0,
            device="cpu",
            verbose=False
        )

        boxes = []
        if res and len(res) > 0:
            r = res[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                img = Image.open(img_path)
                w, h = img.size

                for b in xyxy:
                    x1, y1, x2, y2 = map(float, b)
                    xc = ((x1 + x2) / 2) / w
                    yc = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    boxes.append((0, xc, yc, bw, bh))

        if boxes:
            with open(lbl_path, "a") as fh:
                for b in boxes:
                    fh.write(
                        f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n"
                    )

if __name__ == "__main__":
    process("train")
    process("valid")
    print("Auto-add persons finished.")

