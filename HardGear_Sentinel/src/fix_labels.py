import os
from pathlib import Path

# 🔒 YOUR FINAL CLASS STANDARD
FINAL_CLASSES = {
    "person": 0,
    "helmet": 1,
    "goggles": 2,
    "mask": 3,
    "gloves": 4,
    "vest": 5,
    "boots": 6
}

# 📝 CHANGE THIS for each dataset you process
ROBOFLOW_CLASSES = {
    0: "boots",
    1: "gloves",
    2: "mask",
    3: "helmet",
    4: "vest",
    5: "goggles"
}


INPUT_LABELS_DIR = "../roboflow_dataset/labels"
OUTPUT_LABELS_DIR = "../dataset/train/labels"

os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

def fix_labels():
    for label_file in Path(INPUT_LABELS_DIR).glob("*.txt"):
        new_lines = []

        with open(label_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                old_class_id = int(parts[0])

                if old_class_id not in ROBOFLOW_CLASSES:
                    continue

                class_name = ROBOFLOW_CLASSES[old_class_id]

                if class_name not in FINAL_CLASSES:
                    continue

                new_class_id = FINAL_CLASSES[class_name]
                parts[0] = str(new_class_id)
                new_lines.append(" ".join(parts))

        if new_lines:
            with open(Path(OUTPUT_LABELS_DIR) / label_file.name, "w") as f:
                f.write("\n".join(new_lines))

    print("✅ Labels converted successfully!")

if __name__ == "__main__":
    fix_labels()
