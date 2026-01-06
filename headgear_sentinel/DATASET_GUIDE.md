# HeadGear Sentinel - Dataset Preparation Guide

## 📊 Creating Your Training Dataset

This guide explains how to organize images for training.

---

## 🗂️ FOLDER STRUCTURE (MANDATORY)

Your dataset MUST be organized like this:

```
dataset/
├── class_0_person/
│   ├── person_001.jpg
│   ├── person_002.jpg
│   ├── person_003.jpg
│   └── ... (50-200 images)
│
├── class_1_helmet/
│   ├── helmet_001.jpg
│   ├── helmet_002.jpg
│   ├── helmet_003.jpg
│   └── ... (50-200 images)
│
├── class_2_goggles/
│   ├── goggles_001.jpg
│   ├── goggles_002.jpg
│   └── ... (50-200 images)
│
├── class_3_mask/
│   ├── mask_001.jpg
│   ├── mask_002.jpg
│   └── ... (50-200 images)
│
├── class_4_gloves/
│   ├── gloves_001.jpg
│   ├── gloves_002.jpg
│   └── ... (50-200 images)
│
├── class_5_vest/
│   ├── vest_001.jpg
│   ├── vest_002.jpg
│   └── ... (50-200 images)
│
└── class_6_boots/
    ├── boots_001.jpg
    ├── boots_002.jpg
    └── ... (50-200 images)
```

**IMPORTANT:**
- Folder names MUST be exactly as shown (class_X_name)
- Image filenames don't matter (can be anything)
- Image file types: .jpg, .jpeg, .png (case-insensitive)
- No labels files needed (system uses folder structure)

---

## 📸 IMAGE REQUIREMENTS

### File Format
- Format: JPEG or PNG
- Color space: Can be color (BGR) or grayscale
- File size: Any (will be resized to 64×64 during training)

### Image Size
```
Good:
  640×480 pixels
  800×600 pixels
  1024×768 pixels
  
Acceptable:
  200×200 pixels (small but workable)
  1920×1080 pixels (large, fine)
  
Bad:
  Smaller than 80×80 (too small, loses detail)
  Extremely large (>5000px, slow processing)
```

**Tip:** If dataset images are very large (>2000px), resize them:
```bash
# Linux command to resize all images in a folder
for file in dataset/class_1_helmet/*.jpg; do
  convert "$file" -resize 1024x768 "$file"
done
```

### Content Guidelines

Each image should be a clear photo of the specific object:

**Person (Class 0):**
- Full body or upper body of construction worker
- Clear face visible (for training diversity)
- Different angles, lighting conditions
- With or without PPE (doesn't matter)
- Examples:
  ```
  ✓ Worker standing facing camera
  ✓ Worker from side angle
  ✓ Worker in bright light
  ✓ Worker in shadow
  ✓ Worker indoors
  ✓ Worker outdoors
  ```

**Helmet (Class 1):**
- Construction/hard hats only (yellow, orange, white)
- Can be worn or standalone
- Different angles
- Close-ups or mid-range
- Examples:
  ```
  ✓ Helmet on person's head
  ✓ Helmet lying on ground
  ✓ Helmet from front view
  ✓ Helmet from side view
  ✓ Helmet from above
  ```

**Goggles (Class 2):**
- Safety goggles, eye protection
- Can be worn or held
- Different styles
- Various lighting
- Examples:
  ```
  ✓ Goggles on person's face
  ✓ Goggles on forehead
  ✓ Goggles in hand
  ✓ Goggles close-up
  ```

**Mask (Class 3):**
- Dust masks, respirators, N95 masks
- Worn on face
- Different types
- Various angles
- Examples:
  ```
  ✓ Mask covering nose and mouth
  ✓ Mask hanging around neck
  ✓ Mask in hand
  ✓ Different mask types
  ```

**Gloves (Class 4):**
- Work gloves, safety gloves
- On hands (preferred)
- Different colors/types
- Worn or displayed
- Examples:
  ```
  ✓ Gloves on worker's hands
  ✓ Gloves laying flat
  ✓ Pair of gloves
  ✓ Single glove
  ```

**Vest (Class 5):**
- Safety vests, reflective vests
- Hi-visibility clothing
- Worn on person
- Various colors (orange, yellow, white)
- Examples:
  ```
  ✓ Vest worn by worker
  ✓ Vest hanging
  ✓ Reflective vest close-up
  ✓ Different vest types
  ```

**Boots (Class 6):**
- Safety boots, work boots
- Steel-toed boots
- On feet or ground
- Various angles
- Examples:
  ```
  ✓ Boots on person's feet
  ✓ Pair of boots on ground
  ✓ Boot close-up
  ✓ Worker wearing boots (feet visible)
  ```

---

## 🎯 DATASET SIZE RECOMMENDATIONS

| Dataset Size | Training Time | Accuracy | Recommendation |
|--------------|---------------|----------|-----------------|
| 30 img/class | 2-3 minutes | ~70% | Minimum (not great) |
| 50 img/class | 3-5 minutes | ~80% | Acceptable |
| 100 img/class | 5-10 minutes | ~85% | Good |
| 150 img/class | 10-15 minutes | ~88% | Very Good |
| 200+ img/class | 15-30 minutes | ~90%+ | Excellent |

**Recommendation for semester project:** Start with 100-150 images per class.

---

## 📥 GETTING IMAGES (SOURCES)

### Option 1: Use Existing Datasets
```
Google Dataset Search
  → Search "PPE dataset" or "construction worker dataset"
  → Download relevant images
  → Organize into folders

Kaggle Datasets
  → https://www.kaggle.com/
  → Search for: "PPE detection", "safety equipment"
  → Download and extract

COCO Dataset
  → Contains people, various objects
  → Can extract relevant classes
```

### Option 2: Collect Your Own
```
Phone/Camera
  → Take photos of PPE items
  → Take photos of yourself or colleagues
  → Different lighting, angles

From Web
  → Google Images (image search)
  → Free stock photos (Unsplash, Pexels, Pixabay)
  → Construction/safety websites

Advantages:
  ✓ Best for your specific use case
  ✓ Realistic data
  ✓ You understand the data
```

### Option 3: Download Using Tools
```bash
# Using bing-image-downloader (install first)
pip install bing-image-downloader

# Then run Python script:
python3 -c "
from bing_image_downloader import downloader

# Download helmet images
downloader.download(
    'construction safety helmet',
    limit=100,
    output_dir='dataset',
    adult_filter_off=True,
    force_replace=False
)
"
```

---

## 🖼️ MANUAL ORGANIZATION STEPS

### Using File Manager

1. Create base folder:
   ```bash
   mkdir -p dataset
   cd dataset
   ```

2. Create class folders:
   ```bash
   mkdir class_0_person
   mkdir class_1_helmet
   mkdir class_2_goggles
   mkdir class_3_mask
   mkdir class_4_gloves
   mkdir class_5_vest
   mkdir class_6_boots
   ```

3. Download or copy images:
   - Place person images in `class_0_person/`
   - Place helmet images in `class_1_helmet/`
   - ... etc

4. Verify structure:
   ```bash
   ls -la dataset/
   ls -la dataset/class_1_helmet/
   ```

### Using Command Line

```bash
# Create structure in one command
mkdir -p dataset/{class_0_person,class_1_helmet,class_2_goggles,class_3_mask,class_4_gloves,class_5_vest,class_6_boots}

# Copy images (example)
cp /path/to/helmets/* dataset/class_1_helmet/
cp /path/to/people/* dataset/class_0_person/
# ... repeat for other classes

# Verify
find dataset -type f -name "*.jpg" | wc -l  # Count total images
find dataset -type d | xargs -I {} sh -c 'echo -n "{}: "; ls {} | wc -l'  # Count per folder
```

---

## ✅ QUALITY CHECKS

Before training, verify your dataset:

### Check 1: Folder Structure
```bash
ls -la dataset/

Expected output:
  class_0_person/
  class_1_helmet/
  class_2_goggles/
  class_3_mask/
  class_4_gloves/
  class_5_vest/
  class_6_boots/
```

### Check 2: Image Counts
```bash
for dir in dataset/class_*/; do
  count=$(ls "$dir"*.jpg "$dir"*.jpeg "$dir"*.png 2>/dev/null | wc -l)
  echo "$(basename $dir): $count images"
done

Expected output (minimum):
  class_0_person: 50+ images
  class_1_helmet: 50+ images
  class_2_goggles: 50+ images
  class_3_mask: 50+ images
  class_4_gloves: 50+ images
  class_5_vest: 50+ images
  class_6_boots: 50+ images
```

### Check 3: Image Validity
```bash
# Test that images can be read
python3 << 'EOF'
import cv2
from pathlib import Path

bad_images = []
for class_dir in Path('dataset').glob('class_*'):
    for img_file in class_dir.glob('*.jpg') + class_dir.glob('*.jpeg') + class_dir.glob('*.png'):
        img = cv2.imread(str(img_file))
        if img is None:
            bad_images.append(str(img_file))

if bad_images:
    print(f"Found {len(bad_images)} bad images:")
    for f in bad_images[:10]:  # Show first 10
        print(f"  - {f}")
else:
    print("All images are valid!")
EOF
```

### Check 4: Image Variety
```
For each class, verify images have:
  ✓ Different angles (front, side, top)
  ✓ Different lighting (bright, dark, natural, artificial)
  ✓ Different scales/distances
  ✓ Different contexts (indoors, outdoors, lab, real site)
  ✓ Different qualities (clear, slightly blurry)
  
Why? Better variety = better trained model = better generalization
```

---

## 🔄 DATA AUGMENTATION (OPTIONAL)

If you have fewer images, augment them:

```python
# augment.py - Run this to create variations
import cv2
import os
from pathlib import Path
import numpy as np

def augment_image(image_path, output_dir, num_variations=3):
    """Create variations of an image"""
    img = cv2.imread(image_path)
    base_name = Path(image_path).stem
    
    for i in range(num_variations):
        # Random rotation (-15 to +15 degrees)
        angle = np.random.uniform(-15, 15)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, rot_mat, (w, h))
        
        # Random brightness (-30 to +30)
        brightness = np.random.uniform(-30, 30)
        bright = cv2.convertScaleAbs(rotated, alpha=1, beta=brightness)
        
        # Save
        out_path = Path(output_dir) / f"{base_name}_aug{i}.jpg"
        cv2.imwrite(str(out_path), bright)

# Usage
for class_dir in Path('dataset').glob('class_*'):
    for img_file in class_dir.glob('*.jpg'):
        augment_image(str(img_file), str(class_dir), num_variations=2)
```

Run it:
```bash
python3 augment.py
```

**Result:** Each image produces 2 variations, effectively doubling your dataset.

---

## 🚀 QUICK START SCRIPT

Create `prepare_dataset.sh`:

```bash
#!/bin/bash

# Automatic dataset structure creation and validation

echo "Creating dataset structure..."
mkdir -p dataset/{class_0_person,class_1_helmet,class_2_goggles,class_3_mask,class_4_gloves,class_5_vest,class_6_boots}

echo "Dataset structure created!"
echo ""
echo "Next steps:"
echo "1. Download or copy images for each class"
echo "2. Place images in respective class_X_* folders"
echo "3. Run: python3 headgear_sentinel.py"
echo "4. In GUI: Click 'Load Dataset' → 'Train Model'"
echo ""
echo "Folder structure:"
ls -la dataset/
```

Run it:
```bash
chmod +x prepare_dataset.sh
./prepare_dataset.sh
```

---

## 📊 EXAMPLE DATASET STATISTICS

For a good project:

```
Total Images: 700-900
Images per class: 100-150

Breakdown:
  Class 0 (Person):  120 images
  Class 1 (Helmet):  100 images
  Class 2 (Goggles): 100 images
  Class 3 (Mask):    100 images
  Class 4 (Gloves):  100 images
  Class 5 (Vest):    120 images
  Class 6 (Boots):   140 images
  ─────────────────────
  Total:             780 images

Training time: ~10-15 minutes
Expected accuracy: 85-92% per class
Model file size: ~5-8 MB
```

---

## 🐛 TROUBLESHOOTING

### Issue: "No images found for class X"
**Solution:**
- Check folder name matches exactly (case-sensitive on Linux)
- Verify images are in correct class folder
- Check image extensions are .jpg, .jpeg, or .png

### Issue: "Image format not supported"
**Solution:**
- Only use JPEG or PNG files
- Convert other formats:
  ```bash
  convert image.bmp image.jpg  # ImageMagick
  ```

### Issue: Images are too large, loading is slow
**Solution:**
- Resize images to 800-1024px max width
- ```bash
  for f in dataset/class_*/*.jpg; do
    convert "$f" -resize 1024x "$f"
  done
  ```

### Issue: Not enough training images
**Solution:**
- Collect more images (go with 100+ per class)
- Or use data augmentation (see above)
- Or accept lower accuracy

---

**Ready to start? See README.md for running instructions.**
