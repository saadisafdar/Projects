#!/usr/bin/env python3
"""
Sample Dataset Generator for HeadGear Sentinel
Creates synthetic test dataset for quick testing and development
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random

CLASSES = ['Person', 'Helmet', 'Goggles', 'Mask', 'Gloves', 'Safety Vest', 'Boots']
DATASET_DIR = 'dataset'
IMAGES_PER_CLASS = 50  # Adjust for more/less images


def create_dummy_image(class_name, image_idx):
    """Create a dummy image for testing."""
    width, height = 400, 300
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    
    # Draw based on class
    if class_name == 'Person':
        # Draw simple person shape
        draw.ellipse([150, 40, 250, 120], fill=(200, 150, 100))  # Head
        draw.rectangle([140, 130, 260, 240], fill=(100, 100, 150))  # Body
        draw.rectangle([120, 130, 140, 240], fill=(200, 150, 100))  # Left arm
        draw.rectangle([260, 130, 280, 240], fill=(200, 150, 100))  # Right arm
        draw.rectangle([140, 240, 170, 300], fill=(50, 50, 50))  # Left leg
        draw.rectangle([230, 240, 260, 300], fill=(50, 50, 50))  # Right leg
    
    elif class_name == 'Helmet':
        # Draw helmet shape
        draw.ellipse([100, 50, 300, 200], outline=(0, 0, 0), width=5, fill=(200, 100, 50))
        draw.rectangle([80, 180, 320, 250], outline=(0, 0, 0), width=3)
    
    elif class_name == 'Goggles':
        # Draw goggles
        draw.ellipse([80, 120, 160, 180], outline=(0, 0, 0), width=3, fill=(200, 200, 200))
        draw.ellipse([240, 120, 320, 180], outline=(0, 0, 0), width=3, fill=(200, 200, 200))
        draw.line([(160, 150), (240, 150)], fill=(0, 0, 0), width=3)
    
    elif class_name == 'Mask':
        # Draw mask
        draw.rectangle([100, 150, 300, 250], fill=(50, 100, 255), outline=(0, 0, 0), width=3)
        draw.ellipse([120, 160, 160, 190], fill=(255, 255, 255))
        draw.ellipse([240, 160, 280, 190], fill=(255, 255, 255))
    
    elif class_name == 'Gloves':
        # Draw gloves
        draw.ellipse([60, 200, 140, 280], fill=(255, 100, 50), outline=(0, 0, 0), width=3)
        draw.ellipse([260, 200, 340, 280], fill=(255, 100, 50), outline=(0, 0, 0), width=3)
    
    elif class_name == 'Safety Vest':
        # Draw vest
        draw.polygon([(120, 100), (280, 100), (300, 280), (100, 280)], 
                    fill=(255, 200, 0), outline=(0, 0, 0))
        draw.rectangle([150, 120, 250, 260], fill=(255, 100, 100))
    
    elif class_name == 'Boots':
        # Draw boots
        draw.rectangle([100, 240, 160, 300], fill=(50, 50, 50), outline=(0, 0, 0), width=3)
        draw.rectangle([240, 240, 300, 300], fill=(50, 50, 50), outline=(0, 0, 0), width=3)
    
    # Add class name and index
    draw.text((10, 10), f"{class_name} #{image_idx}", fill=(0, 0, 0))
    
    # Add some random noise for variation
    pixels = image.load()
    for _ in range(100):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        r, g, b = pixels[x, y]
        noise = random.randint(-20, 20)
        pixels[x, y] = (
            max(0, min(255, r + noise)),
            max(0, min(255, g + noise)),
            max(0, min(255, b + noise))
        )
    
    return image


def generate_dataset():
    """Generate sample dataset."""
    print("🛡️  HeadGear Sentinel - Sample Dataset Generator")
    print("=" * 50)
    
    # Create dataset directory
    os.makedirs(DATASET_DIR, exist_ok=True)
    print(f"\n✓ Creating dataset structure in '{DATASET_DIR}/'")
    
    # Create class folders and images
    for class_name in CLASSES:
        class_dir = os.path.join(DATASET_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        print(f"\n  Creating {IMAGES_PER_CLASS} images for {class_name}...")
        
        for img_idx in range(1, IMAGES_PER_CLASS + 1):
            # Create image
            image = create_dummy_image(class_name, img_idx)
            
            # Save image
            filename = f"{class_name.lower().replace(' ', '_')}_{img_idx:03d}.jpg"
            filepath = os.path.join(class_dir, filename)
            image.save(filepath, quality=85)
            
            if img_idx % 10 == 0:
                print(f"    Created {img_idx}/{IMAGES_PER_CLASS} images")
        
        print(f"  ✓ {IMAGES_PER_CLASS} images created for {class_name}")
    
    print("\n" + "=" * 50)
    print("✅ Sample dataset created successfully!")
    print(f"\nDataset structure:")
    print(f"  Location: {DATASET_DIR}/")
    print(f"  Classes: {len(CLASSES)}")
    print(f"  Total images: {len(CLASSES) * IMAGES_PER_CLASS}")
    print(f"  Images per class: {IMAGES_PER_CLASS}")
    
    print("\n📖 Next steps:")
    print("  1. Run: python3 headgear_sentinel.py")
    print("  2. Click '📂 Load Dataset Folder' in GUI")
    print("  3. Select the 'dataset' folder")
    print("  4. Configure training parameters")
    print("  5. Click '🚀 Start Training'")
    print("\n💡 Tip: This sample dataset is for testing only.")
    print("   For better results, use real PPE images.")


if __name__ == '__main__':
    try:
        generate_dataset()
    except KeyboardInterrupt:
        print("\n\n⚠️  Dataset generation cancelled")
    except Exception as e:
        print(f"\n❌ Error: {e}")
