#!/bin/bash

# HeadGear Sentinel Setup Script
# For Linux Mint 22.2 / Ubuntu 24.04+

echo "=============================================="
echo "HeadGear Sentinel Setup"
echo "=============================================="

# Check Python version
echo "[1/5] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found. Installing..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-tk
else
    python_version=$(python3 --version | cut -d' ' -f2)
    echo "✓ Python $python_version found"
fi

# Install system dependencies
echo "[2/5] Installing system dependencies..."
sudo apt install -y python3-tk libopencv-dev python3-opencv

# Create virtual environment (optional but recommended)
echo "[3/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created"
else
    source venv/bin/activate
    echo "✓ Virtual environment already exists"
fi

# Install Python packages
echo "[4/5] Installing Python dependencies..."
pip install --upgrade pip
pip install opencv-python scikit-learn scikit-image joblib numpy Pillow

# Create necessary directories
echo "[5/5] Creating project directories..."
mkdir -p dataset models
mkdir -p dataset/class_0_person
mkdir -p dataset/class_1_helmet
mkdir -p dataset/class_2_goggles
mkdir -p dataset/class_3_mask
mkdir -p dataset/class_4_gloves
mkdir -p dataset/class_5_vest
mkdir -p dataset/class_6_boots

echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Add training images to dataset/ folders"
echo "2. Run: python3 headgear_sentinel.py"
echo "3. Click 'Load Dataset' then 'Train Model'"
echo ""
echo "For help, see README.md"
