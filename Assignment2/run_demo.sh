#!/usr/bin/env bash
# run_demo.sh - macOS/Linux automated setup and runner

set -e

echo "=== Face Recognition System Setup & Run ==="

echo "[1/4] Checking Python environment..."
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

echo "Installing requirements..."
pip install -r requirements.txt --quiet
echo "Environment ready."

echo "[2/4] Checking for dataset..."
if [ ! -d "data/lfw" ] || [ -z "$(ls -A data/lfw 2>/dev/null)" ]; then
    echo "Downloading LFW dataset (this might take a minute)..."
    python scripts/download_lfw.py --data-dir data/lfw
else
    echo "Dataset already exists. Skipping download."
fi

echo "[3/4] Building facial embedding gallery..."
if [ ! -d "data/gallery" ] || [ ! -f "data/gallery/gallery.json" ]; then
    echo "Building gallery with top 5 identities..."
    python scripts/build_gallery.py --images-dir data/lfw --output-dir data/gallery --top-n 5
else
    echo "Gallery already exists. Skipping build."
fi

echo "[4/4] Starting live Graphical Interface..."
python -m facerec.gui --gallery-dir data/gallery
