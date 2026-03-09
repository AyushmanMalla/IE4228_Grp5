@echo off
:: run_demo.bat - Windows automated setup and runner

echo === Face Recognition System Setup ^& Run ===

echo [1/4] Checking Python environment...
if not exist ".venv\" (
    echo Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

echo Installing requirements...
pip install -r requirements.txt --quiet
echo Environment ready.

echo [2/4] Checking for dataset...
if not exist "data\lfw\" (
    echo Downloading LFW dataset (this might take a minute)...
    python scripts\download_lfw.py --data-dir data\lfw
) else (
    echo Dataset already exists. Skipping download.
)

echo [3/4] Building facial embedding gallery...
if not exist "data\gallery\gallery.json" (
    echo Building gallery with top 5 identities...
    python scripts\build_gallery.py --images-dir data\lfw --output-dir data\gallery --top-n 5
) else (
    echo Gallery already exists. Skipping build.
)

echo [4/4] Starting live Graphical Interface...
python -m facerec.gui --gallery-dir data\gallery
