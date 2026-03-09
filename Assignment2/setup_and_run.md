# Face Recognition Pipeline: Setup and Run Guide

This guide walks you through setting up the environment, downloading the dataset, generating the embeddings gallery, and running the live graphical interface. It includes instructions that work across macOS, Linux, and Windows.
---
QUICK START GUIDE
---

## 1. Kaggle API Setup (Data Download)

The project uses the **LFW (Labeled Faces in the Wild)** dataset for local gallery generation and benchmarking. We use the Kaggle API to fetch this smoothly.

1. Log in or sign up at [kaggle.com](https://www.kaggle.com/).
2. Go to your **Settings** (Click on your profile picture -> Settings).
3. Scroll down to the **API** section and click **Create New Token**.
4. A file named `kaggle.json` will be downloaded to your machine.

You must place this `kaggle.json` file in the correct hidden folder for your operating system:
- **macOS / Linux:** `~/.kaggle/kaggle.json`
- **Windows:** `C:\Users\<YourUsername>\.kaggle\kaggle.json`

*(Note: Make sure to create the `.kaggle` folder if it doesn't already exist. The script also has a fallback down-loader, but the Kaggle source is the most reliable).*

### 2. Run the Entrypoint Script

Once the Kaggle API key is in place, simply run the entrypoint script for your operating system. Ensure you run this from within the `Assignment2` directory.

**For macOS / Linux:**
```bash
chmod +x run_demo.sh
./run_demo.sh
```

**For Windows (Command Prompt or PowerShell):**
```cmd
.\run_demo.bat
```

The script will handle everything and the real-time webcam dashboard will appear!

---

## 🛠 Advanced / Manual Setup

If you prefer to set up the environment and run the scripts manually, or if you want to run specific parts of the pipeline (like benchmarking), follow these steps.

### 1. Environment Setup

It is highly recommended to use a virtual environment, as the project relies on specific versions of `insightface`, `opencv-python`, and `onnxruntime`.

**Prerequisites:** Python 3.10+ (tested up to 3.13)

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Ingestion & Processing

Ensure your virtual environment is activated before running these scripts.

**Download the Dataset:**
```bash
python scripts/download_lfw.py --data-dir data/lfw
```
*This downloads and extracts the raw face images into the `data/lfw/` directory.*

**Build the Gallery:**
```bash
python scripts/build_gallery.py --images-dir data/lfw --output-dir data/gallery --top-n 5
```
*This calculates the 512D ArcFace embeddings for the top 5 most frequent identities in the dataset.*

### 3. Running the Application Manually

**The Live GUI Dashboard:**
```bash
python -m facerec.gui --gallery-dir data/gallery
```
**Options:**
- `--camera <index>`: Change the webcam index if you have multiple cameras (default is `0`).
- `--device <cpu|cuda>`: Hardware execution provider (default is `cpu`, which is highly optimized for this pipeline).

**Performance Benchmarking:**
To test raw FPS, inference time, and detection accuracy overhead without rendering the GUI:
```bash
python scripts/benchmark.py --data-dir data/lfw --gallery-dir data/gallery
```
