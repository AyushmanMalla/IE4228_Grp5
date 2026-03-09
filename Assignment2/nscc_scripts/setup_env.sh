#!/bin/bash
# ===========================================================================
#  NSCC Aspire2a — Environment Setup for IE4228 Face Recognition
#  
#  Run this ONCE from a login node to create the Python venv.
#  Usage:  bash nscc_scripts/setup_env.sh
# ===========================================================================

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-<YourProjectID>}"
SCRATCH="/scratch/$USER"
ENV_DIR="$SCRATCH/facerec-env"

echo "=== NSCC Environment Setup ==="
echo "User:       $USER"
echo "Env path:   $ENV_DIR"

# --- Load system modules ---
module purge
module load python/3.10.4

# --- Create venv ---
if [ -d "$ENV_DIR" ]; then
    echo "Venv already exists at $ENV_DIR — activating..."
else
    echo "Creating venv at $ENV_DIR..."
    python3 -m venv "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"

# --- Install dependencies ---
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r "$(dirname "$0")/../requirements.txt"

# Install onnxruntime-gpu for A100 CUDA support (replaces CPU-only version)
pip install onnxruntime-gpu>=1.17.0

echo ""
echo "=== Setup complete ==="
echo "Activate with:  source $ENV_DIR/bin/activate"
