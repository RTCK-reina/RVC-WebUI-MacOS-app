#!/bin/bash

# RVC WebUI Launch Script for macOS
set -e

echo "Starting RVC WebUI for macOS..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 could not be found. Please install Python 3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Downgrade pip to avoid metadata errors with older packages (omegaconf)
pip install "pip<24.1" setuptools wheel
pip install Cython numpy

# Install fairseq from source to avoid version.txt error
# We use a specific commit or tag if needed, but main or v0.12.2 usually works better for source builds than PyPI tarball on M1
pip install git+https://github.com/facebookresearch/fairseq.git@v0.12.2

# Install other dependencies
# fail-fast if installation fails
pip install -r requirements/macos.txt

# Set environment variables for macOS
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Check for model files
if [ ! -f "assets/hubert/hubert_base.pt" ]; then
    echo "WARNING: hubert_base.pt not found in assets/hubert/"
    echo "Attempting to download..."
    python3 tools/download_models.py || echo "Auto-download failed. Please download https://huggingface.co/rvc-boss/uvr5_weights/resolve/main/hubert_base.pt manually to assets/hubert/"
fi

if [ ! -f "assets/rmvpe/rmvpe.pt" ]; then
    echo "WARNING: rmvpe.pt not found in assets/rmvpe/"
     echo "Attempting to download..."
     # Assuming tools/download_models.py handles this, otherwise manual instruction:
     echo "Please download https://huggingface.co/rvc-boss/uvr5_weights/resolve/main/rmvpe.pt manually to assets/rmvpe/"
fi

# Launch the application
echo "Launching RVC WebUI..."
# Use python3 explicitely
python3 web.py --port 7865 --noautoopen

echo "RVC WebUI has stopped."
