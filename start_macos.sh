#!/bin/bash

# RVC WebUI Launch Script for macOS
echo "Starting RVC WebUI for macOS..."

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Set environment variables for macOS
export PYTORCH_ENABLE_MPS_FALLBACK=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# Set Python path
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Check if the required models exist
if [ ! -f "assets/hubert/hubert_base.pt" ]; then
    echo "ERROR: hubert_base.pt not found in assets/hubert/"
    echo "Please download the hubert_base.pt model file and place it in assets/hubert/"
    echo "You can download it from: https://huggingface.co/rvc-boss/uvr5_weights/resolve/main/hubert_base.pt"
    exit 1
fi

if [ ! -f "assets/rmvpe/rmvpe.pt" ]; then
    echo "ERROR: rmvpe.pt not found in assets/rmvpe/"
    echo "Please download the rmvpe.pt model file and place it in assets/rmvpe/"
    echo "You can download it from: https://huggingface.co/rvc-boss/uvr5_weights/resolve/main/rmvpe.pt"
    exit 1
fi

# Launch the application
echo "Launching RVC WebUI..."
python web.py --port 7865 --noautoopen

echo "RVC WebUI has stopped."
