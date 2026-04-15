#!/usr/bin/env bash
# Create the `rvc` conda environment used by build_app.sh.
#
# Usage:
#   ./setup_conda_env.sh
#
# This assumes conda / miniforge is already installed.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${CONDA_ENV_NAME:-rvc}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

echo "==> Creating conda env '${ENV_NAME}' (python ${PYTHON_VERSION})"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' already exists, reusing."
else
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing PyTorch (Apple Silicon / CPU)"
pip install torch torchvision torchaudio

echo "==> Installing RVC deps (requirements/app.txt)"
pip install -r "${ROOT_DIR}/requirements/app.txt"

echo "==> Installing conda-pack (for the .app bundle build)"
pip install conda-pack

echo "==> Done. Activate with: conda activate ${ENV_NAME}"
