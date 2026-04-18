#!/usr/bin/env bash
# Build the standalone RVC Swift.app bundle.
#
# Requirements:
#   brew install xcodegen
#   conda install -n base -c conda-forge conda-pack
#   (plus a working conda / miniforge installation with an `rvc` env)
#
# Usage:
#   ./build_app.sh [--skip-conda] [--skip-xcode] [--skip-sign]
#
# Produces: build/RVC Swift.app

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
APP_NAME="RVC Swift.app"
APP_PATH="${BUILD_DIR}/${APP_NAME}"

SKIP_CONDA=0
SKIP_XCODE=0
SKIP_SIGN=0
CONDA_ENV_NAME="${CONDA_ENV_NAME:-rvc}"

for arg in "$@"; do
    case "$arg" in
        --skip-conda) SKIP_CONDA=1 ;;
        --skip-xcode) SKIP_XCODE=1 ;;
        --skip-sign)  SKIP_SIGN=1 ;;
        *) echo "Unknown arg: $arg" >&2; exit 1 ;;
    esac
done

echo "==> Build root: ${ROOT_DIR}"
mkdir -p "${BUILD_DIR}"

# ---------------------------------------------------------------------------
# Step 1: Pack the Python environment with conda-pack.
# ---------------------------------------------------------------------------
PYTHON_BUNDLE="${BUILD_DIR}/python_env"

if [[ $SKIP_CONDA -eq 0 ]]; then
    echo "==> Packing conda environment '${CONDA_ENV_NAME}' with conda-pack"
    rm -rf "${PYTHON_BUNDLE}"
    mkdir -p "${PYTHON_BUNDLE}"
    conda-pack --ignore-editable-packages \
               --ignore-missing-files \
               -n "${CONDA_ENV_NAME}" \
               -o "${BUILD_DIR}/rvc_env.tar.gz" \
               --format tar.gz \
               --n-threads 4
    tar -xzf "${BUILD_DIR}/rvc_env.tar.gz" -C "${PYTHON_BUNDLE}"
    # Fix absolute shebangs / RPATHs for the new location.
    ( cd "${PYTHON_BUNDLE}" && ./bin/conda-unpack )
    rm "${BUILD_DIR}/rvc_env.tar.gz"
fi

# ---------------------------------------------------------------------------
# Step 2: Generate Xcode project and build the Swift app.
# ---------------------------------------------------------------------------
if [[ $SKIP_XCODE -eq 0 ]]; then
    echo "==> Generating Xcode project"
    ( cd "${ROOT_DIR}/RVCApp" && xcodegen )

    echo "==> Building Swift app"
    xcodebuild \
        -project "${ROOT_DIR}/RVCApp/RVCApp.xcodeproj" \
        -scheme RVCApp \
        -configuration Release \
        build

    # SYMROOT in project.yml outputs directly to build/Release/.
    rm -rf "${APP_PATH}"
    cp -R "${BUILD_DIR}/Release/RVC Swift.app" "${APP_PATH}"
fi

# ---------------------------------------------------------------------------
# Step 3: Copy Python backend + assets into the bundle.
# ---------------------------------------------------------------------------
# Pre-flight: required model weights must already be downloaded.
# Running the app without these produces "Model file not found" errors at
# inference / realtime VC time.
REQUIRED_ASSETS=(
    "assets/hubert/hubert_base.pt"
    "assets/rmvpe/rmvpe.pt"
)
MISSING_ASSETS=()
for asset in "${REQUIRED_ASSETS[@]}"; do
    asset_path="${ROOT_DIR}/${asset}"
    if [[ ! -s "${asset_path}" ]] || [[ $(stat -f%z "${asset_path}" 2>/dev/null || stat -c%s "${asset_path}") -lt 10000 ]]; then
        MISSING_ASSETS+=("${asset}")
    fi
done
if (( ${#MISSING_ASSETS[@]} > 0 )); then
    echo "==> ERROR: required model assets are missing or empty:" >&2
    for asset in "${MISSING_ASSETS[@]}"; do echo "      ${asset}" >&2; done
    echo "==> Run ./tools/download_assets.sh --all first, then re-run this script." >&2
    exit 1
fi

echo "==> Populating Resources/"
RES_DIR="${APP_PATH}/Contents/Resources"
mkdir -p "${RES_DIR}/rvc_backend" "${RES_DIR}/python"

# Python code (everything rpc_server.py imports).
for d in configs infer rvc i18n tools; do
    rsync -a --delete "${ROOT_DIR}/${d}" "${RES_DIR}/rvc_backend/"
done
cp "${ROOT_DIR}/rpc_server.py"   "${RES_DIR}/rvc_backend/"
cp "${ROOT_DIR}/rpc_training.py" "${RES_DIR}/rvc_backend/"
cp "${ROOT_DIR}/.env"          "${RES_DIR}/rvc_backend/" 2>/dev/null || true
cp "${ROOT_DIR}/sha256.env"    "${RES_DIR}/rvc_backend/" 2>/dev/null || true

# Assets (models). These are the network-bundled resources.
rsync -a --delete "${ROOT_DIR}/assets" "${RES_DIR}/rvc_backend/"

# Python runtime.
rsync -a --delete "${PYTHON_BUNDLE}/" "${RES_DIR}/python/"

# ---------------------------------------------------------------------------
# Step 4: Code sign every .so / .dylib, then the app itself.
# ---------------------------------------------------------------------------
if [[ $SKIP_SIGN -eq 0 ]]; then
    IDENTITY="${CODE_SIGN_IDENTITY:--}"  # '-' = ad-hoc signing by default
    echo "==> Code signing all dylibs / sos with identity: ${IDENTITY}"
    while IFS= read -r -d '' lib; do
        codesign --force --sign "${IDENTITY}" \
            --options runtime --timestamp=none "${lib}"
    done < <(find "${RES_DIR}/python" \( -name "*.so" -o -name "*.dylib" \) -print0)
    codesign --force --deep --sign "${IDENTITY}" \
        --entitlements "${ROOT_DIR}/RVCApp/RVCApp/RVCApp.entitlements" \
        --options runtime --timestamp=none "${APP_PATH}"
fi

echo "==> Bundle ready: ${APP_PATH}"
du -sh "${APP_PATH}" || true
