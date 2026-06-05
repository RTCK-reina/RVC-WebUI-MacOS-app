#!/usr/bin/env bash
# Build the standalone RVC-WebUI.app bundle.
#
# Requirements:
#   brew install xcodegen
#   conda install -n base -c conda-forge conda-pack
#   (plus a working conda / miniforge installation with an `rvc` env)
#
# Usage:
#   ./build_app.sh [--skip-conda] [--skip-xcode] [--skip-sign]
#
# Produces: build/RVC-WebUI.app
# (xcodebuild itself emits "RVC Swift.app" driven by PRODUCT_NAME in
#  RVCApp/project.yml; this script renames on copy so the distributed
#  bundle / Release zip / documentation all use RVC-WebUI.app.)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
# xcodebuild output is "RVC Swift.app" (driven by PRODUCT_NAME in RVCApp/project.yml),
# but the distribution bundle users see — Release zip, /Applications entry,
# documentation examples — is "RVC-WebUI.app". We rename on copy below.
XCODE_OUTPUT_NAME="RVC Swift.app"
APP_NAME="RVC-WebUI.app"
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
fi

# Rename xcodebuild output (SYMROOT -> build/Release/"${XCODE_OUTPUT_NAME}")
# to the distribution bundle name. Done outside the SKIP_XCODE gate so that
# --skip-xcode runs can still refresh ${APP_PATH} from an earlier xcodebuild.
XCODE_APP_PATH="${BUILD_DIR}/Release/${XCODE_OUTPUT_NAME}"
if [[ ! -d "${XCODE_APP_PATH}" ]]; then
    echo "==> ERROR: Xcode output bundle not found: ${XCODE_APP_PATH}" >&2
    echo "==> Run without --skip-xcode to regenerate it." >&2
    exit 1
fi
rm -rf "${APP_PATH}"
cp -R "${XCODE_APP_PATH}" "${APP_PATH}"

# ---------------------------------------------------------------------------
# Step 3: Copy Python backend + assets into the bundle.
# ---------------------------------------------------------------------------
# Pre-flight: required model weights must already be downloaded and match the
# expected SHA-256 tracked in sha256.env. Verifying the hash (not just file
# size) catches HTML error pages and partial downloads that happen to exceed
# a size floor — both produce silently broken .app bundles otherwise.
SHA_FILE="${ROOT_DIR}/sha256.env"
if [[ ! -f "${SHA_FILE}" ]]; then
    echo "==> ERROR: checksum file not found: ${SHA_FILE}" >&2
    exit 1
fi

if command -v shasum >/dev/null 2>&1; then
    _sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
elif command -v sha256sum >/dev/null 2>&1; then
    _sha256() { sha256sum "$1" | awk '{print $1}'; }
else
    echo "==> ERROR: neither 'shasum' nor 'sha256sum' is available for asset verification." >&2
    exit 1
fi

# "<relative asset path>:<sha256.env key>" pairs. Keep the required set minimal
# (hubert + rmvpe are the two assets whose absence breaks single inference /
# realtime VC / UVR5); pretrained_v{1,2} and UVR5 weights are optional at
# build time and tolerated at runtime.
REQUIRED_ASSETS=(
    "assets/hubert/hubert_base.pt:sha256_hubert_base_pt"
    "assets/rmvpe/rmvpe.pt:sha256_rmvpe_pt"
)

asset_errors=0
for entry in "${REQUIRED_ASSETS[@]}"; do
    asset="${entry%%:*}"
    key="${entry##*:}"
    asset_path="${ROOT_DIR}/${asset}"

    if [[ ! -s "${asset_path}" ]]; then
        echo "==> ERROR: required asset missing or empty: ${asset}" >&2
        asset_errors=$((asset_errors + 1))
        continue
    fi

    expected=$(awk -F= -v k="${key}" '
        { gsub(/[[:space:]]/, "", $1) }
        $1 == k { gsub(/[[:space:]]/, "", $2); print $2; exit }
    ' "${SHA_FILE}")

    if [[ ! "${expected}" =~ ^[A-Fa-f0-9]{64}$ ]]; then
        echo "==> ERROR: ${key} not found (or invalid) in ${SHA_FILE}" >&2
        asset_errors=$((asset_errors + 1))
        continue
    fi

    actual=$(_sha256 "${asset_path}")
    if [[ "${actual}" != "${expected}" ]]; then
        echo "==> ERROR: SHA-256 mismatch for ${asset}: expected ${expected} got ${actual}" >&2
        asset_errors=$((asset_errors + 1))
    fi
done

if (( asset_errors > 0 )); then
    echo "==> Run ./tools/download_assets.sh --all first, then re-run this script." >&2
    exit 1
fi

echo "==> Populating Resources/"
RES_DIR="${APP_PATH}/Contents/Resources"
mkdir -p "${RES_DIR}/rvc_backend" "${RES_DIR}/python"

# Strip macOS App Management 'com.apple.provenance' xattr from source trees
# before rsync. Without this, macOS Sequoia (14+) blocks rsync mkstempat with
# EPERM when it tries to write into the .app bundle, failing Step 3 with
# "Operation not permitted" on conda-pack output files.
#
# This MUST cover every tree we rsync/cp into the bundle below — not just
# python_env + assets. A checkout that picked up provenance xattrs (e.g. files
# that were quarantined/downloaded) would otherwise hit the same EPERM on the
# configs/infer/rvc/i18n/tools rsync and the rpc_*.py copies, leaving a
# partially populated .app.
xattr -cr "${PYTHON_BUNDLE}" 2>/dev/null || true
for d in assets configs infer rvc i18n tools; do
    xattr -cr "${ROOT_DIR}/${d}" 2>/dev/null || true
done
xattr -c "${ROOT_DIR}/rpc_server.py" "${ROOT_DIR}/rpc_training.py" 2>/dev/null || true

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
