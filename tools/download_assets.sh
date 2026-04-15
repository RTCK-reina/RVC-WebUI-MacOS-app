#!/usr/bin/env bash
# Download RVC model assets from HuggingFace lj1995/VoiceConversionWebUI.
# This repository has all models in one place without CDN redirects.
#
# Usage:
#   ./tools/download_assets.sh [--uvr5] [--rmvpe] [--pretrained] [--hubert] [--all]

set -u  # no -e because we tolerate some file-specific failures

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BASE_URL="https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"

download() {
    local url="$1"
    local dest="$2"
    mkdir -p "$(dirname "$dest")"
    if [ -s "$dest" ] && [ $(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest") -gt 10000 ]; then
        echo "  [skip] $dest (already present)"
        return 0
    fi
    echo "  [dl] $url -> $dest"
    curl -sSL --connect-timeout 15 --retry 3 --retry-delay 2 -o "$dest" "$url"
    if [ $? -eq 0 ] && [ -s "$dest" ]; then
        local sz=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest")
        echo "     OK ($sz bytes)"
    else
        echo "     FAILED"
        rm -f "$dest"
        return 1
    fi
}

download_hubert() {
    echo "=== HuBERT ==="
    download "$BASE_URL/hubert_base.pt" "assets/hubert/hubert_base.pt"
}

download_rmvpe() {
    echo "=== RMVPE ==="
    download "$BASE_URL/rmvpe.pt" "assets/rmvpe/rmvpe.pt"
    download "$BASE_URL/rmvpe.onnx" "assets/rmvpe/rmvpe.onnx"
}

download_pretrained_v2() {
    echo "=== Pretrained v2 ==="
    for model in D32k.pth D40k.pth D48k.pth G32k.pth G40k.pth G48k.pth \
                 f0D32k.pth f0D40k.pth f0D48k.pth f0G32k.pth f0G40k.pth f0G48k.pth; do
        download "$BASE_URL/pretrained_v2/$model" "assets/pretrained_v2/$model"
    done
}

download_pretrained_v1() {
    echo "=== Pretrained v1 ==="
    for model in D32k.pth D40k.pth D48k.pth G32k.pth G40k.pth G48k.pth \
                 f0D32k.pth f0D40k.pth f0D48k.pth f0G32k.pth f0G40k.pth f0G48k.pth; do
        download "$BASE_URL/pretrained/$model" "assets/pretrained/$model"
    done
}

download_uvr5() {
    echo "=== UVR5 ==="
    # Note: URLs need URL-encoding for the Chinese characters in some filenames.
    # This lj1995 repo has only a subset of models; some use different naming.
    download "$BASE_URL/uvr5_weights/HP2_all_vocals.pth" "assets/uvr5_weights/HP2_all_vocals.pth"
    download "$BASE_URL/uvr5_weights/HP3_all_vocals.pth" "assets/uvr5_weights/HP3_all_vocals.pth"
    download "$BASE_URL/uvr5_weights/HP5_only_main_vocal.pth" "assets/uvr5_weights/HP5_only_main_vocal.pth"
    download "$BASE_URL/uvr5_weights/VR-DeEchoAggressive.pth" "assets/uvr5_weights/VR-DeEchoAggressive.pth"
    download "$BASE_URL/uvr5_weights/VR-DeEchoDeReverb.pth" "assets/uvr5_weights/VR-DeEchoDeReverb.pth"
    download "$BASE_URL/uvr5_weights/VR-DeEchoNormal.pth" "assets/uvr5_weights/VR-DeEchoNormal.pth"
    # Chinese-named variants — use urlencoded paths.
    download "$BASE_URL/uvr5_weights/HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth" \
             "assets/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth"
    download "$BASE_URL/uvr5_weights/HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth" \
             "assets/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth"
    # ONNX de-reverb
    download "$BASE_URL/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx" \
             "assets/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx"
}

if [[ $# -eq 0 ]] || [[ "$1" == "--all" ]]; then
    download_hubert
    download_rmvpe
    download_pretrained_v2
    download_uvr5
    exit 0
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hubert) download_hubert ;;
        --rmvpe) download_rmvpe ;;
        --pretrained) download_pretrained_v2 ;;
        --pretrained-v1) download_pretrained_v1 ;;
        --uvr5) download_uvr5 ;;
        --all)
            download_hubert
            download_rmvpe
            download_pretrained_v2
            download_uvr5
            ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
    shift
done
