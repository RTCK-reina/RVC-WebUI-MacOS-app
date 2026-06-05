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
SHA_FILE="${ROOT_DIR}/sha256.env"

if command -v shasum >/dev/null 2>&1; then
    _sha256() { shasum -a 256 "$1" | awk '{print $1}'; }
elif command -v sha256sum >/dev/null 2>&1; then
    _sha256() { sha256sum "$1" | awk '{print $1}'; }
else
    _sha256() { echo ""; }  # no hasher available — fall back to size gate
fi

# Look up the expected SHA-256 for a sha256.env key. Echoes the 64-hex hash, or
# nothing if the key is absent/invalid. Matches build_app.sh's lookup.
_expected_hash() {
    local key="$1"
    [ -f "$SHA_FILE" ] || return 0
    awk -F= -v k="$key" '
        { gsub(/[[:space:]]/, "", $1) }
        $1 == k { gsub(/[[:space:]]/, "", $2); print $2; exit }
    ' "$SHA_FILE"
}

# download <url> <dest> [sha256_key]
#
# With a sha256 key: skip only when the existing file's hash matches, and after
# downloading, verify the hash — removing the file and failing on mismatch.
# Without a key: fall back to a 10KB size floor. curl uses --fail so an HTTP
# 4xx/5xx error page is never saved as a "present" asset.
download() {
    local url="$1"
    local dest="$2"
    local key="${3:-}"
    mkdir -p "$(dirname "$dest")"

    local expected=""
    if [ -n "$key" ]; then
        expected="$(_expected_hash "$key")"
        case "$expected" in
            *[!A-Fa-f0-9]* | "") expected="" ;;  # absent/invalid → no hash gate
            *) [ "${#expected}" -eq 64 ] || expected="" ;;
        esac
    fi

    if [ -s "$dest" ]; then
        if [ -n "$expected" ]; then
            if [ "$(_sha256 "$dest")" = "$expected" ]; then
                echo "  [skip] $dest (sha256 ok)"
                return 0
            fi
            echo "  [stale] $dest (sha256 mismatch — re-downloading)"
            rm -f "$dest"
        elif [ "$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest")" -gt 10000 ]; then
            echo "  [skip] $dest (already present)"
            return 0
        fi
    fi

    echo "  [dl] $url -> $dest"
    if ! curl -fsSL --connect-timeout 15 --retry 3 --retry-delay 2 -o "$dest" "$url"; then
        echo "     FAILED (curl/http error)"
        rm -f "$dest"
        return 1
    fi
    if [ ! -s "$dest" ]; then
        echo "     FAILED (empty)"
        rm -f "$dest"
        return 1
    fi
    if [ -n "$expected" ]; then
        local actual; actual="$(_sha256 "$dest")"
        if [ "$actual" != "$expected" ]; then
            echo "     FAILED (sha256 mismatch: expected $expected got $actual)"
            rm -f "$dest"
            return 1
        fi
        echo "     OK (sha256 verified)"
    else
        local sz; sz=$(stat -f%z "$dest" 2>/dev/null || stat -c%s "$dest")
        echo "     OK ($sz bytes)"
    fi
}

download_hubert() {
    echo "=== HuBERT ==="
    download "$BASE_URL/hubert_base.pt" "assets/hubert/hubert_base.pt" "sha256_hubert_base_pt"
}

download_rmvpe() {
    echo "=== RMVPE ==="
    download "$BASE_URL/rmvpe.pt" "assets/rmvpe/rmvpe.pt" "sha256_rmvpe_pt"
    download "$BASE_URL/rmvpe.onnx" "assets/rmvpe/rmvpe.onnx" "sha256_rmvpe_onnx"
}

download_pretrained_v2() {
    echo "=== Pretrained v2 ==="
    for model in D32k.pth D40k.pth D48k.pth G32k.pth G40k.pth G48k.pth \
                 f0D32k.pth f0D40k.pth f0D48k.pth f0G32k.pth f0G40k.pth f0G48k.pth; do
        download "$BASE_URL/pretrained_v2/$model" "assets/pretrained_v2/$model" \
                 "sha256_v2_${model%.pth}_pth"
    done
}

download_pretrained_v1() {
    echo "=== Pretrained v1 ==="
    for model in D32k.pth D40k.pth D48k.pth G32k.pth G40k.pth G48k.pth \
                 f0D32k.pth f0D40k.pth f0D48k.pth f0G32k.pth f0G40k.pth f0G48k.pth; do
        download "$BASE_URL/pretrained/$model" "assets/pretrained/$model" \
                 "sha256_v1_${model%.pth}_pth"
    done
}

download_uvr5() {
    echo "=== UVR5 ==="
    # Note: URLs need URL-encoding for the Chinese characters in some filenames.
    # This lj1995 repo has only a subset of models; some use different naming.
    download "$BASE_URL/uvr5_weights/HP2_all_vocals.pth" "assets/uvr5_weights/HP2_all_vocals.pth" \
             "sha256_uvr5_HP2_all_vocals_pth"
    download "$BASE_URL/uvr5_weights/HP3_all_vocals.pth" "assets/uvr5_weights/HP3_all_vocals.pth" \
             "sha256_uvr5_HP3_all_vocals_pth"
    download "$BASE_URL/uvr5_weights/HP5_only_main_vocal.pth" "assets/uvr5_weights/HP5_only_main_vocal.pth" \
             "sha256_uvr5_HP5_only_main_vocal_pth"
    download "$BASE_URL/uvr5_weights/VR-DeEchoAggressive.pth" "assets/uvr5_weights/VR-DeEchoAggressive.pth" \
             "sha256_uvr5_VR-DeEchoAggressive_pth"
    download "$BASE_URL/uvr5_weights/VR-DeEchoDeReverb.pth" "assets/uvr5_weights/VR-DeEchoDeReverb.pth" \
             "sha256_uvr5_VR-DeEchoDeReverb_pth"
    download "$BASE_URL/uvr5_weights/VR-DeEchoNormal.pth" "assets/uvr5_weights/VR-DeEchoNormal.pth" \
             "sha256_uvr5_VR-DeEchoNormal_pth"
    # Chinese-named variants — use urlencoded paths.
    download "$BASE_URL/uvr5_weights/HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth" \
             "assets/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth" \
             "sha256_uvr5_HP2-人声vocals+非人声instrumentals_pth"
    download "$BASE_URL/uvr5_weights/HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth" \
             "assets/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth" \
             "sha256_uvr5_HP5-主旋律人声vocals+其他instrumentals_pth"
    # ONNX de-reverb
    download "$BASE_URL/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx" \
             "assets/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx" \
             "sha256_uvr5_vocals_onnx"
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
