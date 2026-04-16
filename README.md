<div align="center">

# RVC-WebUI-MacOS

**A native macOS `.app` for Retrieval-based Voice Conversion.**  
SwiftUI frontend · bundled Python backend · JSON-RPC over stdio.  
No browser. No network. No `pip install`.

[![macOS](https://img.shields.io/badge/macOS-12.0%2B-black?style=for-the-badge&logo=apple)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-0071c5?style=for-the-badge)](https://developer.apple.com/metal/pytorch/)
[![Licence](https://img.shields.io/github/license/RTCK-reina/RVC-WebUI-MacOS-app?style=for-the-badge)](./LICENSE)

</div>

---

## What is this

RVC-WebUI-MacOS repackages [Retrieval-based Voice Conversion WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI) as a **single standalone `.app`** for Apple Silicon. PyTorch, fairseq, and all pretrained models (HuBERT, RMVPE, UVR5, pretrained_v2) ship inside the bundle. First launch is a double-click — no conda, no pip, no Homebrew, no localhost URL, no internet required after download.

### Differences from the upstream fork ([NevilPatel01/RVC-WebUI-MacOS](https://github.com/NevilPatel01/RVC-WebUI-MacOS))

| Area | Upstream fork | This repo |
|---|---|---|
| Frontend | Gradio web UI / FreeSimpleGUI | SwiftUI `.app` |
| Backend protocol | HTTP / Gradio | JSON-RPC 2.0 over stdin/stdout |
| Realtime VC | FreeSimpleGUI window | SwiftUI Realtime view |
| User data | `cwd`-relative | `~/Documents/RVC-WebUI/` |
| Config paths | `.env` file | `RVC_BASE_DIR` / `RVC_USER_DIR` env vars |
| ONNX inference | Not supported | `OnnxSynthesizer` + CoreML EP (Apple Neural Engine) |
| Training DataLoader | Fixed `num_workers` | Adaptive `num_workers`, `persistent_workers`, `prefetch_factor` |
| Gradient Accumulation | Not supported | Configurable `accumulation_steps` |
| UVR5 preprocessing | Serial | Parallel (`ThreadPoolExecutor`, up to 4 workers) |
| Thread-safe `torch.load` | Global monkey-patch | Scoped `legacy_load()` context manager |
| FAISS zero-division | Crashes on exact match | NaN guard |
| MPS crash (UVR5) | Present | Fixed |
| Test suite | None | 103 tests (pytest) |

---

## Features

- **Fully offline** — all ML weights are inside the bundle. No HuggingFace fetch, no asset download step.
- **Apple Silicon first** — PyTorch MPS backend. Falls back to CPU automatically when MPS can't handle an op (`PYTORCH_ENABLE_MPS_FALLBACK=1`).
- **Resource monitor** — CPU / unified-memory / MPS usage in the toolbar, refreshed every second via `psutil`.
- **Progress & cancellation** — per-task percent, phase label. Cancel buttons only appear for interruptible operations.
- **All RVC features**:
  - Single-file inference
  - Batch inference
  - UVR5 vocal / instrumental separation (HP / DeEcho / DeReverb model chooser)
  - Full training pipeline: preprocess → F0 extract → feature extract → train → build index
  - Model management: info, compare, merge, extract (slim)
  - ONNX export
  - Realtime voice changer with device picker and hot parameter updates
- **Human-readable layout** — all user files under `~/Documents/RVC-WebUI/`, nothing in hidden application-support folders.
- **Lossless output by default** — FLAC; WAV / MP3 / M4A still available.

---

## System requirements

| | Minimum | Recommended |
|---|---|---|
| macOS | 12.0 Monterey | 14.0 Sonoma or later |
| CPU | Apple Silicon (M1) | M2 Pro / M3 Pro or better |
| RAM | 8 GB | 16 GB+ (training is memory-hungry) |
| Disk | 8 GB free | 20 GB+ if training |

Intel Macs are **not supported** — the bundled PyTorch is ARM64-only.

---

## Installation

### End users

1. Download `RVC-WebUI.app.zip` from the latest [Release](https://github.com/RTCK-reina/RVC-WebUI-MacOS-app/releases).
2. Unzip and drag `RVC-WebUI.app` to `/Applications`.
3. First launch: right-click → **Open** → **Open** (Gatekeeper confirmation for unsigned builds).

On first launch the app creates `~/Documents/RVC-WebUI/` and all subdirectories. That is the only location it writes to.

### Developers / building from source

**Prerequisites**: Homebrew, Xcode Command Line Tools, Miniforge or conda.

```bash
# 1. Clone
git clone https://github.com/RTCK-reina/RVC-WebUI-MacOS-app.git
cd RVC-WebUI-MacOS-app

# 2. Create the conda environment (Python 3.10 + PyTorch MPS + fairseq + all deps)
./setup_conda_env.sh
conda activate rvc

# 3. (Optional) smoke-test the Python backend standalone
python tools/test_rpc.py
# Expected: "ready" notification, initialize response, resource_stats every second

# 4. Build the standalone .app bundle (~4 GB including PyTorch and all models)
./build_app.sh
# Output: build/RVC-WebUI.app
```

`build_app.sh` flags:

| Flag | Effect |
|---|---|
| `--skip-conda` | Reuse the previously packed Python env in `build/python_env/` |
| `--skip-xcode` | Reuse the previously built Swift binary |
| `--skip-sign` | Skip code signing (fine for local dev) |

For distribution-signed and notarized builds:

```bash
export CODE_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
./build_app.sh
xcrun notarytool submit build/RVC-WebUI.app --keychain-profile AC_PROFILE --wait
xcrun stapler staple build/RVC-WebUI.app
```

---

## Architecture

```
┌──────────────────────────────────────────────┐
│          SwiftUI .app (RVCApp)               │
│  NavigationSplitView + TabView               │
│  Toolbar: CPU / MEM / MPS resource monitor   │
└───────────────────┬──────────────────────────┘
                    │ JSON-RPC 2.0 over stdio
                    │ (no network, no sockets)
┌───────────────────▼──────────────────────────┐
│    Python subprocess (rpc_server.py)         │
│  Inference · UVR5 · Training · Realtime      │
│  ONNX/CoreML · psutil resource sampling      │
└──────────────────────────────────────────────┘
```

| Component | Location |
|---|---|
| SwiftUI frontend | `RVCApp/` (project generated with xcodegen from `RVCApp/project.yml`) |
| Swift ↔ Python bridge | `RVCApp/RVCApp/Bridge/PythonBridge.swift` |
| JSON-RPC server | `rpc_server.py` |
| Training RPC handlers | `rpc_training.py` |
| VC / inference pipeline | `infer/modules/vc/` |
| UVR5 separation | `infer/modules/uvr5/` |
| Training loop | `infer/modules/train/train.py` |
| ONNX inference | `rvc/onnx/infer.py` |
| Config | `configs/config.py` |
| Pretrained assets | `assets/` (hubert, rmvpe, pretrained_v2, uvr5_weights) |

---

## File layout

**Inside the bundle** (`RVC-WebUI.app/Contents/Resources/`) — read-only:

```
rvc_backend/    # Python code + assets copied from the repo
python/         # Bundled Python 3.10 runtime with all dependencies
```

**User data** (`~/Documents/RVC-WebUI/`) — all your files:

```
input/
  audio/            # Source files for inference
  training/         # Training datasets
output/
  inference/        # Single-file conversion results
  batch/            # Batch conversion results
  separation/
    vocals/
    accompaniment/
  onnx/             # ONNX export outputs
models/             # Trained voice models (.pth)
indices/            # FAISS .index files
logs/               # Training checkpoints + logs (one dir per experiment)
configs/inuse/      # Runtime config copies (v1/, v2/)
temp/               # Scratch space, cleared at startup
```

---

## Performance optimizations

Four quality-preserving optimizations are included. All are on by default or configurable via JSON / env var.

### 1. ONNX Runtime + CoreML ExecutionProvider (Apple Neural Engine)

When `RVC_USE_ONNX=1` and a `.onnx` file exists alongside the `.pth` model, inference runs through ONNX Runtime with the CoreML ExecutionProvider, offloading computation to the Apple Neural Engine.

**How to enable:**

```bash
# Install ONNX Runtime (already listed in requirements/app.txt for macOS)
pip install onnxruntime

# Export the PyTorch model to ONNX first (via the app's "ONNX Export" screen
# or rpc_export_onnx RPC method), then launch with:
RVC_USE_ONNX=1 python rpc_server.py
```

The `.onnx` file must be in the same directory as the `.pth` and share the same stem (e.g., `my_voice.pth` → `my_voice.onnx`). If the file is absent or `onnxruntime` is not installed, inference falls back to PyTorch silently.

### 2. DataLoader optimization

Training DataLoader workers are configured automatically and persistently:

| Parameter | Default | Where to change |
|---|---|---|
| `num_workers` | `min(max(cpu_count // 2, 2), 6)` | `configs/v1/40k.json` → `train.num_workers` |
| `prefetch_factor` | `8` | `configs/v1/40k.json` → `train.prefetch_factor` |
| `persistent_workers` | `true` (when `num_workers > 0`) | automatic |

`num_workers=0` disables background workers (main-thread loading, no prefetch). `persistent_workers=True` keeps worker processes alive between epochs, eliminating per-epoch spawn overhead.

### 3. Gradient Accumulation

Simulates a larger effective batch size without extra VRAM by accumulating gradients over multiple micro-batches before calling `optimizer.step()`.

| Parameter | Default | Effect |
|---|---|---|
| `accumulation_steps` | `1` | `1` = disabled (original behaviour) |
| | `4` | Effective batch size = `batch_size × 4` |

**How to change:**

```json
// configs/v1/40k.json  (or 32k.json, 48k.json, v2/*)
{
  "train": {
    "accumulation_steps": 4
  }
}
```

Mathematically equivalent to a larger batch (same gradient sum). Does not affect audio quality.

### 4. UVR5 audio separation — parallel preprocessing

Before running the separation model, input files that need format conversion (non-44100 Hz or mono) are resampled in parallel using a `ThreadPoolExecutor` (up to 4 workers). The GPU/MPS model inference step remains serial. This reduces wall-clock time when separating large batches.

---

## Configuration reference

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `RVC_USE_ONNX` | *(unset)* | Set to `1`, `true`, or `yes` to enable ONNX inference |
| `RVC_BASE_DIR` | repo root / bundle Resources | Path to the read-only bundle assets |
| `RVC_USER_DIR` | `~/Documents/RVC-WebUI` | Path to the user data directory |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` (set at launch) | Fall back CPU-side for MPS-unsupported ops |
| `OMP_NUM_THREADS` | `1` (set at launch) | OpenMP thread count (avoids MPS contention) |
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | *(unset)* | Set to `0.0` to reduce MPS OOM during training |

### Training config parameters (`configs/v*/**.json`)

All values live under the `"train"` key:

| Key | Default | Description |
|---|---|---|
| `batch_size` | `4` | Samples per micro-batch |
| `accumulation_steps` | `1` | Gradient accumulation steps (effective batch = `batch_size × steps`) |
| `num_workers` | auto | DataLoader background workers (0 = main thread) |
| `prefetch_factor` | `8` | Batches to prefetch per worker (ignored when `num_workers=0`) |
| `epochs` | `20000` | Total training epochs |
| `learning_rate` | `1e-4` | Initial learning rate |
| `lr_decay` | `0.999875` | Multiplicative LR decay per epoch |
| `fp16_run` | `true` | Half-precision training (forced `false` on MPS/CPU) |
| `log_interval` | `200` | Steps between log lines |
| `seed` | `1234` | Random seed |

---

## Running without the `.app` (developer mode)

```bash
conda activate rvc

# Start the JSON-RPC backend directly
python rpc_server.py [--port 7865] [--base-dir /path/to/repo] [--user-dir ~/Documents/RVC-WebUI]

# Optionally open the legacy Gradio web UI (requires gradio / main.txt deps)
python web.py
```

In Xcode, running the SwiftUI target in debug mode automatically launches `rpc_server.py` via the active conda environment, giving a fast iteration loop without rebuilding the `.app`.

---

## Testing

```bash
conda activate rvc
pip install pytest

# Run all tests (103 tests, ~2 skipped on machines without real MPS/GPU)
pytest tests/ -v

# Run a specific suite
pytest tests/test_torch_compat.py -v
pytest tests/test_perf_optimizations.py -v
pytest tests/test_rpc_protocol.py -v
```

Test suites:

| File | What it covers |
|---|---|
| `test_torch_compat.py` | `legacy_load()` thread-safety and reentrancy |
| `test_perf_optimizations.py` | ONNX loading, DataLoader config, gradient accumulation, UVR5 parallelization |
| `test_rpc_protocol.py` | JSON-RPC message format, task registry, `_dispatch`, `_list_files` |
| `test_threadpool.py` | Training thread pool and cancellation |
| `test_device.py` | `empty_device_cache()` on MPS/CPU |
| `test_audio.py` | Audio I/O utilities (skipped without real numpy/PyAV) |

---

## Troubleshooting

**"RVC-WebUI.app is damaged and can't be opened"**  
Ad-hoc signed builds trigger Gatekeeper on fresh downloads. Fix:
```bash
xattr -cr /Applications/RVC-WebUI.app
```

**"No supported NVIDIA GPU found"**  
Expected — this is an upstream log line, not an error. The app runs on MPS.

**Training fails in feature extraction**  
Ensure `infer/lib/torch_compat.py` is imported before `fairseq` at each call site (`extract_feature_print.py`, `infer/modules/vc/utils.py`, `infer/lib/rtrvc.py`). This shim disables PyTorch 2.6+'s `weights_only=True` default that trips fairseq's HuBERT loader.

**MPS out-of-memory during training**  
Lower `batch_size`, close other apps, or set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`.

**ONNX inference not activating**  
Check that `RVC_USE_ONNX=1` is set, `onnxruntime` is installed (`pip install onnxruntime`), and a `.onnx` file with the same name as your `.pth` exists in the same directory.

**First launch is slow**  
fairseq + torch cold-import takes ~3 s on M1 / ~2 s on M3. The splash shows "waiting for backend" until `alive` lands — no action needed.

---

## Development

The Xcode project is regenerated each build from `RVCApp/project.yml` via xcodegen. Don't hand-edit `RVCApp.xcodeproj`.

For quick Python-only iteration against an existing `.app`:
```bash
# Re-sync Python source into the bundle without rebuilding Swift or repacking conda
./build_app.sh --skip-conda --skip-xcode

# Or even faster — rsync just the changed module
rsync -a infer/ build/RVC-WebUI.app/Contents/Resources/rvc_backend/infer/
```

Branch history of significant changes merged into `main`:

| PR | Area |
|---|---|
| #1 | Initial SwiftUI `.app` + JSON-RPC backend |
| #2 | MPS inference crash fix + UVR5 file-destruction bug fix |
| #3 | `legacy_load()` context manager, FAISS NaN guard, training fixes |
| #4 | FAISS index cache, load_vc performance |
| #5 | Thread-pool improvements, busy-wait elimination, async save/readwave |
| #6 | Training memory reduction, GPU cache cap, MPS fallback fixes |
| #7 | Copilot review follow-ups (13 fixes) |
| #8 | ONNX+CoreML EP, DataLoader optimization, Gradient Accumulation, UVR5 parallel preprocessing, 103 tests |

---

## Credits

- Upstream voice conversion: [fumiama/Retrieval-based-Voice-Conversion-WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI)
- Core building blocks: [ContentVec](https://github.com/auspicious3000/contentvec), [VITS](https://github.com/jaywalnut310/vits), [HiFi-GAN](https://github.com/jik876/hifi-gan), [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui), [audio-slicer](https://github.com/openvpi/audio-slicer), [RMVPE](https://github.com/Dream-High/RMVPE) (pretrained by [yxlllc](https://github.com/yxlllc/RMVPE) and [RVC-Boss](https://github.com/RVC-Boss))
- Initial macOS fork: [Nevil Patel (NevilPatel01)](https://github.com/NevilPatel01/RVC-WebUI-MacOS)
- Native `.app` + performance work: this repository

## License

MIT. See [LICENSE](./LICENSE).
