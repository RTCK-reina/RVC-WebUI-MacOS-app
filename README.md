<div align="center">

# RVC-WebUI-MacOS

**A native macOS `.app` of Retrieval-based Voice Conversion.**
SwiftUI frontend + bundled Python backend. No browser, no network, no pip install.

[![macOS](https://img.shields.io/badge/macOS-12.0%2B-black?style=for-the-badge&logo=apple)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-0071c5?style=for-the-badge)](https://developer.apple.com/metal/pytorch/)
[![Licence](https://img.shields.io/github/license/RTCKPRO/RVC-WebUI-MacOS?style=for-the-badge)](./LICENSE)

[**English**](./README.md) · [**日本語**](./docs/jp/README.ja.md) · [**中文简体**](./docs/cn/README.cn.md) · [**한국어**](./docs/kr/README.ko.md) · [**Français**](./docs/fr/README.fr.md) · [**Português**](./docs/pt/README.pt.md) · [**Türkçe**](./docs/tr/README.tr.md)

</div>

---

## What is this

RVC-WebUI-MacOS repackages the [Retrieval-based Voice Conversion WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI) as a **single standalone `.app`** for Apple Silicon. Everything — PyTorch, fairseq, all pretrained models (HuBERT, RMVPE, UVR5, pretrained_v2) — ships inside the bundle. First launch is a double-click; no conda, no pip, no Homebrew, no localhost URL, no internet required after download.

The original project uses Gradio in a browser and FreeSimpleGUI for the realtime VC window. This fork replaces both with a **SwiftUI frontend** that talks to a **subprocess Python backend** over JSON-RPC on stdin/stdout.

## Features

- **Fully offline** — all ML weights are inside the bundle. No asset download step, no HuggingFace fetch.
- **Apple Silicon first** — PyTorch MPS backend out of the box. Correctly falls back to CPU when MPS can't handle an op.
- **Always-on resource monitor** — CPU / unified-memory / MPS usage in the toolbar, refreshed every second.
- **Honest progress bars** — per-task percent, phase label, ETA. Cancel buttons only appear where the operation is actually interruptible.
- **Robust cancellation** — 3-tier defence (cooperative flag → SIGTERM → SIGKILL + process-tree reap) kills any training or UVR5 job in 1–2 seconds.
- **All RVC features in one app**:
  - Single-file and batch inference
  - UVR5 vocal / instrumental separation with model-chooser guide (which HP/DeEcho/DeReverb to pick, and why)
  - Optional auto-polish chain (second-pass DeReverb) after vocal extraction
  - Full training pipeline: preprocess → F0 / feature extract → train → index (one-click `train_all` runs the entire chain)
  - Model management: compare, merge, extract (slim), info edit
  - ONNX export
  - Realtime voice changer with device picker, hot parameter updates, and per-block diagnostics (meter / badge / metrics / event log)
- **Human-readable layout** — every user file lives under `~/Documents/RVC-WebUI/`, nothing scattered across hidden application-support folders.
- **Defaults that don't degrade audio** — output is int16 FLAC (lossless); WAV / MP3 / M4A still available. Volume is preserved as-is — no accidental float32 re-scaling.
- **Build integrity** — `download_assets.sh` and `build_app.sh` verify every model file against SHA-256 checksums before bundling.

## System requirements

| | Minimum | Recommended |
|---|---|---|
| macOS | 12.0 Monterey | 14.0 Sonoma or later |
| CPU | Apple Silicon (M1) | M2 Pro / M3 Pro or better |
| RAM | 8 GB | 16 GB+ (training is memory hungry) |
| Disk | 8 GB free | 20 GB+ if training |

Intel Macs are **not supported** — the bundled PyTorch is ARM64-only.

## Installation

### For end users

1. Download `RVC-WebUI.app.zip` from the latest [Release](https://github.com/RTCKPRO/RVC-WebUI-MacOS/releases).
2. Unzip, drag `RVC-WebUI.app` into `/Applications`.
3. Double-click to launch. On first run, Gatekeeper may ask you to confirm — right-click the app → **Open** → **Open** in the dialog.

On first launch the app creates `~/Documents/RVC-WebUI/` and subdirectories for your inputs, outputs, models, and logs. That's the only place it writes.

### For developers / building from source

```bash
# Prereqs: Homebrew, Xcode CLT, Miniforge/conda
brew install xcodegen
conda install -n base -c conda-forge conda-pack

# 1. Clone
git clone https://github.com/RTCKPRO/RVC-WebUI-MacOS.git
cd RVC-WebUI-MacOS

# 2. Create the conda env (Python 3.10 + PyTorch MPS + fairseq etc.)
./setup_conda_env.sh
conda activate rvc

# 3. (Optional) Smoke-test the Python backend standalone
python tools/test_rpc.py
# expect: "ready" notification → initialize response → resource_stats every second

# 4. Download model assets from HuggingFace (hubert / rmvpe / pretrained_v2 / uvr5_weights, about 2 GB)
./tools/download_assets.sh --all

# 5. Build the full .app bundle
./build_app.sh
# Produces: build/RVC-WebUI.app  (about 4 GB including PyTorch and all models)
```

Build flags:

- `--skip-conda` — reuse previously packed Python env (`build/python_env/`)
- `--skip-xcode` — reuse previously built Swift binary
- `--skip-sign` — skip code signing (fine for local dev; not for distribution)

For distribution-signed builds:

```bash
export CODE_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
./build_app.sh
xcrun notarytool submit build/RVC-WebUI.app --keychain-profile AC_PROFILE --wait
xcrun stapler staple build/RVC-WebUI.app
```

## Architecture

```
┌──────────────────────────────────────────────┐
│          SwiftUI .app (RVCApp)               │
│   NavigationSplitView + TabView              │
│   toolbar: CPU / MEM / MPS monitor           │
│   Views: Inference / Separation / Training   │
│           Realtime / Models                  │
└───────────────────┬──────────────────────────┘
                    │ JSON-RPC 2.0 over stdio
                    │ (no network, no sockets)
┌───────────────────▼──────────────────────────┐
│        Python subprocess (rpc_server.py)     │
│   27 RPC methods across two modules:         │
│   rpc_server  — vc_single · vc_multi · uvr5  │
│     model_info/compare/merge/extract · onnx  │
│     realtime_start/stop/update_params        │
│     cancel · shutdown · resource_stats       │
│   rpc_training — preprocess · extract_f0     │
│     train · train_index · train_all          │
│   psutil + torch.mps resource sampling       │
└──────────────────────────────────────────────┘
```

- Frontend: `RVCApp/` — SwiftUI, generated with `xcodegen` from `project.yml`
- Bridge: `RVCApp/RVCApp/Bridge/PythonBridge.swift` — launches the Python subprocess, dispatches RPC calls, routes progress / resource notifications to `@Published` state
- Backend: `rpc_server.py` + `rpc_training.py` — JSON-RPC methods wrap `infer/modules/vc`, `infer/modules/uvr5`, and training scripts; stdout is line-buffered for prompt first response. Blocking methods (inference, training, UVR5) are serialized on a dedicated worker thread so two heavy ops never overlap.
- Assets: `assets/hubert/`, `assets/rmvpe/`, `assets/pretrained_v2/`, `assets/uvr5_weights/` — all copied into `.app/Contents/Resources/rvc_backend/assets/` at build time, verified against `sha256.env`
- Python runtime: `build/python_env/` via `conda-pack`, then embedded at `.app/Contents/Resources/python/`

See [`BUILD_NATIVE_APP.md`](./BUILD_NATIVE_APP.md) for the full build pipeline and architecture notes.

## File layout

**Inside the bundle** (`RVC-WebUI.app/Contents/Resources/`) — read-only:

```
rvc_backend/    # Python code + assets, copied from repo
python/         # Bundled Python 3.10 runtime with all deps
```

**In your home directory** (`~/Documents/RVC-WebUI/`) — all your data:

```
input/
  audio/          # Drop files here for inference
  training/       # Training datasets
output/
  inference/      # Single-file conversion results (FLAC by default)
  batch/          # Batch conversion results
  separation/     # UVR5 vocals/ and accompaniment/
  onnx/           # ONNX exports
models/           # Your trained .pth voice models
indices/          # FAISS .index files
logs/             # Training checkpoints + logs, one dir per experiment
configs/inuse/    # Runtime config
temp/             # Scratch space, cleared at startup
```

## Troubleshooting

**"RVC-WebUI.app is damaged and can't be opened"** — Ad-hoc signed builds trip Gatekeeper on fresh downloads. Fix:
```bash
xattr -cr /Applications/RVC-WebUI.app
```

**"No supported NVIDIA GPU found"** — Expected. The app runs on MPS; this is a log line from an upstream code path, not an error.

**Training fails immediately in feature extraction** — Fixed in this fork. If you're building from a very old checkout, make sure `infer/lib/torch_compat.py` exists and is imported before `fairseq` in `extract_feature_print.py`, `infer/modules/vc/utils.py`, and `infer/lib/rtrvc.py`. This shim disables PyTorch 2.6+'s `weights_only=True` default that fairseq's HuBERT loader trips on.

**Inference output is clipping / much louder than input** — Fixed. Earlier versions accidentally passed `f32=True` to the audio writer, which re-scaled int16 samples into float32 range — a ~256× volume boost. Current builds preserve the original int16 amplitude.

**MPS out-of-memory during training** — drop `batch_size_per_gpu`, close other apps, or set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` (already set at launch, but worth checking in `~/Documents/RVC-WebUI/logs/<exp>/train.log`).

**Cancel / Stop button doesn't kill training** — Fixed. Training now uses a 3-tier kill chain (cooperative cancel flag → SIGTERM with 2 s timeout → SIGKILL + `psutil` process-tree reap). If the backend itself hangs, PythonBridge performs a hard restart of the subprocess.

**First launch is slow** — fairseq + torch cold-import is ~3 s on M1, ~2 s on M3. The splash shows "waiting for backend" until `alive` lands; no action needed.

## Testing

The Python backend has a comprehensive test suite (`tests/`):

```bash
conda activate rvc
pytest                    # run all tests
pytest tests/test_rpc_protocol.py   # RPC protocol compliance
pytest tests/test_audio.py          # audio I/O & int16 volume preservation
```

Key test modules:
- `test_rpc_protocol.py` — JSON-RPC 2.0 compliance, method dispatch, error codes
- `test_rpc_integration.py` / `test_rpc_runtime.py` — end-to-end backend lifecycle
- `test_audio.py` — format conversion, sample-rate handling, volume preservation
- `test_process_ckpt_merge.py` — checkpoint merge / extract correctness
- `test_realtime_vc_sola.py` — SOLA cross-fade algorithm
- `test_perf_optimizations.py` — thread pool, memory, and latency assertions
- `test_torch_compat.py` — PyTorch 2.6+ compatibility shim

## Development

The SwiftUI project is regenerated every build from `RVCApp/project.yml` via xcodegen, so don't hand-edit `RVCApp.xcodeproj`. Open `RVCApp.xcodeproj` in Xcode and Run — in dev mode the app launches the repo's `rpc_server.py` via your active conda env (not the bundled Python), which gives you a much faster iteration loop.

Python-side changes:
- Source lives at repo root (`rpc_server.py`, `rpc_training.py`, `infer/`, `rvc/`, `configs/`, `i18n/`, `tools/`)
- `./build_app.sh --skip-conda --skip-xcode` re-syncs the Python backend into an existing `.app` without rebuilding the Swift binary or repacking Python
- For ad-hoc iteration against an already-built `.app`, `rsync -a infer/ build/RVC-WebUI.app/Contents/Resources/rvc_backend/infer/` is enough

Gradio / FreeSimpleGUI UI code has been fully removed — the only frontend is SwiftUI. Legacy `gui.py` remains as a reference but is not used at runtime.

## Credits

- Upstream voice conversion framework: [fumiama/Retrieval-based-Voice-Conversion-WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI)
- Core building blocks: [ContentVec](https://github.com/auspicious3000/contentvec), [VITS](https://github.com/jaywalnut310/vits), [HIFIGAN](https://github.com/jik876/hifi-gan), [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui), [audio-slicer](https://github.com/openvpi/audio-slicer), [RMVPE](https://github.com/Dream-High/RMVPE) (pretrained model by [yxlllc](https://github.com/yxlllc/RMVPE) and [RVC-Boss](https://github.com/RVC-Boss))
- Initial macOS fork: [Nevil Patel](https://github.com/NevilPatel01/RVC-WebUI-MacOS)
- Native `.app` rework: this repository

## License

MIT. See [LICENSE](./LICENSE).
