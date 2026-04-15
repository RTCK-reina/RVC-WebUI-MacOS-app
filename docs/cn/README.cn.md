<div align="center">

# RVC-WebUI-MacOS

**基于检索的语音转换（RVC）的 macOS 原生 `.app`。**
SwiftUI 前端 + 内置 Python 后端。无需浏览器、无需联网、无需 pip install。

[![macOS](https://img.shields.io/badge/macOS-12.0%2B-black?style=for-the-badge&logo=apple)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-0071c5?style=for-the-badge)](https://developer.apple.com/metal/pytorch/)
[![Licence](https://img.shields.io/github/license/RTCKPRO/RVC-WebUI-MacOS?style=for-the-badge)](../../LICENSE)

[**English**](../../README.md) · [**日本語**](../jp/README.ja.md) · [**中文简体**](./README.cn.md) · [**한국어**](../kr/README.ko.md) · [**Français**](../fr/README.fr.md) · [**Português**](../pt/README.pt.md) · [**Türkçe**](../tr/README.tr.md)

</div>

---

## 这是什么

RVC-WebUI-MacOS 将 [Retrieval-based Voice Conversion WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI) 重新打包为面向 Apple Silicon 的**单一独立 `.app`**。PyTorch、fairseq、所有预训练模型（HuBERT、RMVPE、UVR5、pretrained_v2）全部打包在 bundle 内部。下载后双击即可启动；无需 conda、pip、Homebrew，无需 localhost URL，下载完成后也无需联网。

原项目在浏览器中使用 Gradio，实时变声窗口使用 FreeSimpleGUI。本分支将两者均替换为 **SwiftUI 前端**，通过 stdin/stdout 上的 JSON-RPC 与作为子进程运行的 **Python 后端**通信。

## 功能

- **完全离线** — 所有 ML 权重均内置于 bundle。无资源下载步骤，无 HuggingFace 拉取。
- **Apple Silicon 优先** — 原生支持 PyTorch MPS 后端。MPS 不支持的算子会正确回退到 CPU。
- **常驻资源监视器** — 工具栏每秒刷新 CPU / 统一内存 / MPS 使用率。
- **诚实的进度条** — 按任务显示百分比、阶段标签、预计剩余时间。取消按钮仅出现在真正可中断的操作上。
- **所有 RVC 功能集成于一个应用**:
  - 单文件推理与批量推理
  - UVR5 人声/伴奏分离，附带模型选择指引（何时该选 HP / DeEcho / DeReverb 以及原因）
  - 可选的自动精修链（人声提取后进行第二次 DeReverb）
  - 完整训练流程: 预处理 → F0 / 特征提取 → 训练 → 索引
  - 模型管理: 比较、融合、提取（瘦身）、信息编辑
  - ONNX 导出
  - 实时变声器，支持设备选择与参数热更新
- **清晰的文件布局** — 所有用户文件位于 `~/Documents/RVC-WebUI/` 之下，不会散落在隐藏的 Application Support 目录中。
- **不降低音质的默认值** — 输出默认为 FLAC（无损）；仍可选 WAV / MP3 / M4A。

## 系统要求

| | 最低要求 | 推荐 |
|---|---|---|
| macOS | 12.0 Monterey | 14.0 Sonoma 或更新 |
| CPU | Apple Silicon (M1) | M2 Pro / M3 Pro 或更高 |
| 内存 | 8 GB | 16 GB 以上（训练非常吃内存） |
| 磁盘 | 8 GB 可用 | 训练时 20 GB 以上 |

**不支持** Intel Mac — 内置 PyTorch 仅为 ARM64。

## 安装

### 终端用户

1. 从最新 [Release](https://github.com/RTCKPRO/RVC-WebUI-MacOS/releases) 下载 `RVC-WebUI.app.zip`。
2. 解压后将 `RVC-WebUI.app` 拖入 `/Applications`。
3. 双击启动。首次运行时 Gatekeeper 可能要求确认 — 右键点击应用 → **打开** → 在弹出对话框中再点 **打开**。

首次启动时，应用会创建 `~/Documents/RVC-WebUI/` 以及用于输入、输出、模型、日志的子目录。这是应用写入的唯一位置。

### 开发者 / 从源代码构建

```bash
# 前置要求: Homebrew、Xcode CLT、Miniforge/conda
brew install xcodegen conda-pack

# 1. 克隆
git clone https://github.com/RTCKPRO/RVC-WebUI-MacOS.git
cd RVC-WebUI-MacOS

# 2. 创建 conda 环境（Python 3.10 + PyTorch MPS + fairseq 等）
./setup_conda_env.sh
conda activate rvc

# 3. （可选）独立冒烟测试 Python 后端
python tools/test_rpc.py
# 期望: "ready" 通知 → initialize 响应 → 每秒一次 resource_stats

# 4. 构建完整 .app bundle
./build_app.sh
# 产物: build/RVC-WebUI.app  （含 PyTorch 与全部模型，约 4 GB）
```

构建参数:

- `--skip-conda` — 复用已打包的 Python 环境（`build/python_env/`）
- `--skip-xcode` — 复用已构建的 Swift 可执行文件
- `--skip-sign` — 跳过代码签名（本地开发可以，不适合分发）

用于分发的签名构建:

```bash
export CODE_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
./build_app.sh
xcrun notarytool submit build/RVC-WebUI.app --keychain-profile AC_PROFILE --wait
xcrun stapler staple build/RVC-WebUI.app
```

## 架构

```
┌──────────────────────────────────────────────┐
│          SwiftUI .app (RVCApp)               │
│   NavigationSplitView + TabView              │
│   工具栏: CPU / MEM / MPS 监视器             │
└───────────────────┬──────────────────────────┘
                    │ JSON-RPC 2.0 over stdio
                    │ （无网络、无 socket）
┌───────────────────▼──────────────────────────┐
│        Python 子进程 (rpc_server.py)         │
│   VC · UVR5 · 训练 · 实时 · ONNX             │
│   psutil + torch.mps 采集资源                │
└──────────────────────────────────────────────┘
```

- 前端: `RVCApp/` — SwiftUI，由 `xcodegen` 从 `project.yml` 生成
- 桥接: `RVCApp/RVCApp/Bridge/PythonBridge.swift` — 启动 Python 子进程、分发 RPC 调用、将进度/资源通知路由到 `@Published` 状态
- 后端: `rpc_server.py` + `rpc_training.py` — JSON-RPC 方法封装 `infer/modules/vc`、`infer/modules/uvr5` 以及训练脚本；stdout 采用行缓冲以尽早响应
- 资源文件: `assets/hubert/`、`assets/rmvpe/`、`assets/pretrained_v2/`、`assets/uvr5_weights/` — 构建时全部复制到 `.app/Contents/Resources/rvc_backend/assets/`
- Python 运行时: 通过 `conda-pack` 打包到 `build/python_env/`，再嵌入 `.app/Contents/Resources/python/`

完整构建流程与架构说明参见 [`BUILD_NATIVE_APP.md`](../../BUILD_NATIVE_APP.md)。

## 文件布局

**Bundle 内部** (`RVC-WebUI.app/Contents/Resources/`) — 只读:

```
rvc_backend/    # 从仓库复制的 Python 代码 + 资源
python/         # 内置的 Python 3.10 运行时（含全部依赖）
```

**你的 home 目录** (`~/Documents/RVC-WebUI/`) — 所有用户数据:

```
input/
  audio/          # 放置推理用音频文件
  training/       # 训练数据集
output/
  inference/      # 单文件转换结果（默认 FLAC）
  batch/          # 批量转换结果
  separation/     # UVR5 的 vocals/ 与 accompaniment/
  onnx/           # ONNX 导出
models/           # 你训练好的 .pth 声音模型
indices/          # FAISS .index 文件
logs/             # 训练检查点与日志，每个实验一个目录
configs/inuse/    # 运行时配置
temp/             # 临时空间，启动时清空
```

## 故障排查

**"RVC-WebUI.app 已损坏，无法打开"** — Ad-hoc 签名的构建在全新下载时会被 Gatekeeper 拦截。解决:
```bash
xattr -cr /Applications/RVC-WebUI.app
```

**"No supported NVIDIA GPU found"** — 属于预期。应用运行于 MPS 之上；这是上游代码路径打出的日志行，并非错误。

**训练在特征提取阶段立即失败** — 本分支已修复。如果你是从非常旧的检出构建的，请确认 `infer/lib/torch_compat.py` 存在，并且在 `extract_feature_print.py`、`infer/modules/vc/utils.py`、`infer/lib/rtrvc.py` 中都在 `fairseq` 之前被 import。这个 shim 会关闭 PyTorch 2.6+ 的 `weights_only=True` 默认值，fairseq 的 HuBERT 加载器会被它绊倒。

**训练期间 MPS 内存不足** — 降低 `batch_size_per_gpu`、关闭其他应用，或设置 `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`（启动时已设置，但值得在 `~/Documents/RVC-WebUI/logs/<exp>/train.log` 中确认）。

**首次启动缓慢** — fairseq + torch 冷启动在 M1 约 3 秒、M3 约 2 秒。启动画面会显示"等待后端"直到收到 `alive`，无需操作。

## 开发

SwiftUI 工程每次构建时都会由 xcodegen 根据 `RVCApp/project.yml` 重新生成，因此不要手工编辑 `RVCApp.xcodeproj`。在 Xcode 中打开 `RVCApp.xcodeproj` 后直接 Run — 开发模式下应用会使用当前激活的 conda 环境启动仓库根目录下的 `rpc_server.py`（而不是内置 Python），迭代速度更快。

Python 侧变更:
- 源码位于仓库根目录（`rpc_server.py`、`rpc_training.py`、`infer/`、`rvc/`、`configs/`、`i18n/`、`tools/`）
- `./build_app.sh --skip-conda --skip-xcode` 可以在不重建 Swift 二进制、不重新打包 Python 的情况下把 Python 后端重新同步到已有 `.app`
- 对于已构建 `.app` 的临时迭代，`rsync -a infer/ build/RVC-WebUI.app/Contents/Resources/rvc_backend/infer/` 即可

## 致谢

- 上游语音转换框架: [fumiama/Retrieval-based-Voice-Conversion-WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI)
- 组件来源: [ContentVec](https://github.com/auspicious3000/contentvec)、[VITS](https://github.com/jaywalnut310/vits)、[HIFIGAN](https://github.com/jik876/hifi-gan)、[Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)、[audio-slicer](https://github.com/openvpi/audio-slicer)、[RMVPE](https://github.com/Dream-High/RMVPE)（预训练模型由 [yxlllc](https://github.com/yxlllc/RMVPE) 与 [RVC-Boss](https://github.com/RVC-Boss) 提供）
- 早期 macOS 分支: [Nevil Patel](https://github.com/NevilPatel01/RVC-WebUI-MacOS)
- 原生 `.app` 重构: 本仓库

## 许可证

MIT。详见 [LICENSE](../../LICENSE)。
