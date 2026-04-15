<div align="center">

# RVC-WebUI-MacOS

**Retrieval-based Voice Conversion을 macOS 네이티브 `.app`으로 재구성한 것.**
SwiftUI 프런트엔드 + 번들된 Python 백엔드. 브라우저 불필요, 네트워크 불필요, pip install 불필요.

[![macOS](https://img.shields.io/badge/macOS-12.0%2B-black?style=for-the-badge&logo=apple)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-0071c5?style=for-the-badge)](https://developer.apple.com/metal/pytorch/)
[![Licence](https://img.shields.io/github/license/RTCKPRO/RVC-WebUI-MacOS?style=for-the-badge)](../../LICENSE)

[**English**](../../README.md) · [**日本語**](../jp/README.ja.md) · [**中文简体**](../cn/README.cn.md) · [**한국어**](./README.ko.md) · [**Français**](../fr/README.fr.md) · [**Português**](../pt/README.pt.md) · [**Türkçe**](../tr/README.tr.md)

</div>

---

## 이것이 무엇인가

RVC-WebUI-MacOS는 [Retrieval-based Voice Conversion WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI)를 Apple Silicon용 **단일 독립 실행 `.app`** 으로 재패키징한 것입니다. PyTorch, fairseq 및 모든 사전 학습 모델(HuBERT, RMVPE, UVR5, pretrained_v2)이 번들 내부에 모두 포함되어 있습니다. 다운로드 후 더블클릭으로 바로 실행됩니다 — conda도, pip도, Homebrew도, localhost URL도, 다운로드 이후의 인터넷 연결도 필요 없습니다.

원 프로젝트는 브라우저에서 Gradio를, 실시간 VC 창에서는 FreeSimpleGUI를 사용합니다. 이 포크는 두 가지 모두를 **SwiftUI 프런트엔드**로 대체하고, 하위 프로세스로 실행되는 **Python 백엔드**와 stdin/stdout 상의 JSON-RPC로 통신합니다.

## 특징

- **완전 오프라인** — 모든 ML 가중치가 번들 안에 있습니다. 자산 다운로드 단계나 HuggingFace 가져오기가 없습니다.
- **Apple Silicon 우선** — PyTorch MPS 백엔드 기본 지원. MPS가 처리하지 못하는 연산은 올바르게 CPU로 폴백합니다.
- **상시 리소스 모니터** — 툴바에 CPU / 통합 메모리 / MPS 사용률을 매초 갱신하여 표시.
- **정직한 진행 표시줄** — 작업별 %, 단계 라벨, ETA 표시. 취소 버튼은 실제로 중단 가능한 작업에만 노출됩니다.
- **하나의 앱에 모든 RVC 기능**:
  - 단일 파일 추론 및 배치 추론
  - UVR5 보컬/반주 분리(어느 HP/DeEcho/DeReverb 모델을 언제 선택할지에 대한 가이드 포함)
  - 선택적 자동 마무리 체인(보컬 추출 이후 2차 DeReverb)
  - 전체 학습 파이프라인: 전처리 → F0 / 특징 추출 → 학습 → 인덱스
  - 모델 관리: 비교, 융합, 추출(슬림화), 정보 편집
  - ONNX 내보내기
  - 실시간 보이스 체인저 — 장치 선택 및 파라미터 핫 리로드 지원
- **알아보기 쉬운 파일 배치** — 모든 사용자 파일은 `~/Documents/RVC-WebUI/` 아래에 모이며, 숨겨진 Application Support 폴더에 흩어지지 않습니다.
- **오디오 품질을 낮추지 않는 기본값** — 출력은 기본 FLAC(무손실); WAV / MP3 / M4A도 선택 가능.

## 시스템 요구 사항

| | 최소 | 권장 |
|---|---|---|
| macOS | 12.0 Monterey | 14.0 Sonoma 이상 |
| CPU | Apple Silicon (M1) | M2 Pro / M3 Pro 이상 |
| RAM | 8 GB | 16 GB 이상 (학습은 메모리를 많이 씁니다) |
| 디스크 | 8 GB 여유 공간 | 학습 시 20 GB 이상 |

Intel Mac은 **지원하지 않습니다** — 번들된 PyTorch는 ARM64 전용입니다.

## 설치

### 최종 사용자

1. 최신 [Release](https://github.com/RTCKPRO/RVC-WebUI-MacOS/releases)에서 `RVC-WebUI.app.zip`을 다운로드합니다.
2. 압축을 풀고 `RVC-WebUI.app`을 `/Applications`로 드래그합니다.
3. 더블클릭하여 실행합니다. 최초 실행 시 Gatekeeper가 확인을 요구할 수 있습니다 — 앱을 우클릭 → **열기** → 대화상자에서 **열기**.

첫 실행 시 앱은 입력, 출력, 모델, 로그용 하위 디렉터리와 함께 `~/Documents/RVC-WebUI/`를 생성합니다. 앱이 쓰는 유일한 위치입니다.

### 개발자 / 소스로부터 빌드

```bash
# 전제: Homebrew, Xcode CLT, Miniforge/conda
brew install xcodegen conda-pack

# 1. 클론
git clone https://github.com/RTCKPRO/RVC-WebUI-MacOS.git
cd RVC-WebUI-MacOS

# 2. conda 환경 생성 (Python 3.10 + PyTorch MPS + fairseq 등)
./setup_conda_env.sh
conda activate rvc

# 3. (선택) Python 백엔드 단독 스모크 테스트
python tools/test_rpc.py
# 예상: "ready" 알림 → initialize 응답 → 매초 resource_stats 알림

# 4. 전체 .app 번들 빌드
./build_app.sh
# 결과물: build/RVC-WebUI.app  (PyTorch와 모든 모델 포함 약 4 GB)
```

빌드 옵션:

- `--skip-conda` — 이전에 팩한 Python 환경(`build/python_env/`)을 재사용
- `--skip-xcode` — 이전에 빌드한 Swift 바이너리를 재사용
- `--skip-sign` — 코드 서명을 건너뜀 (로컬 개발에는 괜찮지만 배포에는 불가)

배포용 서명 빌드:

```bash
export CODE_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
./build_app.sh
xcrun notarytool submit build/RVC-WebUI.app --keychain-profile AC_PROFILE --wait
xcrun stapler staple build/RVC-WebUI.app
```

## 아키텍처

```
┌──────────────────────────────────────────────┐
│          SwiftUI .app (RVCApp)               │
│   NavigationSplitView + TabView              │
│   툴바: CPU / MEM / MPS 모니터               │
└───────────────────┬──────────────────────────┘
                    │ JSON-RPC 2.0 over stdio
                    │ (네트워크/소켓 미사용)
┌───────────────────▼──────────────────────────┐
│       Python 하위 프로세스 (rpc_server.py)    │
│   VC · UVR5 · 학습 · 실시간 · ONNX           │
│   psutil + torch.mps 로 리소스 샘플링        │
└──────────────────────────────────────────────┘
```

- 프런트엔드: `RVCApp/` — SwiftUI, `project.yml`에서 `xcodegen`으로 생성
- 브리지: `RVCApp/RVCApp/Bridge/PythonBridge.swift` — Python 하위 프로세스 실행, RPC 호출 디스패치, 진행/리소스 알림을 `@Published` 상태로 라우팅
- 백엔드: `rpc_server.py` + `rpc_training.py` — JSON-RPC 메서드가 `infer/modules/vc`, `infer/modules/uvr5` 및 학습 스크립트를 래핑; stdout은 즉각적인 첫 응답을 위해 라인 버퍼링
- 자산: `assets/hubert/`, `assets/rmvpe/`, `assets/pretrained_v2/`, `assets/uvr5_weights/` — 빌드 시 `.app/Contents/Resources/rvc_backend/assets/`로 복사
- Python 런타임: `conda-pack`으로 `build/python_env/`에 패키징, 이후 `.app/Contents/Resources/python/`에 포함

빌드 파이프라인과 아키텍처 세부 사항은 [`BUILD_NATIVE_APP.md`](../../BUILD_NATIVE_APP.md)를 참조하세요.

## 파일 배치

**번들 내부** (`RVC-WebUI.app/Contents/Resources/`) — 읽기 전용:

```
rvc_backend/    # 저장소에서 복사된 Python 코드 + 자산
python/         # 번들된 Python 3.10 런타임 (모든 의존성 포함)
```

**홈 디렉터리** (`~/Documents/RVC-WebUI/`) — 모든 사용자 데이터:

```
input/
  audio/          # 추론용 파일 배치
  training/       # 학습 데이터셋
output/
  inference/      # 단일 파일 변환 결과 (기본 FLAC)
  batch/          # 배치 변환 결과
  separation/     # UVR5의 vocals/ 와 accompaniment/
  onnx/           # ONNX 내보내기
models/           # 학습한 .pth 음성 모델
indices/          # FAISS .index 파일
logs/             # 학습 체크포인트 및 로그, 실험별 디렉터리 1개
configs/inuse/    # 런타임 설정
temp/             # 스크래치 공간, 시작 시 정리
```

## 문제 해결

**"RVC-WebUI.app이 손상되어 열 수 없습니다"** — Ad-hoc 서명 빌드는 새로 다운로드한 직후 Gatekeeper에 걸릴 수 있습니다. 해결:
```bash
xattr -cr /Applications/RVC-WebUI.app
```

**"No supported NVIDIA GPU found"** — 예상된 동작입니다. 앱은 MPS 위에서 동작하며, 이는 상위 코드 경로에서 출력되는 로그 라인이지 오류가 아닙니다.

**학습이 특징 추출에서 곧바로 실패** — 이 포크에서 수정되었습니다. 아주 오래된 체크아웃에서 빌드 중이라면 `infer/lib/torch_compat.py`가 존재하는지, 그리고 `extract_feature_print.py`, `infer/modules/vc/utils.py`, `infer/lib/rtrvc.py`에서 `fairseq`보다 먼저 import되는지 확인하세요. 이 shim은 PyTorch 2.6+의 `weights_only=True` 기본값을 비활성화하며, fairseq의 HuBERT 로더는 이 기본값에 걸립니다.

**학습 중 MPS 메모리 부족** — `batch_size_per_gpu`를 낮추고, 다른 앱을 닫고, 또는 `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`을 설정하세요 (시작 시 이미 설정되지만 `~/Documents/RVC-WebUI/logs/<exp>/train.log`에서 확인할 가치가 있습니다).

**첫 실행이 느림** — fairseq + torch의 콜드 임포트는 M1에서 약 3초, M3에서 약 2초입니다. `alive`가 도착할 때까지 스플래시에 "백엔드 대기 중"이 표시됩니다. 별도의 조작이 필요 없습니다.

## 개발

SwiftUI 프로젝트는 매 빌드마다 `RVCApp/project.yml`에서 xcodegen으로 재생성되므로 `RVCApp.xcodeproj`를 손으로 편집하지 마세요. Xcode에서 `RVCApp.xcodeproj`를 열고 Run을 누르면 됩니다 — 개발 모드에서 앱은 번들된 Python이 아닌 현재 활성 conda 환경으로 저장소의 `rpc_server.py`를 실행하므로 반복 속도가 훨씬 빠릅니다.

Python 측 변경:
- 소스는 저장소 루트에 있습니다 (`rpc_server.py`, `rpc_training.py`, `infer/`, `rvc/`, `configs/`, `i18n/`, `tools/`)
- `./build_app.sh --skip-conda --skip-xcode`는 Swift 바이너리 재빌드나 Python 재패키징 없이 기존 `.app`에 Python 백엔드만 재동기화합니다
- 이미 빌드된 `.app`에 대한 임시 반복에는 `rsync -a infer/ build/RVC-WebUI.app/Contents/Resources/rvc_backend/infer/`로 충분합니다

## 크레딧

- 상위 음성 변환 프레임워크: [fumiama/Retrieval-based-Voice-Conversion-WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI)
- 구성 요소: [ContentVec](https://github.com/auspicious3000/contentvec), [VITS](https://github.com/jaywalnut310/vits), [HIFIGAN](https://github.com/jik876/hifi-gan), [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui), [audio-slicer](https://github.com/openvpi/audio-slicer), [RMVPE](https://github.com/Dream-High/RMVPE) (사전 학습 모델은 [yxlllc](https://github.com/yxlllc/RMVPE)와 [RVC-Boss](https://github.com/RVC-Boss) 제작)
- 초기 macOS 포크: [Nevil Patel](https://github.com/NevilPatel01/RVC-WebUI-MacOS)
- 네이티브 `.app` 재구성: 이 저장소

## 라이선스

MIT. [LICENSE](../../LICENSE)를 참조하세요.
