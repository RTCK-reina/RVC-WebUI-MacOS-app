<div align="center">

# RVC-WebUI-MacOS

**Retrieval-based Voice Conversion を macOS ネイティブ `.app` として再構築したもの。**
SwiftUI フロントエンド + バンドル済み Python バックエンド。ブラウザ不要・ネットワーク不要・pip install 不要。

[![macOS](https://img.shields.io/badge/macOS-12.0%2B-black?style=for-the-badge&logo=apple)](https://www.apple.com/macos/)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MPS-0071c5?style=for-the-badge)](https://developer.apple.com/metal/pytorch/)
[![Licence](https://img.shields.io/github/license/RTCKPRO/RVC-WebUI-MacOS?style=for-the-badge)](../../LICENSE)

[**English**](../../README.md) · [**日本語**](./README.ja.md) · [**中文简体**](../cn/README.cn.md) · [**한국어**](../kr/README.ko.md) · [**Français**](../fr/README.fr.md) · [**Português**](../pt/README.pt.md) · [**Türkçe**](../tr/README.tr.md)

</div>

---

## これは何か

RVC-WebUI-MacOS は、[Retrieval-based Voice Conversion WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI) を Apple Silicon 向けの **単一スタンドアロン `.app`** として再パッケージしたものです。PyTorch、fairseq、すべての事前学習モデル（HuBERT、RMVPE、UVR5、pretrained_v2）はすべてバンドル内に同梱されています。ダウンロード後は、ダブルクリックするだけで起動できます — conda も pip も Homebrew も localhost URL も、ダウンロード後のインターネット接続も不要です。

本家プロジェクトはブラウザ内で Gradio を、リアルタイム VC ウィンドウに FreeSimpleGUI を使います。本フォークはその両方を **SwiftUI フロントエンド** に置き換え、**サブプロセスとして起動した Python バックエンド** と stdin/stdout 上の JSON-RPC でやり取りします。

## 特徴

- **完全オフライン** — すべての ML 重みがバンドル内にあります。アセットのダウンロード手順や HuggingFace 取得は一切ありません。
- **Apple Silicon ファースト** — PyTorch MPS バックエンドを標準採用。MPS が未対応の演算は正しく CPU にフォールバックします。
- **常時表示のリソースモニター** — ツールバーに CPU / ユニファイドメモリ / MPS 使用率を 1 秒毎に表示。
- **正直な進捗バー** — タスクごとの %、フェーズラベル、ETA を表示。キャンセルボタンは実際に中断可能な処理にのみ出します。
- **RVC の全機能を 1 つのアプリに集約**:
  - 単一ファイル推論・バッチ推論
  - UVR5 ボーカル/伴奏分離（どの HP/DeEcho/DeReverb モデルをいつ選ぶべきかのガイド付き）
  - オプションの自動仕上げチェーン（抽出後の 2nd パス DeReverb）
  - 学習パイプライン一式: 前処理 → F0 / 特徴量抽出 → 学習 → インデックス
  - モデル管理: 比較・融合・抽出（軽量化）・情報編集
  - ONNX 書き出し
  - リアルタイムボイスチェンジャー（デバイス選択 + パラメータのホットリロード）
- **分かりやすいファイル配置** — ユーザーファイルはすべて `~/Documents/RVC-WebUI/` 配下に集約。隠し Application Support フォルダに散らばることはありません。
- **音質を劣化させないデフォルト** — 出力はデフォルトで FLAC（可逆圧縮）。WAV / MP3 / M4A も選択可能。

## 動作環境

| | 最低要件 | 推奨 |
|---|---|---|
| macOS | 12.0 Monterey | 14.0 Sonoma 以降 |
| CPU | Apple Silicon (M1) | M2 Pro / M3 Pro 以上 |
| RAM | 8 GB | 16 GB 以上（学習時はメモリを多く消費します） |
| ディスク | 8 GB の空き | 学習する場合 20 GB 以上 |

Intel Mac は **非対応** です — バンドル済み PyTorch は ARM64 専用です。

## インストール

### エンドユーザー向け

1. 最新の [Release](https://github.com/RTCKPRO/RVC-WebUI-MacOS/releases) から `RVC-WebUI.app.zip` をダウンロード。
2. 解凍し、`RVC-WebUI.app` を `/Applications` にドラッグ。
3. ダブルクリックで起動。初回起動時に Gatekeeper が確認を求めてきたら、アプリを右クリック → **開く** → ダイアログで再度 **開く**。

初回起動時にアプリは `~/Documents/RVC-WebUI/` と、入出力・モデル・ログ用のサブディレクトリを作成します。書き込み先はこの場所だけです。

### 開発者向け / ソースからビルド

```bash
# 前提: Homebrew、Xcode CLT、Miniforge/conda
brew install xcodegen
conda install -n base -c conda-forge conda-pack

# 1. クローン
git clone https://github.com/RTCKPRO/RVC-WebUI-MacOS.git
cd RVC-WebUI-MacOS

# 2. conda 環境の作成（Python 3.10 + PyTorch MPS + fairseq 等）
./setup_conda_env.sh
conda activate rvc

# 3. （任意）Python バックエンド単体のスモークテスト
python tools/test_rpc.py
# 期待値: "ready" 通知 → initialize 応答 → 1 秒毎に resource_stats 通知

# 4. HuggingFace からモデルアセットを取得（hubert / rmvpe / pretrained_v2 / uvr5_weights、約 2 GB）
./tools/download_assets.sh --all

# 5. フルの .app バンドルをビルド
./build_app.sh
# 生成物: build/RVC-WebUI.app  （PyTorch と全モデルを含めて約 4 GB）
```

ビルドオプション:

- `--skip-conda` — 既存の Python env パック（`build/python_env/`）を再利用
- `--skip-xcode` — 既存の Swift ビルド成果物を再利用
- `--skip-sign` — コード署名をスキップ（ローカル開発用、配布には不可）

配布用署名ビルド:

```bash
export CODE_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
./build_app.sh
xcrun notarytool submit build/RVC-WebUI.app --keychain-profile AC_PROFILE --wait
xcrun stapler staple build/RVC-WebUI.app
```

## アーキテクチャ

```
┌──────────────────────────────────────────────┐
│          SwiftUI .app (RVCApp)               │
│   NavigationSplitView + TabView              │
│   ツールバー: CPU / MEM / MPS モニター       │
└───────────────────┬──────────────────────────┘
                    │ JSON-RPC 2.0 over stdio
                    │ （ネットワーク/ソケット不使用）
┌───────────────────▼──────────────────────────┐
│        Python サブプロセス (rpc_server.py)    │
│   VC · UVR5 · 学習 · リアルタイム · ONNX     │
│   psutil + torch.mps でリソース集計          │
└──────────────────────────────────────────────┘
```

- フロントエンド: `RVCApp/` — SwiftUI、`project.yml` から `xcodegen` で生成
- ブリッジ: `RVCApp/RVCApp/Bridge/PythonBridge.swift` — Python サブプロセス起動、RPC ディスパッチ、進捗/リソース通知を `@Published` 状態にルーティング
- バックエンド: `rpc_server.py` + `rpc_training.py` — JSON-RPC メソッドが `infer/modules/vc`、`infer/modules/uvr5`、学習スクリプトをラップ。stdout は行バッファリング済みで初回応答を早める
- アセット: `assets/hubert/`、`assets/rmvpe/`、`assets/pretrained_v2/`、`assets/uvr5_weights/` — ビルド時に `.app/Contents/Resources/rvc_backend/assets/` へコピー
- Python ランタイム: `conda-pack` で `build/python_env/` を生成、`.app/Contents/Resources/python/` に埋め込み

ビルドパイプラインとアーキテクチャの詳細は [`BUILD_NATIVE_APP.md`](../../BUILD_NATIVE_APP.md) を参照してください。

## ファイル配置

**バンドル内** (`RVC-WebUI.app/Contents/Resources/`) — 読み取り専用:

```
rvc_backend/    # リポジトリからコピーされた Python コード + アセット
python/         # バンドル済み Python 3.10 ランタイム（全依存含む）
```

**ホームディレクトリ内** (`~/Documents/RVC-WebUI/`) — すべてのユーザーデータ:

```
input/
  audio/          # 推論対象の音声ファイルを置く場所
  training/       # 学習用データセット
output/
  inference/      # 単一推論結果（デフォルト FLAC）
  batch/          # バッチ変換結果
  separation/     # UVR5 の vocals/ と accompaniment/
  onnx/           # ONNX 書き出し
models/           # 学習済み .pth 声モデル
indices/          # FAISS .index ファイル
logs/             # 学習チェックポイントとログ（実験ごとに 1 ディレクトリ）
configs/inuse/    # ランタイム設定
temp/             # 一時領域（起動時に削除）
```

## トラブルシューティング

**「RVC-WebUI.app は壊れているため開けません」** — アドホック署名ビルドはダウンロード直後に Gatekeeper に引っかかることがあります。対処:
```bash
xattr -cr /Applications/RVC-WebUI.app
```

**「No supported NVIDIA GPU found」** — これは想定通りです。本アプリは MPS 上で動作しており、上流のコードパス由来のログ行で、エラーではありません。

**学習が特徴量抽出で即失敗する** — 本フォークでは修正済みです。かなり古いチェックアウトからビルドしている場合は、`infer/lib/torch_compat.py` が存在し、`extract_feature_print.py`、`infer/modules/vc/utils.py`、`infer/lib/rtrvc.py` で `fairseq` より前に import されていることを確認してください。このシムは PyTorch 2.6+ の `weights_only=True` デフォルトを無効化するもので、fairseq の HuBERT ローダはこれに引っかかります。

**学習中の MPS メモリ不足** — `batch_size_per_gpu` を下げる、他のアプリを閉じる、あるいは `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` を設定してください（起動時にすでに設定済みですが、`~/Documents/RVC-WebUI/logs/<exp>/train.log` で確認する価値はあります）。

**初回起動が遅い** — fairseq + torch のコールドインポートは M1 で約 3 秒、M3 で約 2 秒です。`alive` が届くまでスプラッシュに「バックエンド待機中」と表示されます。操作不要です。

## 開発

SwiftUI プロジェクトは毎回 `RVCApp/project.yml` から xcodegen で再生成されるので、`RVCApp.xcodeproj` を手で編集しないでください。`RVCApp.xcodeproj` を Xcode で開いて Run するだけ — 開発モードではバンドルされた Python ではなくアクティブな conda 環境からリポジトリ直下の `rpc_server.py` を起動するため、反復が高速です。

Python 側の変更:
- ソースはリポジトリルート（`rpc_server.py`、`rpc_training.py`、`infer/`、`rvc/`、`configs/`、`i18n/`、`tools/`）
- `./build_app.sh --skip-conda --skip-xcode` で既存 `.app` に Python バックエンドだけを再同期できます（Swift バイナリや Python の再パッキングはしない）
- ビルド済み `.app` に対するアドホック反復なら `rsync -a infer/ build/RVC-WebUI.app/Contents/Resources/rvc_backend/infer/` で十分です

## クレジット

- 上流の音声変換フレームワーク: [fumiama/Retrieval-based-Voice-Conversion-WebUI](https://github.com/fumiama/Retrieval-based-Voice-Conversion-WebUI)
- 構成要素: [ContentVec](https://github.com/auspicious3000/contentvec)、[VITS](https://github.com/jaywalnut310/vits)、[HIFIGAN](https://github.com/jik876/hifi-gan)、[Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui)、[audio-slicer](https://github.com/openvpi/audio-slicer)、[RMVPE](https://github.com/Dream-High/RMVPE)（事前学習モデルは [yxlllc](https://github.com/yxlllc/RMVPE) と [RVC-Boss](https://github.com/RVC-Boss) による）
- 初期 macOS フォーク: [Nevil Patel](https://github.com/NevilPatel01/RVC-WebUI-MacOS)
- ネイティブ `.app` 再構築: 本リポジトリ

## ライセンス

MIT。[LICENSE](../../LICENSE) を参照してください。
