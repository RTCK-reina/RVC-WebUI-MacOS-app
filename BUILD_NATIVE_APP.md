# RVC-WebUI ネイティブ .app ビルド手順

## 概要

SwiftUI フロントエンド + Python バックエンド（JSON-RPC over stdio）で動作する単一 `.app`。
Web 依存なし、ネットワーク通信なし、全リソースを `.app` 内にバンドル。

## 構成

- `rpc_server.py` — Python JSON-RPC サーバー（stdin/stdout 通信）
- `configs/config.py` — `base_dir`/`user_dir` 対応に改修済み
- `RVCApp/` — SwiftUI フロントエンド（xcodegen で生成）
- `requirements/app.txt` — .app 用 Python 依存（gradio 等除外、psutil 追加）
- `build_app.sh` — ビルドスクリプト
- `setup_conda_env.sh` — 開発用 conda 環境セットアップ

## ユーザーデータディレクトリ

全ユーザーファイルは `~/Documents/RVC-WebUI/` 配下:
```
input/audio/      # 入力音声
input/training/   # 学習データ
output/inference/ # 単一推論結果
output/batch/     # バッチ結果
output/separation/vocals, accompaniment/
output/onnx/
models/           # ユーザー声モデル (.pth)
indices/          # FAISS インデックス
logs/             # 学習ログ
configs/inuse/    # ランタイム設定
temp/             # 一時ファイル
```

## 開発時のクイックスタート

```bash
# 1. conda 環境セットアップ
./setup_conda_env.sh
conda activate rvc

# 2. バックエンド単体動作確認
python tools/test_rpc.py
# -> "ready" 通知 → initialize 応答 → resource_stats 通知が複数件受信される

# 3. Xcode プロジェクト生成
brew install xcodegen  # 初回のみ
cd RVCApp && xcodegen
open RVCApp.xcodeproj

# 4. Xcode で Run すると開発モードで .app が起動し、
#    自動的にリポジトリ直下の rpc_server.py を subprocess 起動します。
```

## .app のビルド

```bash
# まず conda 環境が整っている前提。
./build_app.sh

# 結果:
#   build/RVC-WebUI.app
```

オプション:
- `--skip-conda`: conda-pack をスキップ（env 再利用）
- `--skip-xcode`: Xcode ビルドをスキップ
- `--skip-sign`: コード署名をスキップ

## コード署名（配布用）

```bash
export CODE_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
./build_app.sh

# 公証
xcrun notarytool submit build/RVC-WebUI.app \
  --keychain-profile "AC_PROFILE" --wait
xcrun stapler staple build/RVC-WebUI.app
```

## 実装完了機能（Phase 1–9）

- [x] Phase 1: Python RPC サーバー `rpc_server.py` + `configs/config.py` 改修
- [x] Phase 2: `PythonBridge.swift` + Xcode プロジェクト骨格（xcodegen）
- [x] Phase 3: リソースモニター（Python psutil/mps 収集 + Swift StatusBarView）
- [x] Phase 4: 単一推論画面 + 進捗バー
- [x] Phase 5: バッチ推論 + UVR5 分離画面
- [x] Phase 6: トレーニング画面（preprocess → F0/feature → train → index、ワンクリック学習付き、ログテール＆エポック推定）
- [x] Phase 7: モデル管理（比較・融合・情報・抽出・ONNX 出力）
- [x] Phase 8: リアルタイム VC（sounddevice stream + パラメータホットリロード、入出力デバイス選択）
- [x] Phase 9: ビルドスクリプト + conda-pack + パッケージング

## 今後の作業（Phase 10）

- [ ] Developer ID での本番コード署名
- [ ] 公証ワークフロー（notarytool + stapler）
- [ ] 統合テストスイート
- [ ] 配布用 DMG 生成

詳細は `.claude/plans/wobbly-mapping-starfish.md` を参照。
