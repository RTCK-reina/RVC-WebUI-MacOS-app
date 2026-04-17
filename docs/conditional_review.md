# CONDITIONAL 項目 影響評価レポート

> 実装判断に必要な情報の収集結果。実装は行わない。

## C-1: Gradient Clipping の導入

**現状**: `total_grad_norm()` (rvc/layers/utils.py:68) で勾配ノルムを計測・ログしているが、
クリッピングは行っていない。`scaler.unscale_()` → `total_grad_norm()` → `scaler.step()` の順で実行。

**既存モデルとの互換性**: クリッピングは optimizer state に影響しないため、
既存チェックポイントからの resume に互換性の問題はない。新規学習のみ挙動が変わる。

**推奨値**: `max_norm=1.0`（音声合成タスクの標準値。VITS 原論文でも同値を使用）。

**リスク**:
- MPS では fp16 が無効化されている (`fp16_run = False`)。fp32 学習ではgrad explosion が起きにくいため、
  クリッピングの恩恵は限定的。
- 導入するなら TensorBoard のgrad_norm_g / grad_norm_d を数エポック観察し、
  実際にスパイクが発生しているか確認してからが安全。

**実装箇所**: train.py:608-611, 624-628, 696-704 の `scaler.unscale_()` と `scaler.step()` の間に
`torch.nn.utils.clip_grad_norm_()` を挿入。

## C-2: Learning Rate Warmup の追加

**現状**: `ExponentialLR` のみ使用 (train.py:364-369)。`warmup_epochs: 0` が全 config のデフォルト。
`init_lr_ratio: 1` も設定されているが使用箇所なし。

**既存モデルとの互換性**: LR スケジュール変更は既存チェックポイントからの resume 時に
学習曲線が不連続になる。新規学習のみ適用すべき。

**推奨**:
- `warmup_epochs: 5`（初期 5 エポックで lr を 0 → target lr まで線形上昇）
- `torch.optim.lr_scheduler.SequentialLR` で `LinearLR(start_factor=0.01, total_iters=5)` →
  `ExponentialLR` をチェーンする実装が最小侵襲。

**リスク**:
- バッチサイズ 4 + 小データセット（数十分）の場合、5 エポックでも数十 step しかないため
  warmup の効果が薄い。エポック数ではなく step 数ベースにする方が堅実。
- config.json の `warmup_epochs` フィールドは既に存在するが未使用。活用可能。

## C-3: バッチサイズの最適化

**��状**: config.json で `batch_size: 4`。Swift 側プリセットで quality=4, balanced=4, speed=8。
GPU メモリに応じた自動調整は行われて���ない。

**既存モデルとの互換性**: バッチサイズ変更は学習ダイナミクスを変える。
既存 checkpoint からの resume でも問題ないが、loss 曲線の傾きが変わる。

**推奨**:
- MPS 16GB: batch_size=4〜6（Apple Silicon のメモリ共有のため慎重に）
- accumulation_steps との組み合わせで実効バッチサイズを拡大する方がメモリ安全
- 自動調整するなら `torch.mps.recommended_max_memory()` を参照し、
  モデルサイズから逆算して決定

**リスク**:
- batch_size を大きくすると `if_cache_gpu=True` 時のメモリ消費が二重に増える
- DistributedBucketSampler のバケット境界 `[100, 200, ..., 900]` はbatch_size=4 前提で
  調整されている可能性あり

## C-4: FAISS インデックス種類の変更

**現状**: `rpc_training.py` でインデックス構築時に `IVFFlat` を使用（推定）。
`faiss.index_factory` の呼び出し箇所は rpc_train_index 内。

**��存モデルとの互換性**: インデックスはモデルとは独立。再構築すれば良い。
ただし `.index` ファイルを配布している場合は format 変更が breaking change。

**推奨**:
- 現行の `IVFFlat` → `IVF256,Flat` は小〜中規模データ（数万〜数十万ベクタ）に適切
- 大規模データ（100万+）なら `IVF1024,PQ32` で検索速度と精度のトレードオフ改善
- ただし RVC の推論時インデックス検索は 1 フレームずつで、レイテンシより精度が重要

**���スク**:
- PQ 量子化はわずかに検索精度が低下する（音質に影響する可能性）
- MPS 環境では faiss-gpu が使えないため CPU 検索のみ。インデックス種類変更の恩恵は限定的

## C-5: n_clusters の動的調整

**��状**: `n_clusters=10000` 固定 (rpc_training.py:648)。`big_npy.shape[0] > 2e5` のときのみ
KMeans を実行。

**既存モデルとの互換性**: インデックス品質のみに影響。モデル自体には影響なし。

**推奨式**: `n_clusters = max(256, min(10000, N // 20))`
- N < 5120 → n_clusters=256（最小値。これ以下は KMeans の意味が薄い）
- N >= 200000 → n_clusters=10000（現行と同じ上限）
- 中間: N // 20 で線形スケール

**根拠**:
- sklearn の MiniBatchKMeans は n_clusters が N に対して大きすぎると収束が遅い
- 小データセット（5分の音声 ≈ 数千ベクタ）で 10000 clusters は過剰
- `big_npy.shape[0] > 2e5` の閾値も `N // 20 > 10000` に置き換え可能

**リスク**:
- クラスタ数削減は検索精度低下の可能性があるが、小データでは元々ベクタ数が少ないため影響は軽微
- 閾値 `2e5` の変更は既存ワークフローとの挙動差異を生む
