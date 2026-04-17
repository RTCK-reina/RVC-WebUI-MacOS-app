# トレーニング運用ガイド

> コード変更なしで即対応可能な運用ルールとチェックリスト。

## Ops-1: プリセット推奨値

### 現行プリセット (TrainingView.swift)

| パラメータ | 高品質 | バランス | 高速 |
|-----------|--------|---------|------|
| f0_method | rmvpe | rmvpe | fcpe/pm |
| 並列プロセス | CPU-2 | CPU-1 | CPU全数 |
| save_epoch | 5 | 5 | 10 |
| total_epoch | ≥300 | 200-400 | ≤180 |
| batch_size (GPU) | 4 | 4 | 8 |
| batch_size (CPU) | 2 | 2 | 3 |
| if_save_latest | false | true | true |
| if_cache_gpu | true | false | true |
| if_save_every_weights | true | true | false |

### 推奨調整

- **高品質プリセットの total_epoch**: データ量に応じて調整。
  - 10分未満の音声: 300-500 epoch
  - 10-30分: 200-300 epoch
  - 30分以上: 150-200 epoch
  - 過学習の兆候（loss が再上昇）が見えたら打ち切る

- **高速プリセットの batch_size=8**: MPS 16GB 環境で if_cache_gpu=true と併用すると
  メモリ圧迫の可能性あり。OOM が出たら batch_size=4 に下げる。

## Ops-2: save_epoch の設定指針

### デフォルト値

| プリセット | save_epoch |
|-----------|------------|
| 高品質 | 5 |
| バランス | 5 |
| 高速 | 10 |

### 条件分岐の目安

- **total_epoch ≤ 50**: save_epoch = 2-3（短い学習では頻繁に保存）
- **50 < total_epoch ≤ 200**: save_epoch = 5（標準）
- **total_epoch > 200**: save_epoch = 10（ディスク節約）

### チェックリスト

- [ ] if_save_every_weights=true の場合、各エポックで G_xxx.pth + D_xxx.pth が保存される。
      500 epoch × save_epoch=5 → 100 チェックポイント ≈ 60GB。ディスク空き容量を確認。
- [ ] if_save_latest=true にすると最新の G/D のみ保持。ディスク節約だがロールバック不可。
- [ ] 最終的に使うモデルは `logs/<exp>/` 配下の .pth。不要な中間チェックポイントは手動削除。

## Ops-3: if_cache_gpu の使用判断

### 概要

`if_cache_gpu=true` にすると、データセット全体を GPU メモリにキャッシュする。
初回エポックでキャッシュし、2 エポック目以降はディスク I/O がゼロになる。

### 判断基準

| 条件 | 推奨値 | 理由 |
|------|--------|------|
| Apple Silicon 8GB | **false** | メモリ不足で OOM リスク高 |
| Apple Silicon 16GB + batch_size ≤ 4 | **true** | 16GB の統合メモリなら小〜中規模データで余裕あり |
| Apple Silicon 16GB + batch_size ≥ 8 | **false** | バッチサイズとの併用でメモリ圧迫 |
| Apple Silicon 32GB+ | **true** | 大半のデータセットでキャッシュ可能 |
| データセット > 1時間 | **false** | キャッシュサイズが巨大（数GB〜）になる |
| データセット ≤ 15分 | **true** | キャッシュサイズが数百MB 程度で収まる |

### チェックリスト

- [ ] アクティビティモニタの「メモリ」タブでメモリプレッシャーを監視
- [ ] 学習開始直後に OOM や swap 増大が見えたら即停止し false に切り替え
- [ ] ステータスバーの GPU メモリ表示 (XXXMB) が急増していないか確認
- [ ] CPU 学習（GPU 非使用）の場合は if_cache_gpu の設定は無視される
