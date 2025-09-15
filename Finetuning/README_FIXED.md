# 修正版: Nutrition5k Depth Finetuning (データリーク対策済み)

## 問題の発見と解決

### 発見された問題
元のデータ分割では、同じ料理の異なる撮影（数秒〜数分違い）がtrain/test/valに分散していました：
- 5分以内の撮影: Train-Test間で1,005件、Train-Val間で1,313件
- これにより99.4%という異常な性能向上（実質的なデータリーク）

### 解決策
時系列を考慮したセッションベースのデータ分割を実装：
- 5分以内の撮影を「セッション」としてグループ化
- セッション単位でtrain/val/testに分割
- 結果: セット間の時系列的重複を完全に排除

## 使用方法

### 1. 環境準備
```bash
# Python 3.11環境が必要
source venv_new/bin/activate  # または適切な仮想環境

# 必要なパッケージ
pip install torch torchvision transformers>=4.45.0 tqdm pillow
```

### 2. データセットの準備
```bash
# Nutrition5kデータセットが必要
# ./nutrition5k/nutrition5k_dataset/ に配置
```

### 3. 修正版でのトレーニング
```bash
# 時系列考慮型分割を使用した学習（推奨）
python -m Finetuning.train_dav2_metric_fixed \
  --n5k_root ./nutrition5k/nutrition5k_dataset \
  --epochs 3 \
  --lr 5e-6 \
  --weight_decay 0.1 \
  --use_temporal_split True \
  --out_dir checkpoints/dav2_metric_n5k_fixed
```

### 4. 評価
```bash
# 修正版での評価
python -m Finetuning.eval_depth_n5k \
  --n5k_root ./nutrition5k/nutrition5k_dataset \
  --ckpt_dir checkpoints/dav2_metric_n5k_fixed \
  --split test
```

## 主な変更点

### データ分割の改善
- `temporal_aware_splits/`: セッションベースの新しい分割ファイル
- `load_split_ids()`: `use_temporal=True`で新分割を使用

### トレーニングパラメータの調整
- エポック数: 5 → 3（過学習防止）
- 学習率: 1e-5 → 5e-6（安定性向上）
- Weight decay: 0.01 → 0.1（正則化強化）
- Early stopping: 2エポックの忍耐パラメータ

### 検証の強化
- 初期性能の記録
- 改善率の監視（>50%で警告）
- データリークチェックの自動実行

## ファイル構成

```
Finetuning/
├── train_dav2_metric_fixed.py    # 修正版トレーニングスクリプト
├── temporal_aware_splits/         # 新しいデータ分割
│   ├── temporal_train_ids.txt    # 2280 samples (357 sessions)
│   ├── temporal_val_ids.txt      # 428 samples (76 sessions)
│   └── temporal_test_ids.txt     # 557 samples (78 sessions)
├── analyze_dish_temporal_correlation.py  # 時系列分析
├── create_proper_splits.py        # 新分割の生成
└── check_data_leak.py            # データリークチェック
```

## 期待される結果

修正版では：
- **改善率**: 10-30%程度（現実的な範囲）
- **汎化性能**: 他のデータセットでも動作
- **信頼性**: データリークなしの正当な評価

## 注意事項

1. **既存のチェックポイント**: 古いチェックポイントは信頼できません
2. **クロスドメイン評価**: NYU Depth V2等での追加評価を推奨
3. **データ拡張**: さらなる汎化のためにaugmentationの追加を検討

## トラブルシューティング

### メモリ不足
```bash
# バッチサイズを減らす
--batch_size 1
```

### 学習が不安定
```bash
# 学習率をさらに下げる
--lr 1e-6
```

### データが見つからない
```bash
# データセットパスを確認
ls ./nutrition5k/nutrition5k_dataset/imagery/realsense_overhead/
```