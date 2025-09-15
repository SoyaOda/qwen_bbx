# Nutrition5k Depth Finetuning プロジェクト

## 📋 概要

Depth Anything V2 (DAV2) Metric-Indoor-Largeモデルを、Nutrition5kデータセット（食事トレイの深度推定）でファインチューニングするプロジェクトです。

## 🚨 重要な発見事項

### データリーク問題の発見と対策

#### 問題の発見
初期の実装で**99.4%という異常な性能向上**を観測。調査の結果、以下が判明：

1. **時系列的データリーク**: 同じ料理の連続撮影（数秒〜数分違い）がtrain/test/valに分散
2. **セッション内相関**: 5分以内の撮影1,000件以上が異なるセットに分割
3. **実質的なメモリゼーション**: モデルが汎化ではなく特定パターンを記憶

#### 対策実装
- **セッションベース分割**: 5分以内の撮影を同一セッションとしてグループ化
- **時系列考慮型分割**: セッション単位でtrain/val/testに分割
- **結果**: セット間の時系列的重複を完全に排除

## 📁 ファイル構成

```
Finetuning/
├── README.md                      # このファイル
├── README_FIXED.md               # 修正版の詳細ドキュメント
│
├── 📊 データ分析・検証スクリプト
│   ├── analyze_dish_temporal_correlation.py  # 時系列相関分析
│   ├── check_data_leak.py                   # データリーク検出
│   ├── investigate_anomaly.py               # 異常値の詳細調査
│   ├── quick_comparison.py                  # Pretrained vs Finetuned簡易比較
│   └── evaluate_pretrained_vs_finetuned.py  # 詳細な性能比較
│
├── 🏋️ トレーニングスクリプト
│   ├── train_dav2_metric.py                 # オリジナル版（データリークあり）
│   └── train_dav2_metric_fixed.py           # 修正版（データリーク対策済み）
│
├── 🔧 評価・テストスクリプト
│   ├── eval_depth_n5k.py                    # モデル評価
│   ├── test_dataloader.py                   # データローダーテスト
│   ├── test_model_loading.py                # モデル読み込みテスト
│   └── test_quick_training.py               # 簡易トレーニングテスト
│
├── 📂 データ管理
│   ├── create_proper_splits.py              # 時系列考慮型分割の生成
│   ├── validate_dataset.py                  # データセット検証
│   ├── verify_and_clean_data.py            # データクリーニング
│   ├── check_data_stats.py                 # データ統計確認
│   └── check_full_dataset.py               # 全データセット確認
│
├── 📦 モジュール
│   ├── datasets/
│   │   └── nutrition5k_depthonly.py        # データセットクラス
│   └── losses/
│       └── silog.py                        # Scale-Invariant Logarithmic Loss
│
└── 📁 データ分割ファイル
    ├── cleaned_splits_v2/                  # クリーンなID（オリジナル分割）
    │   ├── clean_train_ids.txt (2479)
    │   ├── clean_val_ids.txt (275)
    │   └── clean_test_ids.txt (507)
    └── temporal_aware_splits/              # 時系列考慮型分割（推奨）
        ├── temporal_train_ids.txt (2280)
        ├── temporal_val_ids.txt (428)
        └── temporal_test_ids.txt (557)
```

## 🔧 使用方法

### 環境準備
```bash
# Python 3.11環境の有効化
source venv_new/bin/activate

# 必要なパッケージのインストール
pip install torch torchvision transformers>=4.45.0 tqdm pillow numpy
```

### データセットの配置
```bash
# Nutrition5kデータセットを以下に配置
./nutrition5k/nutrition5k_dataset/
```

## 📊 実行コマンドと結果

### 1. データリーク検出
```bash
python Finetuning/check_data_leak.py
```
**結果**: ID自体の重複はなし（Train∩Test=0）

### 2. 時系列相関分析
```bash
python Finetuning/analyze_dish_temporal_correlation.py
```
**結果**: 
- Train-Test間で5分以内: 1,005件 ⚠️
- 連続撮影が異なるセットに: 100件以上
- 例: dish_1556575273(train) → dish_1556575327(test)、差54秒

### 3. 時系列考慮型分割の作成
```bash
python Finetuning/create_proper_splits.py
```
**結果**:
- 511セッション検出（平均6.4 dishes/session）
- Train: 2280 (357 sessions)
- Val: 428 (76 sessions) 
- Test: 557 (78 sessions)
- セット間の時系列重複: 0 ✅

### 4. オリジナル版トレーニング（データリークあり）
```bash
python -m Finetuning.train_dav2_metric \
  --n5k_root ./nutrition5k/nutrition5k_dataset \
  --epochs 5 \
  --batch_size 2 \
  --lr 1e-5
```
**結果**: AbsRel改善率 99.4% ⚠️（異常）

### 5. 修正版トレーニング（推奨）
```bash
venv_new/bin/python3.11 -m Finetuning.train_dav2_metric_fixed \
  --n5k_root ./nutrition5k/nutrition5k_dataset \
  --epochs 3 \
  --lr 5e-6 \
  --weight_decay 0.1 \
  --use_temporal_split True \
  --out_dir checkpoints/dav2_metric_n5k_fixed
```
**結果**:
- Initial: AbsRel=1.5757, RMSE=0.5945
- Final: AbsRel=0.0103, RMSE=0.0354
- 改善率: 99.3% ⚠️（依然として異常）

### 6. Pretrained vs Finetuned比較
```bash
venv_new/bin/python3.11 Finetuning/quick_comparison.py
```
**結果**:
- Pretrained: スケール比 2.7倍（GT平均0.37m → 予測1.0m）
- Finetuned: スケール比 1.0倍（完璧に一致）

### 7. 詳細な異常調査
```bash
venv_new/bin/python3.11 Finetuning/investigate_anomaly.py
```
**結果**:
- Pretrained: 相対誤差 1.72
- Finetuned: 相対誤差 0.008（ほぼ完璧）
- モデル重みの変化: 最小限（0.0002程度）

### 8. 評価スクリプト
```bash
python -m Finetuning.eval_depth_n5k \
  --n5k_root ./nutrition5k/nutrition5k_dataset \
  --ckpt_dir checkpoints/dav2_metric_n5k_fixed \
  --split test
```

## 📈 観察結果のまとめ

### 問題の本質
1. **データリーク対策後も99.3%改善**: 時系列分割でも解決せず
2. **スケール特化**: Indoor Metric（0.5-10m）→ 食事トレイ（0.3-0.4m）
3. **データセット固有バイアス**: Nutrition5k特有のパターンを過度に学習

### 学習曲線の異常
- Epoch 1で既に99.1%改善
- Loss減少は正常だが、評価指標が異常
- 初回エポックで「スケール調整」が完了している可能性

### 推定される原因
1. **狭いダイナミックレンジ**: 0.3-0.4mの狭い範囲に特化
2. **一貫した撮影条件**: 固定カメラ、一定の高さ、類似照明
3. **メトリックモデルの特性**: 既に室内シーンに適応済み

## 🎯 推奨事項

### 現状の評価
- ✅ **技術的成功**: データリーク排除、正常動作
- ⚠️ **実用性疑問**: Nutrition5k専用モデル化
- ❌ **汎化性能未検証**: クロスドメイン評価必須

### 今後の方針
1. **より控えめなファインチューニング**
   ```bash
   --epochs 1
   --lr 1e-6
   --weight_decay 0.5
   ```

2. **部分的な学習**
   - LoRAやアダプター層のみ
   - 最終層のみのファインチューニング

3. **データ拡張**
   - ランダムスケーリング
   - ノイズ追加
   - カラージッタリング

4. **クロスドメイン評価**
   - NYU Depth V2
   - KITTI
   - 独自収集データ

## 📝 結論

本プロジェクトは、深度推定モデルのファインチューニングにおける**データリークと過学習の危険性**を明確に示しました。時系列考慮型分割により明示的なデータリークは排除できましたが、**データセット固有の特性への過適応**という根本的な問題が残っています。

現在のモデルは**Nutrition5kデータセットでは優秀**ですが、**他のシーンでの汎用性は期待できません**。実用的なモデルを得るには、より慎重なファインチューニング戦略とクロスドメイン評価が必要です。

## 🔗 参考資料

- [Nutrition5k Paper](https://arxiv.org/abs/2103.03375)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [データリークに関する考察](./analyze_dish_temporal_correlation.py)