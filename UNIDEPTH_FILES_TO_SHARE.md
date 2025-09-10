# UniDepth v2 実装 - 共有ファイル一覧

## 1. コア実装ファイル

### メインモジュール
- `src/unidepth_runner.py` - UniDepth v2推論エンジン（カメラパラメータスケーリング機能付き）
- `src/plane_fit.py` - RANSAC平面フィッティング
- `src/volume_estimator.py` - 体積推定モジュール
- `src/vis_depth.py` - 深度マップ可視化

### 実行スクリプト
- `src/run_unidepth.py` - メインパイプライン（Qwen→SAM2→UniDepth→体積）

## 2. 設定ファイル
- `config.yaml` - 全体設定（特に`unidepth`セクションの`K_scale_factor: 6.0`が重要）

## 3. ドキュメント

### 必須ドキュメント
- `README_UNIDEPTH.md` - UniDepth v2統合の完全ガイド
- `SOLUTION_VOLUME_ISSUE.md` - 体積推定問題の分析と解決策
- `md_files/unidepth_spec.md` - 元の仕様書

### 参考ドキュメント
- `README.md` - プロジェクト全体の概要
- `CLAUDE.md` - AI向けの作業指示

## 4. テスト・検証スクリプト（オプション）
- `test_normalization_effect.py` - ImageNet正規化の影響検証
- `test_fixed_K.py` - カメラパラメータ調整テスト
- `debug_volume.py` - 体積計算のデバッグ

## 5. 重要な設定値

### config.yamlの主要部分
```yaml
unidepth:
  model_repo: "lpiccinelli/unidepth-v2-vitl14"
  device: "cuda"
  K_scale_factor: 6.0  # 最重要：体積調整用スケーリング係数

plane:
  ring_margin_px: 40
  ransac_threshold_m: 0.006
  
volume:
  clip_negative_height: true
```

## 6. 依存関係情報

### requirements（主要部分）
```
unidepth  # GitHub: lpiccinelli-eth/UniDepth
torch>=2.0
torchvision
numpy
opencv-python
Pillow
```

## 7. 共有時の重要ポイント

### 必ず伝えるべきこと
1. **K_scale_factor = 6.0が必須** - UniDepthの推定カメラパラメータが小さすぎるため
2. **ImageNet正規化は不要** - 体積のオーダーにほとんど影響しない（0.8倍程度）
3. **逆投影は正しく実装済み** - K^{-1}を使用、K^{-T}ではない
4. **Python 3.11.13環境** - venv/bin/python3.11を使用

### 体積が異常な場合の調整方法
- 大きすぎる（数十L）→ K_scale_factorを8-10に増やす
- 小さすぎる（数mL）→ K_scale_factorを3-4に減らす
- 目安：食品一人前は100-500mL

## 共有用アーカイブ作成コマンド

```bash
# 必須ファイルのみ
tar -czf unidepth_implementation.tar.gz \
  src/unidepth_runner.py \
  src/plane_fit.py \
  src/volume_estimator.py \
  src/vis_depth.py \
  src/run_unidepth.py \
  config.yaml \
  README_UNIDEPTH.md \
  SOLUTION_VOLUME_ISSUE.md \
  md_files/unidepth_spec.md

# 全関連ファイル
tar -czf unidepth_complete.tar.gz \
  src/*.py \
  config.yaml \
  *.md \
  md_files/unidepth_spec.md \
  test_*.py \
  debug_volume.py
```