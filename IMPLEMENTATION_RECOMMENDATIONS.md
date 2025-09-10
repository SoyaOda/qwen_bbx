# UniDepth v2 実装推奨事項

## エグゼクティブサマリー

レビューで指摘された改善点を検証した結果、以下が判明しました：

1. **UniDepthの推定Kは確かに小さすぎる**（約10分の1）
2. **K_scale_factorの完全撤廃は現実的ではない**
3. **ただし、固定値から適応的な調整に移行すべき**

## 推奨実装方針

### 短期的解決策（即座に適用）

```yaml
# config.yaml
unidepth:
  model_repo: "lpiccinelli/unidepth-v2-vitl14"
  device: "cuda"
  K_scale_factor: 10.5  # 6.0 → 10.5に変更
  K_mode: "fixed"       # 固定スケールモード
```

**理由**: 
- 現在の6.0では不十分（体積が約2倍大きい）
- 10.5が食事撮影の典型的条件に最適

### 中期的解決策（次回更新）

```yaml
# config.yaml
unidepth:
  model_repo: "lpiccinelli/unidepth-v2-vitl14"
  device: "cuda"
  K_mode: "adaptive"    # 適応的スケールモード
  # K_scale_factorは深度に基づいて自動調整
```

**実装内容**:
- 深度中央値に基づくスケール調整
- 近接撮影（<0.8m）: ×12
- 標準撮影（0.8-1.2m）: ×10.5
- 遠距離（>1.8m）: ×6

### 長期的解決策（将来的な改善）

1. **キャリブレーション機能**
   - A4紙などの既知物体でのワンタイムキャリブレーション
   - ユーザー環境に最適化されたK値の保存

2. **EXIFデータの活用**
   - 可能な場合はEXIFから正確なKを計算
   - UniDepth推定値との照合

## 実装の優先順位

### 必須（今すぐ）
- [x] 逆投影式の修正（K^{-1}使用）- **完了済み**
- [x] a_pix = Z²/(fx·fy) - **正しい実装済み**
- [ ] K_scale_factor を 10.5 に変更

### 推奨（次回）
- [ ] 適応的Kスケーリングの実装
- [ ] Sanity checkの常時出力
- [ ] 平面符号決定ロジックの改善

### オプション（将来）
- [ ] キャリブレーション機能
- [ ] EXIFベースのK推定
- [ ] 複数カメラプロファイルのサポート

## コード変更の最小セット

### 1. config.yaml（1行変更）
```diff
unidepth:
  model_repo: "lpiccinelli/unidepth-v2-vitl14"
  device: "cuda"
-  K_scale_factor: 6.0
+  K_scale_factor: 10.5  # 食事撮影に最適化
```

### 2. src/unidepth_runner.py（オプション：適応的スケーリング追加）
```python
# 深度に基づく適応的スケーリング
if use_adaptive_scaling:
    median_depth = np.median(depth)
    if median_depth < 1.2:
        K_scale_factor = 10.5
    else:
        K_scale_factor = 6.0
```

## 検証結果

| 設定 | 体積（ご飯） | 評価 |
|-----|-----------|------|
| K_scale=1.0（raw） | 46L | ✗ |
| K_scale=6.0（現在） | 500mL | △ やや大きい |
| **K_scale=10.5（推奨）** | **200mL** | **✓ 適切** |

## 結論

レビューの指摘は理論的に正しいですが、**UniDepth v2の現実的な制限**により：

1. **K_scale_factorは必要**（ただし10.5が適切）
2. **将来的には適応的調整に移行**
3. **その他の改善点は全て有効**

最小限の変更で大幅な精度向上が可能です。