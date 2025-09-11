# 食品体積推定のための深度推定モデル最適化戦略に関する技術的調査

## 現状の問題と実装詳細

### 実装済みの最適化手法とその結果

#### 1. 現在の実装アーキテクチャ
```
[入力画像] → [Depth Pro] → [深度マップ] → [平面推定] → [高さマップ] → [体積積分]
                ↓                           ↓
           [焦点距離推定]              [水平性事前分布]
```

#### 2. 実装した最適化手法

**A. 焦点距離の厳密化**
```python
# 35mm換算から横/縦を分離
fx = W * (f35 / 36)
fy = H * (f35 / 24)
# 体積計算で fx*fy を使用
a_pix = depth^2 / (fx * fy)
```

**B. 平面推定の堅牢化**
```python
def estimate_table_plane():
    # 1. リング領域の自動拡張（4%→15%）
    # 2. 深度勾配フィルタ（フラット領域のみ）
    # 3. 水平性事前分布（n·[0,0,1] ≥ cos(15°)）
    # 4. 複数距離閾値でRANSAC（2-10mm）
```

**C. 焦点距離の最適化**
```python
def refine_fpx_by_flatness():
    # EXIFがない場合、平面の水平性を最大化するfxを探索
    # スケール係数 [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
```

### 実験結果の詳細

#### テストケース: train_00000.jpg（ご飯）
| 手法 | 体積 | 平均高さ | 焦点距離 | 平面水平度 | 備考 |
|------|------|----------|----------|------------|------|
| UniDepth v2 | 126.9 mL | 約20mm | K_scale=5.0補正 | - | 期待値に近い |
| Depth Pro (初期) | 11,680 mL | 89.9mm | fx=531px | 24° | 約60倍過大 |
| Depth Pro (最適化後) | 6.4 mL | 0.0mm | fx=531px | 24.4° | 過小、高さ異常 |
| 期待値 | 100-300 mL | 10-30mm | - | ~0° | - |

### 根本的な問題分析

#### 1. モデル特性の不整合
- **Depth Pro**: 一般的なシーン（屋外、建築物、人物）で訓練
- **食品撮影の特徴**: 
  - 近距離撮影（20-50cm）
  - 下向き視点（60-90度）
  - 小さな高さ変化（1-10cm）
  - 反射・透明物（スープ、ソース）

#### 2. 平面推定の失敗パターン
```
観測された法線ベクトル: [0.014, 0.413, 0.910]
→ 24.4度の傾き（期待: <5度）

原因仮説:
- 深度の系統的バイアス（中心部が過小評価）
- エッジ効果（器の縁を平面として誤認）
- スケールの不確実性（metric depthの絶対値が不正確）
```

#### 3. 焦点距離推定の限界
- FOVヘッドが食品クローズアップで不安定
- 35mm換算EXIFの欠如（スマホ画像の多く）
- シーン依存の最適値（料理により20-50%変動）

## Fine-tuningの必要性評価

### Fine-tuningが必要な理由

1. **ドメインギャップ**: 一般シーン→食品特化
2. **スケール精度**: cmレベルの絶対精度が必要
3. **平面の安定性**: テーブル面の確実な検出
4. **エッジ処理**: 器の境界での深度不連続性

### Fine-tuningなしでの限界
- 焦点距離の手動調整では体積が10倍以上変動
- 平面推定が安定しない（24度の傾きなど）
- 後処理での補正に限界

## モデル比較と推奨

### 候補モデルの特性比較

| モデル | 長所 | 短所 | Fine-tuning難易度 | 推奨度 |
|--------|------|------|-------------------|--------|
| **Depth Pro** | ・絶対スケール<br>・高解像度<br>・FOV推定 | ・食品データなし<br>・計算量大<br>・コード非公開部分あり | 高（Appleの制約） | ★★☆ |
| **UniDepth v2** | ・オープンソース<br>・カメラ内参推定<br>・軽量 | ・K_scale必要<br>・解像度制限 | 中 | ★★★ |
| **DPT (MiDaS)** | ・豊富な派生版<br>・軽量<br>・実装簡単 | ・相対深度のみ<br>・スケール不明 | 低 | ★★☆ |
| **Depth Anything** | ・最新SOTA<br>・ゼロショット性能<br>・効率的 | ・新しい（エコシステム未成熟） | 中 | ★★★ |
| **Metric3D** | ・絶対深度<br>・カメラ内参不要<br>・堅牢 | ・精度がDepth Pro以下 | 中 | ★★☆ |

### 推奨: UniDepth v2 または Depth Anything

**理由:**
1. オープンソースで改変可能
2. 食品データでのFine-tuning実績あり（論文あり）
3. 推論速度と精度のバランス

## Fine-tuning戦略

### 1. データセット構築

#### A. 既存データセット
- **Nutrition5k**: 5,000枚の食事画像（深度なし、体積ラベルあり）
- **Food-101**: 101,000枚（深度なし、カテゴリのみ）
- **RGBD食品データ**: 限定的だが存在

#### B. 合成データ生成
```python
# Blender/Unity での食品3Dモデル配置
for food_model in food_3d_models:
    for angle in [45, 60, 75, 90]:  # 撮影角度
        for distance in [20, 30, 40, 50]:  # cm
            render_rgbd_pair()
```

#### C. 実測データ収集
```
必要機器:
- iPhone 12 Pro以上（LiDAR付き）
- Kinect Azure
- RealSense D435

収集手順:
1. 既知体積の食品を用意（50, 100, 200, 300mL）
2. 複数角度・距離から撮影
3. 深度とRGBのペアを記録
4. 平面フィッティングでGT作成
```

### 2. Fine-tuning実装

#### A. 損失関数の設計
```python
def food_depth_loss(pred, gt, mask):
    # 1. 深度の絶対誤差
    L_depth = F.l1_loss(pred[mask], gt[mask])
    
    # 2. 勾配の一致（エッジ保持）
    L_grad = gradient_loss(pred, gt, mask)
    
    # 3. 平面の水平性
    plane_n = fit_plane(pred, mask_ring)
    L_plane = 1 - abs(plane_n[2])  # n_z → 1
    
    # 4. 体積の一致
    vol_pred = compute_volume(pred, K, mask)
    L_volume = abs(vol_pred - vol_gt) / vol_gt
    
    return L_depth + 0.1*L_grad + 0.05*L_plane + 0.1*L_volume
```

#### B. データ拡張
```python
augmentations = {
    'geometric': RandomCrop, RandomRotation, RandomScale,
    'photometric': ColorJitter, GaussianBlur,
    'depth_specific': DepthNoise, DepthSmoothing,
    'food_specific': PlateRotation, FoodMaskErosion
}
```

### 3. 段階的学習戦略

```
Stage 1: 一般深度の維持（frozen backbone）
  └→ Stage 2: 食品領域の深度精度（ROI focus）
       └→ Stage 3: 平面推定の最適化（plane prior）
            └→ Stage 4: 体積損失での微調整（volume metric）
```

## 実装推奨事項

### 短期対策（Fine-tuningなし）
1. **複数モデルのアンサンブル**
   ```python
   depth_final = 0.5 * depth_pro + 0.3 * unidepth + 0.2 * dpt
   ```

2. **シーン別パラメータ辞書**
   ```python
   presets = {
       'rice': {'fx_scale': 1.2, 'plane_threshold': 10},
       'soup': {'fx_scale': 0.9, 'plane_threshold': 15},
   }
   ```

3. **信頼度ベースの重み付け**
   ```python
   if gradient_magnitude > threshold:
       confidence = 0  # エッジ部分を除外
   ```

### 中期対策（軽量Fine-tuning）
1. **LoRAアダプター**: 最終層のみ調整
2. **知識蒸留**: Depth Pro → 軽量モデル
3. **Test-time adaptation**: 推論時の自己教師あり学習

### 長期対策（本格的Fine-tuning）
1. **大規模データセット構築**: 10,000枚以上のRGBD食品画像
2. **マルチタスク学習**: 深度+セグメンテーション+体積
3. **物理ベースの制約**: 体積保存則、重力方向

## 質問事項

1. **Fine-tuningの必要性について**
   - 現在の後処理最適化で達成可能な精度の限界は？
   - Fine-tuningのROIを考慮した最小限のデータ量は？

2. **モデル選択について**
   - 食品ドメインでの各モデルの実績・論文はあるか？
   - ライセンス制約を考慮した商用利用可能なモデルは？

3. **データセット戦略について**
   - 合成データと実データの最適な比率は？
   - 既存RGBDデータセットで食品関連のものは？

4. **学習戦略について**
   - 平面推定を組み込んだEnd-to-End学習は有効か？
   - 体積を直接予測するヘッドを追加すべきか？

5. **実装の優先順位について**
   - Quick winとして最も効果的なアプローチは？
   - 精度vs計算コストの最適なトレードオフは？

これらについて、最新の研究動向と実装例を含めた解決策を教えてください。