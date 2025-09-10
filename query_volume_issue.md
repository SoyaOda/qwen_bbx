# UniDepth v2 体積推定の異常値問題に関する調査クエリ

## 現状の問題
UniDepth v2を使用した食品の体積推定で、非現実的に大きな値が出力されています：
- ご飯: 60.3L（正常値: 約200-300ml）
- スナップエンドウ: 25.9L（正常値: 約50-100ml）  
- チキン: 33.8L（正常値: 約150-250ml）

約100〜200倍の過大推定となっています。

## 現在の実装詳細

### 1. 画像前処理（src/unidepth_runner.py）
```python
# 画像読み込みと前処理
image = Image.open(image_path).convert("RGB")
rgb = transforms.ToTensor()(image).unsqueeze(0).to(self.device)

# ImageNet正規化を適用
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
rgb = normalize(rgb)

# カメラ内部パラメータ（仮定値）
intrinsics = torch.tensor([
    [1260.0, 0.0, 640.0],
    [0.0, 1260.0, 360.0],
    [0.0, 0.0, 1.0]
]).to(self.device)
```

### 2. UniDepth v2推論
```python
# モデル推論
predictions = self.model.infer(rgb, intrinsics)
depth = predictions["depth"].squeeze().cpu().numpy()  # メートル単位の深度マップ
```

### 3. 3D点群生成（src/plane_fit.py）
```python
def unproject_depth_to_xyz(depth, Kt):
    """深度マップから3D点群を生成"""
    H, W = depth.shape
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    depth_flat = depth.flatten()
    
    # ピクセル座標を正規化カメラ座標に変換
    uv1 = np.stack([xx_flat, yy_flat, np.ones_like(xx_flat)], axis=0)  # (3, N)
    xyz_cam = Kt @ uv1  # Kt = K^(-1)^T
    xyz_cam = xyz_cam * depth_flat  # Z値でスケーリング
    
    return xyz_cam.T.reshape(H, W, 3)
```

### 4. 体積計算（src/volume_estimator.py）
```python
def integrate_volume(height, a_pix, mask_bool, conf=None, use_conf_weight=False):
    """高さマップから体積を計算
    
    height: メートル単位の高さマップ
    a_pix: ピクセルあたりの実面積（m²）
    """
    if use_conf_weight and conf is not None:
        volume = np.sum(height[mask_bool] * a_pix[mask_bool] * conf[mask_bool])
    else:
        volume = np.sum(height[mask_bool] * a_pix[mask_bool])
    
    return volume

# ピクセル面積の近似計算
def calc_a_pix_approx(xyz, K):
    """a_pix(z) = z² / (fx * fy)"""
    fx = K[0, 0]
    fy = K[1, 1]
    z = xyz[:, :, 2]
    a_pix = (z * z) / (fx * fy)
    return a_pix
```

### 5. 平面フィッティング（RANSAC）
```python
def fit_plane_ransac(points_xyz, cand_mask, dist_th=0.006, max_iters=2000):
    """RANSACで平面をフィッティング
    
    dist_th: 0.006m（6mm）のインライア閾値
    """
    # 3点をランダムに選択して平面を定義
    # ax + by + cz + d = 0の形式で平面を表現
```

## 調査が必要な項目

### 1. **カメラ内部パラメータの妥当性**
- 現在の設定: fx=fy=1260, cx=640, cy=360（1280x720画像用）
- これらの値は適切か？実際のカメラパラメータとの乖離は？
- 焦点距離1260ピクセルは一般的な値か？

### 2. **深度値の単位とスケール**
- UniDepth v2の出力はメートル単位で正しいか？
- 実際の出力範囲（現在: 0.4-1.1m）は妥当か？
- スケールファクターの適用が必要か？

### 3. **ピクセル面積計算の妥当性**
- a_pix = z²/(fx·fy) の公式は正しいか？
- 単位の整合性は取れているか？（m² vs mm² vs cm²）
- 実際の計算例：z=0.5m, fx=fy=1260 → a_pix = 1.57×10⁻⁷ m²

### 4. **座標系と変換の確認**
- カメラ座標系からワールド座標系への変換は正しいか？
- Y軸の向き（上向き vs 下向き）の扱いは？
- 深度値での乗算 `xyz_cam = xyz_cam * depth_flat` は正しいか？

### 5. **UniDepth v2特有の問題**
- メトリック深度（metric depth）vs 相対深度（relative depth）
- UniDepth v2の公式実装での単位系
- 必要なポストプロセッシングや変換はあるか？

### 6. **体積計算の数値例**
仮定：
- 画像サイズ: 1280×720
- 深度: 0.5m
- マスクサイズ: 100×100ピクセル
- fx=fy=1260

計算：
- a_pix = (0.5)² / (1260×1260) = 1.57×10⁻⁷ m²
- 高さ: 0.05m（5cm）
- 体積 = 100×100 × 1.57×10⁻⁷ × 0.05 = 7.85×10⁻⁵ m³ = 0.0785L

**しかし実際は60Lなので、約760倍の誤差がある**

## 可能性のある原因

1. **単位の混同**
   - ピクセル単位とメートル単位の混在
   - ミリメートルとメートルの変換ミス

2. **カメラパラメータの誤り**
   - 焦点距離が実際より大きすぎる/小さすぎる
   - 画像サイズとの不整合

3. **深度スケールの問題**
   - UniDepth v2の出力に追加のスケーリングが必要
   - メトリック深度への変換が不完全

4. **ピクセル面積計算式の誤り**
   - z²の項が不要、またはz⁴が必要？
   - 追加の定数項が必要？

これらについて、UniDepth v2の公式実装やドキュメント、および一般的な3D復元の手法を参考にした解決策を教えてください。特に：
1. UniDepth v2の正確な出力単位とスケール
2. 適切なカメラ内部パラメータの設定方法
3. ピクセル面積から実面積への正しい変換式
4. 他のプロジェクトでの成功事例

これらについて、公式実装を参考にした解決策を教えてください。