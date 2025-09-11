# Depth Pro体積推定最適化のためのクエリプロンプト

## 背景
QwenVL→SAM2.1→深度推定モデルのパイプラインで料理の体積推定を行っています。UniDepth v2からApple Depth Proに移行しましたが、体積推定値が大きすぎる問題が発生しています。

## 現在の実装

### 1. Depth Proモデルの初期化と推論
```python
# モデル初期化
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, DepthProConfig

config = DepthProConfig(
    patch_encoder_preset="dinov2l16_384",
    image_encoder_preset="dinov2l16_384", 
    decoder_features=256,
    checkpoint_uri="checkpoints/depth_pro.pt",
    use_fov_head=True,
    fov_encoder_preset="dinov2l16_384"
)

model, transform = depth_pro.create_model_and_transforms(
    config=config,
    device=torch.device("cuda")
)

# 推論実行
image_np, _, f_px = depth_pro.load_rgb(image_path)  # EXIFから焦点距離取得
image_tensor = transform(image_np)
prediction = model.infer(image_tensor, f_px=f_px)

depth = prediction["depth"]  # 深度マップ [m]
f_px_val = prediction["focallength_px"]  # 推定焦点距離 [pixels]
```

### 2. カメラ内部パラメータ行列の構築
```python
H, W = depth.shape
cx, cy = W / 2.0, H / 2.0
K = np.array([[f_px_val, 0,        cx],
              [0,        f_px_val, cy],
              [0,        0,        1]], dtype=np.float64)
```

### 3. 3D点群への変換
```python
def _unproject_depth_to_xyz(depth, K):
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    
    Z = depth  # [m]
    X = (xx - cx) * Z / fx
    Y = (yy - cy) * Z / fy
    
    xyz = np.stack([X, Y, Z], axis=-1)  # (H,W,3) [m]
    return xyz
```

### 4. 平面推定（RANSAC）
```python
# plane_fit_v2.pyより
def estimate_plane_from_depth_v2(depth, K, masks, margin_ratio=0.04, z_scale_factor=0.012):
    # マスク外側のリング領域を作成
    ring_mask = build_support_ring(union_mask, margin_ratio)
    
    # RANSACで平面フィッティング
    n, d = ransac_plane_fit(points[ring_mask], dist_th, max_iters=2000)
    
    return n, d, points_xyz
```

### 5. 体積計算
```python
# 高さマップ計算
height = height_map_from_plane(points_xyz, n, d, clip_negative=True)

# 画素面積計算
def pixel_area_map(depth, K):
    fx, fy = K[0, 0], K[1, 1]
    a_pix = (depth ** 2) / (fx * fy)  # [m^2]
    return a_pix

# 体積積分
volume_m3 = np.sum(height[mask] * a_pix[mask])  # [m^3]
volume_mL = volume_m3 * 1e6  # [mL]
```

## 観測された問題

### 測定値の比較
| モデル | 深度範囲 | 焦点距離(pixels) | 体積推定 | 画素面積 |
|--------|----------|------------------|----------|----------|
| UniDepth v2 (K_scale=8.0) | 1.127-2.189m | 5485.29 | 126.9 mL | 0.070 mm² |
| Depth Pro | 0.431-1.235m | 531.28 | 8224.2 mL | 1.297 mm² |

### 具体的な症状
1. **体積が約65倍大きい**: Depth Proで8.22L vs UniDepthで126.9mL
2. **深度範囲は妥当**: 0.431-1.235mは現実的な撮影距離
3. **焦点距離が小さい**: 531.28 pixels（UniDepthの1/10）
4. **画素面積が大きい**: 1.297 mm²（UniDepthの18.5倍）

## 考えられる原因

1. **焦点距離の単位や解釈の違い**
   - Depth Proの`focallength_px`の意味が異なる可能性
   - 35mm換算との変換が必要？

2. **深度値のスケーリング**
   - Depth Proの深度が既に何らかのスケーリング済み？
   - メートル単位ではない可能性？

3. **画像解像度の扱い**
   - Depth Proが内部で画像をリサイズしている
   - 元画像解像度と深度マップ解像度の不一致

4. **平面推定の問題**
   - 法線ベクトルの向きが不適切
   - RANSACの閾値がDepth Proに不適合

## 必要な情報とクエリ

1. **Depth Proの出力仕様について**
   - `prediction["depth"]`の正確な単位は？（本当にメートル？）
   - `prediction["focallength_px"]`は何を表している？（ピクセル単位の焦点距離？）
   - 内部でのリサイズ処理はどうなっている？

2. **カメラ内部パラメータの扱い**
   - Depth Proが想定する画像座標系は？
   - principal point (cx, cy)は画像中心で正しい？
   - 焦点距離の正規化方法は？

3. **体積計算の最適化**
   - Depth Pro用の画素面積計算式は？
   - 深度から実世界座標への変換で追加の係数が必要？
   - Apple公式の3D復元例はある？

4. **デバッグ方法**
   - 既知サイズの物体（定規、ボックス等）での検証方法
   - Depth Proの深度を可視化して確認する方法
   - 点群をPLYファイルに出力して3Dビューアで確認

これらについて、公式実装を参考にした解決策を教えてください。