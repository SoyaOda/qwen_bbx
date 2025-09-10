# -*- coding: utf-8 -*-
"""
体積推定モジュール
平面からの高さマップを生成し、マスクごとの体積を積分
"""
import numpy as np
from typing import Dict, Any, Optional, List

def ensure_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    深度マップとカメラ内部パラメータから3D点群を生成
    
    Args:
        depth: 深度マップ (H,W) [m]
        K: カメラ内部パラメータ (3,3)
    
    Returns:
        points: 3D点群 (3,H,W) [m]
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # ピクセル座標のメッシュグリッド
    us = np.arange(W)
    vs = np.arange(H)
    uu, vv = np.meshgrid(us, vs)
    
    # 3D座標を計算（カメラ座標系）
    Z = depth
    X = (uu - cx) / fx * Z
    Y = (vv - cy) / fy * Z
    
    return np.stack([X, Y, Z], axis=0)

def height_map_from_plane(
    points_xyz: np.ndarray,
    plane_n: np.ndarray,
    plane_d: float,
    clip_negative: bool = True
) -> np.ndarray:
    """
    平面からの高さマップを計算
    
    Args:
        points_xyz: 3D点群 (3,H,W) [m]
        plane_n: 平面の法線ベクトル (3,)
        plane_d: 平面方程式の定数項
        clip_negative: 負の高さを0にクリップするか
    
    Returns:
        height: 高さマップ (H,W) [m]
    """
    # 各点の平面からの符号付き距離を計算
    # 平面方程式: n·X + d = 0
    # 点から平面までの距離: h = n·X + d
    X = points_xyz[0]
    Y = points_xyz[1]
    Z = points_xyz[2]
    
    h = plane_n[0] * X + plane_n[1] * Y + plane_n[2] * Z + plane_d
    
    # 皿面が0、上が正になるように符号を調整
    # （法線が上向きの場合、皿の上の点はh > 0になるはず）
    if plane_n[2] > 0:  # 法線が上向き
        h = -h  # 符号を反転
    
    if clip_negative:
        h = np.maximum(h, 0.0)
    
    return h

def pixel_area_map(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    各ピクセルが表す実世界の面積を計算
    
    Args:
        depth: 深度マップ (H,W) [m]
        K: カメラ内部パラメータ (3,3)
    
    Returns:
        area: ピクセル面積マップ (H,W) [m²]
    """
    fx, fy = K[0, 0], K[1, 1]
    
    # 小面積近似: a_pix(z) ≈ (z²)/(fx·fy)
    # これは、カメラから距離zにある平面上の1ピクセルが表す面積
    area = (depth ** 2) / (fx * fy + 1e-12)
    
    return area

def integrate_volume(
    height: np.ndarray,
    a_pix: np.ndarray,
    mask_bool: np.ndarray,
    conf: Optional[np.ndarray] = None,
    use_conf_weight: bool = False
) -> Dict[str, Any]:
    """
    マスク内の体積を積分
    
    Args:
        height: 高さマップ (H,W) [m]
        a_pix: ピクセル面積マップ (H,W) [m²]
        mask_bool: 対象領域のマスク (H,W) bool
        conf: 信頼度マップ (H,W)（オプション）
        use_conf_weight: 信頼度による重み付けを使用するか
    
    Returns:
        dict: {
            "pixels": マスク内のピクセル数,
            "volume_mL": 体積（ミリリットル）,
            "height_mean_mm": 平均高さ（ミリメートル）,
            "height_max_mm": 最大高さ（ミリメートル）
        }
    """
    m = mask_bool.astype(bool)
    
    if not np.any(m):
        return {
            "pixels": 0,
            "volume_mL": 0.0,
            "height_mean_mm": 0.0,
            "height_max_mm": 0.0
        }
    
    # 体積積分: V = Σ h(x,y) * a_pix(x,y)
    if use_conf_weight and conf is not None:
        # 信頼度による重み付け
        w = np.clip(conf, 0.0, 1.0)
        V = float(np.sum(height[m] * a_pix[m] * w[m]))
    else:
        # 重み付けなし
        V = float(np.sum(height[m] * a_pix[m]))
    
    # m³ → mL (1m³ = 1,000,000 mL)
    volume_mL = V * 1e6
    
    # 高さ統計（m → mm）
    heights_m = height[m]
    height_mean_mm = float(np.mean(heights_m)) * 1000
    height_max_mm = float(np.max(heights_m)) * 1000
    
    return {
        "pixels": int(m.sum()),
        "volume_mL": volume_mL,
        "height_mean_mm": height_mean_mm,
        "height_max_mm": height_max_mm
    }

def estimate_volumes(
    depth: np.ndarray,
    K: np.ndarray,
    plane_n: np.ndarray,
    plane_d: float,
    masks: List[np.ndarray],
    labels: List[str],
    confidence: Optional[np.ndarray] = None,
    use_conf_weight: bool = False
) -> List[Dict[str, Any]]:
    """
    複数のマスクに対して体積を推定
    
    Args:
        depth: 深度マップ (H,W) [m]
        K: カメラ内部パラメータ (3,3)
        plane_n: 平面の法線ベクトル (3,)
        plane_d: 平面方程式の定数項
        masks: マスクのリスト
        labels: ラベルのリスト
        confidence: 信頼度マップ（オプション）
        use_conf_weight: 信頼度による重み付けを使用するか
    
    Returns:
        results: 各マスクの体積推定結果のリスト
    """
    # 3D点群を生成
    points_xyz = ensure_points(depth, K)
    
    # 高さマップを計算
    height = height_map_from_plane(points_xyz, plane_n, plane_d, clip_negative=True)
    
    # ピクセル面積マップを計算
    a_pix = pixel_area_map(depth, K)
    
    # 各マスクに対して体積を計算
    results = []
    for i, (mask, label) in enumerate(zip(masks, labels)):
        # 信頼度なしで計算
        vol_plain = integrate_volume(height, a_pix, mask, conf=None, use_conf_weight=False)
        
        # 信頼度ありで計算（可能な場合）
        if confidence is not None and use_conf_weight:
            vol_conf = integrate_volume(height, a_pix, mask, conf=confidence, use_conf_weight=True)
        else:
            vol_conf = vol_plain
        
        result = {
            "id": i,
            "label": label,
            "pixels": vol_plain["pixels"],
            "volume_mL": vol_plain["volume_mL"],
            "volume_mL_conf": vol_conf["volume_mL"] if use_conf_weight else None,
            "height_mean_mm": vol_plain["height_mean_mm"],
            "height_max_mm": vol_plain["height_max_mm"]
        }
        results.append(result)
        
        print(f"  {label}:")
        print(f"    体積: {vol_plain['volume_mL']:.1f} mL")
        print(f"    平均高さ: {vol_plain['height_mean_mm']:.1f} mm")
        print(f"    最大高さ: {vol_plain['height_max_mm']:.1f} mm")
    
    return results