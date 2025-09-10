# -*- coding: utf-8 -*-
"""
平面フィッティングモジュール（改良版）
符号決定ロジックの改善とスケール依存のパラメータ調整
"""
import numpy as np
import cv2
from typing import Tuple, Optional

def build_support_ring(food_union_mask: np.ndarray, margin_ratio: float = 0.04) -> np.ndarray:
    """
    食品マスクの外側リング領域（皿や卓面候補）を作成
    
    Args:
        food_union_mask: 全食品の結合マスク (H,W) bool
        margin_ratio: リング幅の割合（画像最小辺に対する比率）
    
    Returns:
        ring: リング領域のマスク (H,W) bool
    """
    H, W = food_union_mask.shape
    margin_px = int(margin_ratio * min(H, W))
    margin_px = max(margin_px, 10)  # 最小10ピクセル
    
    # カーネルサイズ（奇数にする）
    k = 2 * margin_px + 1
    kernel = np.ones((k, k), np.uint8)
    
    # マスクを膨張
    dil = cv2.dilate(food_union_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    
    # 元のマスクを除外してリングを作成
    ring = np.logical_and(dil, np.logical_not(food_union_mask))
    
    return ring

def fit_plane_ransac(
    points_xyz: np.ndarray,
    cand_mask: np.ndarray,
    z_scale_factor: float = 0.012,
    min_dist_th: float = 0.004,
    max_iters: int = 2000,
    min_support: int = 1000,
    rng_seed: int = 3
) -> Tuple[Tuple[np.ndarray, float], int]:
    """
    RANSAC法で平面を当てはめ（スケール依存の閾値）
    
    Args:
        points_xyz: (3,H,W) or (H,W,3) の3D点群
        cand_mask: (H,W) のbool（RANSAC候補点のマスク）
        z_scale_factor: 深度に対する距離閾値の係数
        min_dist_th: 最小距離閾値 [m]
        max_iters: 最大反復回数
        min_support: 最小サポート点数
        rng_seed: 乱数シード
    
    Returns:
        (n, d): 平面パラメータ（n·X + d = 0、nは単位法線ベクトル）
        inliers: インライア数
    """
    # 入力形状を統一
    if points_xyz.shape[0] == 3 and len(points_xyz.shape) == 3:
        # (3,H,W) -> (H,W,3)
        points_xyz = np.transpose(points_xyz, (1, 2, 0))
    
    H, W = cand_mask.shape
    
    # 候補点を抽出
    P = points_xyz[cand_mask].reshape(-1, 3)
    
    # 有限値のみを使用
    valid_mask = np.isfinite(P).all(axis=1)
    P = P[valid_mask]
    
    if P.shape[0] < min_support:
        raise RuntimeError(f"平面候補点が不足: {P.shape[0]} < {min_support}")
    
    # 深度の中央値からRANSAC閾値を自動調整
    z_median = np.median(P[:, 2])
    dist_th = max(min_dist_th, z_scale_factor * z_median)
    print(f"  RANSAC閾値: {dist_th*1000:.1f}mm (深度中央値: {z_median:.2f}m)")
    
    # RANSAC
    N = P.shape[0]
    best_inl = 0
    best_model = None
    rng = np.random.default_rng(rng_seed)
    
    for _ in range(max_iters):
        # 3点をランダムに選択
        idx = rng.choice(N, 3, replace=False)
        p1, p2, p3 = P[idx]
        
        # 平面の法線を計算（外積）
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        
        # 法線を正規化
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        
        # 平面方程式の定数項
        d = -np.dot(n, p1)
        
        # 点から平面までの距離
        dist = np.abs(P @ n + d)
        
        # インライアをカウント
        inl = np.sum(dist < dist_th)
        
        if inl > best_inl:
            best_inl = inl
            best_model = (n, d)
    
    if best_model is None:
        raise RuntimeError("平面を見つけられませんでした")
    
    # 最終的な平面をインライアで再推定（最小二乗法）
    n, d = best_model
    dist = np.abs(P @ n + d)
    inliers = P[dist < dist_th]
    
    if inliers.shape[0] >= 3:
        # SVDで最適平面を計算
        centroid = np.mean(inliers, axis=0)
        centered = inliers - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        n = Vt[-1]  # 最小特異値に対応する特異ベクトル
        d = -np.dot(n, centroid)
        best_model = (n, d)
    
    return best_model, best_inl

def height_map_from_plane(
    points_xyz: np.ndarray,
    plane_n: np.ndarray,
    plane_d: float,
    table_mask: Optional[np.ndarray] = None,
    food_mask: Optional[np.ndarray] = None,
    clip_negative: bool = True
) -> np.ndarray:
    """
    平面からの高さマップを計算（改良版符号決定）
    
    Args:
        points_xyz: 3D点群 (3,H,W) or (H,W,3)
        plane_n: 平面の法線ベクトル (3,)
        plane_d: 平面方程式の定数項
        table_mask: 卓面候補領域のマスク
        food_mask: 食品領域のマスク
        clip_negative: 負の高さを0にクリップするか
    
    Returns:
        height: 高さマップ (H,W) [m]
    """
    # 入力形状を統一
    if points_xyz.shape[-1] == 3:
        # (H,W,3) -> (3,H,W)
        points_xyz = np.transpose(points_xyz, (2, 0, 1))
    
    X = points_xyz[0]
    Y = points_xyz[1]
    Z = points_xyz[2]
    
    # 符号付き距離を計算
    h = plane_n[0] * X + plane_n[1] * Y + plane_n[2] * Z + plane_d
    
    # 符号の自動決定
    if table_mask is not None and np.any(table_mask):
        # 卓面の中央値を基準に
        med_table = np.median(h[table_mask])
        
        if food_mask is not None and np.any(food_mask):
            med_food = np.median(h[food_mask])
            # 食品が卓面より低い場合は符号を反転
            if med_food < med_table:
                h = -h
                print(f"  高さ符号を反転（食品側を正に）")
        else:
            # 卓面が0に近くなるように調整
            if abs(med_table) > 0.01:  # 1cm以上ずれている場合
                if med_table > 0:
                    h = -h
    else:
        # フォールバック：カメラ座標系のZ軸向きで判定
        if plane_n[2] > 0:
            h = -h
    
    if clip_negative:
        h = np.maximum(h, 0.0)
    
    return h

def estimate_plane_from_depth_v2(
    depth: np.ndarray,
    K: np.ndarray,
    food_masks: list,
    margin_ratio: float = 0.04,
    z_scale_factor: float = 0.012,
    min_support: int = 1000
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    深度マップと食品マスクから平面を推定（改良版）
    
    Args:
        depth: 深度マップ (H,W) [m]
        K: カメラ内部パラメータ (3,3)
        food_masks: 食品マスクのリスト
        margin_ratio: リング幅の割合
        z_scale_factor: RANSAC閾値の係数
        min_support: 最小サポート点数
    
    Returns:
        n: 平面の法線ベクトル (3,)
        d: 平面方程式の定数項
        points_xyz: 3D点群 (H,W,3)
    """
    H, W = depth.shape
    
    # 3D点群を生成（正しい逆投影）
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    points_xyz = np.stack([X, Y, Z], axis=-1)
    
    # 全食品マスクの結合
    if len(food_masks) > 0:
        union_mask = np.zeros((H, W), dtype=bool)
        for mask in food_masks:
            union_mask |= mask
    else:
        # マスクがない場合は画像中央を使用
        union_mask = np.zeros((H, W), dtype=bool)
        cy, cx = H // 2, W // 2
        r = min(H, W) // 4
        yy, xx = np.ogrid[:H, :W]
        union_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    
    # リング領域を作成
    ring = build_support_ring(union_mask, margin_ratio)
    
    # RANSAC（スケール依存の閾値）
    try:
        (n, d), nin = fit_plane_ransac(
            points_xyz, ring,
            z_scale_factor=z_scale_factor,
            min_support=min_support
        )
    except RuntimeError as e:
        print(f"リング領域でのRANSAC失敗: {e}")
        print("画像全体から平面を推定します...")
        
        full_mask = np.logical_not(union_mask)
        (n, d), nin = fit_plane_ransac(
            points_xyz, full_mask,
            z_scale_factor=z_scale_factor,
            min_support=min_support // 2
        )
    
    print(f"平面推定完了: インライア数={nin:.0f}")
    print(f"  法線: [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")
    print(f"  d: {d:.3f}")
    
    return n, d, points_xyz