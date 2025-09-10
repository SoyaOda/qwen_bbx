# -*- coding: utf-8 -*-
"""
平面フィッティングモジュール
食品マスクの外側リング領域から皿/卓面をRANSACで推定
"""
import numpy as np
import cv2
from typing import Tuple, Optional

def build_support_ring(food_union_mask: np.ndarray, margin_px: int = 40) -> np.ndarray:
    """
    食品マスクの外側リング領域（皿や卓面候補）を作成
    
    Args:
        food_union_mask: 全食品の結合マスク (H,W) bool
        margin_px: リングの幅（ピクセル）
    
    Returns:
        ring: リング領域のマスク (H,W) bool
    """
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
    dist_th: float = 0.006,
    max_iters: int = 2000,
    min_support: int = 2000,
    rng_seed: int = 3
) -> Tuple[Tuple[np.ndarray, float], float]:
    """
    RANSAC法で平面を当てはめ
    
    Args:
        points_xyz: (3,H,W) の3D点群（カメラ座標系、メートル単位）
        cand_mask: (H,W) のbool（RANSAC候補点のマスク）
        dist_th: 点-平面距離の閾値 [m]
        max_iters: 最大反復回数
        min_support: 最小サポート点数
        rng_seed: 乱数シード
    
    Returns:
        (n, d): 平面パラメータ（n·X + d = 0、nは単位法線ベクトル）
        inliers: インライア数
    """
    H, W = cand_mask.shape
    ys, xs = np.where(cand_mask)
    
    if ys.size < min_support:
        raise RuntimeError(f"平面候補点が不足: {ys.size} < {min_support}")
    
    # 候補点の3D座標を抽出
    X = points_xyz[0, ys, xs]
    Y = points_xyz[1, ys, xs]
    Z = points_xyz[2, ys, xs]
    P = np.stack([X, Y, Z], axis=1)  # (N, 3)
    
    # 有限値のみを使用
    valid_mask = np.isfinite(P).all(axis=1)
    P = P[valid_mask]
    
    if P.shape[0] < min_support:
        raise RuntimeError(f"有効な平面候補点が不足: {P.shape[0]} < {min_support}")
    
    rs = np.random.RandomState(rng_seed)
    best_inliers = -1
    best_n, best_d = None, None
    
    for _ in range(max_iters):
        # ランダムに3点を選択
        idx = rs.choice(P.shape[0], size=3, replace=False)
        p1, p2, p3 = P[idx]
        
        # 平面の法線ベクトルを計算
        v1, v2 = p2 - p1, p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-8:
            continue  # 3点が同一直線上にある
        
        n = n / n_norm  # 正規化
        d = -np.dot(n, p1)  # 平面方程式のd
        
        # 全点から平面までの距離を計算
        dist = np.abs(P @ n + d)
        inliers = (dist < dist_th)
        n_in = int(inliers.sum())
        
        if n_in > best_inliers:
            # 最小二乗法でリファイン
            Q = P[inliers]
            
            # SVDを使って最適な平面を求める
            # 点群の重心を計算
            centroid = np.mean(Q, axis=0)
            Q_centered = Q - centroid
            
            # SVDで主成分分析
            _, _, vh = np.linalg.svd(Q_centered, full_matrices=False)
            n_ref = vh[-1, :]  # 最小特異値に対応する特異ベクトル（法線）
            
            # 法線の向きを統一（+Z方向が上）
            if n_ref[2] < 0:
                n_ref = -n_ref
            
            # 平面方程式のdを再計算
            d_ref = -np.dot(n_ref, centroid)
            
            best_n, best_d, best_inliers = n_ref, d_ref, n_in
    
    if best_n is None:
        raise RuntimeError("RANSAC平面推定に失敗しました")
    
    return (best_n, float(best_d)), float(best_inliers)

def estimate_plane_from_depth(
    depth: np.ndarray,
    K: np.ndarray,
    food_masks: list,
    margin_px: int = 40,
    dist_th: float = 0.006,
    max_iters: int = 2000,
    min_support: int = 2000
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    深度マップと食品マスクから平面を推定
    
    Args:
        depth: 深度マップ (H,W) [m]
        K: カメラ内部パラメータ (3,3)
        food_masks: 食品マスクのリスト
        margin_px: リングマージン
        dist_th: RANSAC距離閾値
        max_iters: RANSAC最大反復回数
        min_support: 最小サポート点数
    
    Returns:
        n: 平面の法線ベクトル (3,)
        d: 平面方程式の定数項
        points_xyz: 3D点群 (3,H,W)
    """
    H, W = depth.shape
    
    # 3D点群を生成
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # ピクセル座標のメッシュグリッド
    us = np.arange(W)
    vs = np.arange(H)
    uu, vv = np.meshgrid(us, vs)
    
    # 3D座標を計算
    Z = depth
    X = (uu - cx) / fx * Z
    Y = (vv - cy) / fy * Z
    points_xyz = np.stack([X, Y, Z], axis=0)
    
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
    ring = build_support_ring(union_mask, margin_px)
    
    # RANSACで平面フィッティング
    try:
        (n, d), nin = fit_plane_ransac(
            points_xyz, ring,
            dist_th=dist_th,
            max_iters=max_iters,
            min_support=min_support
        )
    except RuntimeError as e:
        # フォールバック：画像全体から平面を推定
        print(f"リング領域でのRANSAC失敗: {e}")
        print("画像全体から平面を推定します...")
        
        full_mask = np.logical_not(union_mask)
        (n, d), nin = fit_plane_ransac(
            points_xyz, full_mask,
            dist_th=dist_th,
            max_iters=max_iters,
            min_support=min_support // 2
        )
    
    print(f"平面推定完了: インライア数={nin:.0f}")
    print(f"  法線: [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")
    print(f"  d: {d:.3f}")
    
    return n, d, points_xyz