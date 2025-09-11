"""
Depth Pro用の堅牢な平面推定
水平性の事前分布とリング領域の自動拡張を使用
"""
import numpy as np
import cv2
from typing import Tuple, Optional

def build_support_ring(mask: np.ndarray, min_margin: float = 0.04, 
                       max_margin: float = 0.15, step: float = 0.02) -> np.ndarray:
    """
    外側リングを4%→最大15%まで自動拡張。最終的にunionしたリングを返す。
    
    Args:
        mask: 食品マスク (H,W) bool
        min_margin: 最小マージン比率
        max_margin: 最大マージン比率
        step: 拡張ステップ
    
    Returns:
        ring: 拡張されたリング領域 (H,W) bool
    """
    H, W = mask.shape
    ring = np.zeros_like(mask, dtype=bool)
    
    for ratio in np.arange(min_margin, max_margin + 1e-6, step):
        k = int(max(1, round(min(H, W) * ratio)))
        # マスクを膨張
        kernel = np.ones((k, k), np.uint8)
        dilated = cv2.dilate(mask.astype(np.uint8), kernel)
        # 外側リング部分のみ取得
        ring_part = (dilated.astype(bool) & (~mask.astype(bool)))
        ring |= ring_part
    
    return ring


def gradient_mask(depth: np.ndarray, thr: float = 0.004) -> np.ndarray:
    """
    深度勾配の小さい（フラット）点だけTrueにする
    
    Args:
        depth: 深度マップ (H,W) [m]
        thr: 勾配閾値 [m/pixel]
    
    Returns:
        mask: フラットな領域 (H,W) bool
    """
    gy, gx = np.gradient(depth)
    g = np.hypot(gx, gy)
    return (g < thr)


def ransac_plane(points_xyz: np.ndarray, dist_th: float, 
                max_iters: int = 2000, seed: int = 0) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    最小実装のRANSAC平面フィッティング
    
    Args:
        points_xyz: 3D点群 (N,3) [m]
        dist_th: インライア判定の距離閾値 [m]
        max_iters: 最大反復回数
        seed: 乱数シード
    
    Returns:
        n: 平面法線（単位ベクトル）
        d: 平面パラメータ（n·p = d）
        inlier_idx: インライアのインデックス
    """
    N = points_xyz.shape[0]
    if N < 3:
        raise ValueError("点群が少なすぎます")
    
    best_inliers = None
    best_model = None
    rng = np.random.default_rng(seed)
    
    for _ in range(max_iters):
        # 3点をランダムに選択
        idx = rng.choice(N, size=3, replace=False)
        p0, p1, p2 = points_xyz[idx]
        
        # 平面の法線を計算
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-9:
            continue  # 3点が一直線上
        
        n = n / n_norm  # 正規化
        d = np.dot(n, p0)
        
        # 符号の正規化（z正方向をデフォルトに）
        if n[2] < 0:
            n, d = -n, -d
        
        # 全点との距離を計算
        dist = np.abs(points_xyz.dot(n) - d)
        inliers = np.where(dist < dist_th)[0]
        
        # 最良モデルの更新
        if best_inliers is None or inliers.size > best_inliers.size:
            best_inliers = inliers
            best_model = (n, d)
    
    if best_model is None:
        raise RuntimeError("RANSAC失敗: 平面が見つかりませんでした")
    
    n, d = best_model
    return n, d, best_inliers


def estimate_table_plane(depth: np.ndarray, K: np.ndarray, food_mask: np.ndarray,
                        z_med: Optional[float] = None, 
                        z_grad_thr: float = 0.004,
                        horiz_deg: float = 15.0,
                        verbose: bool = True) -> Tuple[np.ndarray, float]:
    """
    テーブル平面を推定（水平性の事前分布を使用）
    
    Args:
        depth: 深度マップ (H,W) [m]
        K: カメラ内部パラメータ (3,3)
        food_mask: 食品マスク (H,W) bool
        z_med: 深度中央値（Noneの場合は自動計算）
        z_grad_thr: 勾配閾値 [m/pixel]
        horiz_deg: 水平性の許容角度 [degrees]
        verbose: 詳細出力
    
    Returns:
        n: 平面法線（単位ベクトル）
        d: 平面パラメータ（n·p = d）
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    if z_med is None:
        z_med = np.median(depth)
    
    # 1) 候補点抽出：外側リング∩低勾配
    if verbose:
        print("平面推定: リング領域を構築中...")
    
    ring = build_support_ring(food_mask, min_margin=0.04, max_margin=0.15, step=0.02)
    flat = gradient_mask(depth, thr=z_grad_thr)
    support = (ring & flat)
    
    support_count = np.sum(support)
    if support_count < 100:
        # サポート点が少なすぎる場合は勾配制約を緩める
        if verbose:
            print(f"  サポート点が少ない({support_count})ため、勾配制約を緩めます")
        flat = gradient_mask(depth, thr=z_grad_thr * 2)
        support = (ring & flat)
        support_count = np.sum(support)
    
    if verbose:
        print(f"  サポート点数: {support_count}")
    
    # 2) 深度→3D点群
    v, u = np.where(support)
    Z = depth[v, u]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    P = np.stack([X, Y, Z], axis=1)
    
    # 3) RANSACを複数回走らせ、水平性でフィルタ
    candidates = []
    
    # 深度に応じた距離閾値を試す
    dist_thresholds = [0.002, 0.004, 0.006, 0.008, 0.010]  # 2〜10mm
    
    for dist_th in dist_thresholds:
        try:
            n, d, inliers = ransac_plane(P, dist_th=dist_th, max_iters=1500)
        except RuntimeError:
            continue
        
        nz = float(n[2])
        
        # 水平性（n·[0,0,1]）の事前分布
        cos_th = np.cos(np.deg2rad(horiz_deg))
        is_horizontal = (nz >= cos_th)
        
        if is_horizontal:
            candidates.append((inliers.size, nz, dist_th, (n, d)))
            if verbose:
                print(f"  候補平面: dist_th={dist_th*1000:.1f}mm, "
                      f"インライア={inliers.size}, n_z={nz:.3f}")
    
    # 候補がない場合は制約を緩める
    if not candidates:
        if verbose:
            print(f"  水平な平面が見つからないため、制約を緩めます（最大25度）")
        
        for dist_th in [0.004, 0.006, 0.008, 0.010]:
            try:
                n, d, inliers = ransac_plane(P, dist_th=dist_th, max_iters=1500)
            except RuntimeError:
                continue
            
            nz = float(n[2])
            cos_th_relaxed = np.cos(np.deg2rad(25.0))
            
            if nz >= cos_th_relaxed:
                candidates.append((inliers.size, nz, dist_th, (n, d)))
                if verbose:
                    print(f"  緩和候補: dist_th={dist_th*1000:.1f}mm, "
                          f"インライア={inliers.size}, n_z={nz:.3f}")
    
    if not candidates:
        raise RuntimeError("テーブル平面が見つかりませんでした")
    
    # 4) インライア数→n_zの優先度で決定
    # インライア数を優先し、同じならn_z（垂直性）が高いものを選ぶ
    candidates.sort(key=lambda t: (t[0], t[1]))
    best_inliers, best_nz, best_dist_th, (n, d) = candidates[-1]
    
    if verbose:
        print(f"\n選択された平面:")
        print(f"  法線: [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")
        print(f"  d: {d:.3f}")
        print(f"  インライア数: {best_inliers}")
        print(f"  距離閾値: {best_dist_th*1000:.1f}mm")
        print(f"  水平度: {np.rad2deg(np.arccos(best_nz)):.1f}度")
    
    # 健全性チェック
    if abs(n[2]) < 0.85:
        if verbose:
            print(f"  警告: 平面が傾きすぎています（n_z={n[2]:.3f}）")
    
    return n, d