# -*- coding: utf-8 -*-
"""
カメラキャリブレーションユーティリティ
既知サイズの物体（A4紙など）を使ってKを推定
"""
import numpy as np
from typing import Optional, Tuple

def calibrate_K_from_known_object(
    object_pixels_width: float,
    object_pixels_height: float,
    object_real_width_m: float,
    object_real_height_m: float,
    object_depth_m: float,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    既知サイズの物体からカメラ内部パラメータを推定
    
    Args:
        object_pixels_width: 物体の画像上の幅（ピクセル）
        object_pixels_height: 物体の画像上の高さ（ピクセル）
        object_real_width_m: 物体の実際の幅（メートル）
        object_real_height_m: 物体の実際の高さ（メートル）
        object_depth_m: 物体までの距離（メートル）
        image_shape: (H, W) 画像サイズ
        
    Returns:
        推定されたK行列
    """
    H, W = image_shape
    
    # ピンホールモデル: 実長さ = Z * ピクセル数 / f
    # → f = Z * ピクセル数 / 実長さ
    
    fx = object_depth_m * object_pixels_width / object_real_width_m
    fy = object_depth_m * object_pixels_height / object_real_height_m
    
    cx = W / 2.0
    cy = H / 2.0
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    print(f"キャリブレーション結果:")
    print(f"  fx = {fx:.1f}")
    print(f"  fy = {fy:.1f}")
    print(f"  物体: {object_real_width_m*1000:.0f}x{object_real_height_m*1000:.0f}mm")
    print(f"  画像上: {object_pixels_width:.0f}x{object_pixels_height:.0f}px")
    print(f"  距離: {object_depth_m:.2f}m")
    
    return K

def calibrate_K_from_A4(
    a4_pixels_width: float,
    a4_pixels_height: float,
    a4_depth_m: float,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    A4紙（297x210mm）からKを推定
    """
    A4_WIDTH_M = 0.297
    A4_HEIGHT_M = 0.210
    
    return calibrate_K_from_known_object(
        a4_pixels_width, a4_pixels_height,
        A4_WIDTH_M, A4_HEIGHT_M,
        a4_depth_m, image_shape
    )

def estimate_reasonable_K(
    image_shape: Tuple[int, int],
    fov_degrees: float = 60.0
) -> np.ndarray:
    """
    典型的なFOVから妥当なKを推定
    
    Args:
        image_shape: (H, W)
        fov_degrees: 水平視野角（度）
        
    Returns:
        推定K行列
    """
    H, W = image_shape
    
    # FOVから焦点距離を計算
    # fx = W / (2 * tan(FOV/2))
    fov_rad = np.deg2rad(fov_degrees)
    fx = W / (2 * np.tan(fov_rad / 2))
    fy = fx  # アスペクト比1:1と仮定
    
    cx = W / 2.0
    cy = H / 2.0
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    print(f"FOVベースのK推定:")
    print(f"  FOV: {fov_degrees}°")
    print(f"  fx = fy = {fx:.1f}")
    
    return K

def validate_volume_with_K_range(
    depth: np.ndarray,
    height: np.ndarray,
    mask: np.ndarray,
    K_min_factor: float = 0.5,
    K_max_factor: float = 20.0,
    target_volume_mL: float = 200.0
) -> float:
    """
    体積が目標値になるようなKスケールファクターを探索
    
    Args:
        depth: 深度マップ
        height: 高さマップ
        mask: 対象領域のマスク
        K_min_factor: 最小スケールファクター
        K_max_factor: 最大スケールファクター
        target_volume_mL: 目標体積（mL）
        
    Returns:
        最適なKスケールファクター
    """
    # 現在のK（仮定値）での体積計算
    fx_base = 1000.0  # 基準値
    
    # 二分探索で最適なスケールを見つける
    low, high = K_min_factor, K_max_factor
    
    for _ in range(20):  # 最大20回の反復
        mid = (low + high) / 2.0
        fx = fx_base * mid
        fy = fx
        
        # a_pix計算
        a_pix = (depth ** 2) / (fx * fy)
        
        # 体積計算
        volume_m3 = np.sum(height[mask] * a_pix[mask])
        volume_mL = volume_m3 * 1e6
        
        if abs(volume_mL - target_volume_mL) < 10:  # 10mL以内なら終了
            break
            
        if volume_mL > target_volume_mL:
            low = mid  # fxを大きくする必要
        else:
            high = mid  # fxを小さくする必要
    
    optimal_factor = mid
    optimal_fx = fx_base * optimal_factor
    
    print(f"体積キャリブレーション:")
    print(f"  目標体積: {target_volume_mL:.0f}mL")
    print(f"  推奨fx: {optimal_fx:.0f}")
    print(f"  スケールファクター: {optimal_factor:.2f}")
    print(f"  実際の体積: {volume_mL:.1f}mL")
    
    return optimal_factor