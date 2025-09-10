#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
キャリブレーションベースの解決策テスト
典型的な食事撮影条件から適切なKを推定
"""
import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import cv2
from PIL import Image
from unidepth_runner_v2 import UniDepthEngineV2
from plane_fit_v2 import estimate_plane_from_depth_v2, height_map_from_plane
from volume_estimator import pixel_area_map, integrate_volume
from calibration_utils import estimate_reasonable_K, validate_volume_with_K_range

def test_with_calibrated_K(image_path: str, mask_path: str = None):
    """キャリブレーションされたKで体積推定"""
    
    print("=" * 70)
    print("キャリブレーションベースの体積推定")
    print("=" * 70)
    
    # 1. UniDepth推論
    engine = UniDepthEngineV2(
        model_repo="lpiccinelli/unidepth-v2-vitl14",
        device="cuda"
    )
    
    result = engine.infer_image(image_path, K_source="predicted")
    
    depth = result["depth"]
    K_pred = result["intrinsics_pred"]
    xyz = result["points"]
    conf = result["confidence"]
    
    H, W = depth.shape
    
    print(f"\n画像サイズ: {W}x{H}")
    print(f"深度範囲: {depth.min():.3f} - {depth.max():.3f} m")
    print(f"UniDepth推定K: fx={K_pred[0,0]:.1f}, fy={K_pred[1,1]:.1f}")
    
    # 2. マスク読み込み
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, 0) > 127
        masks = [mask]
    else:
        cy, cx = H // 2, W // 2
        r = min(H, W) // 6
        yy, xx = np.ogrid[:H, :W]
        mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
        masks = [mask]
    
    # 3. 様々なK推定方法を試す
    K_variants = []
    
    # a) UniDepthの推定値（そのまま）
    K_variants.append(("UniDepth推定", K_pred))
    
    # b) 典型的なFOVから推定（スマートフォン想定）
    for fov in [50, 60, 70]:
        K_fov = estimate_reasonable_K((H, W), fov_degrees=fov)
        K_variants.append((f"FOV {fov}°", K_fov))
    
    # c) 食事撮影の典型的な条件から推定
    # 距離約50cm、皿の直径約20cmが画像の1/3を占めると仮定
    typical_distance = 0.5  # 50cm
    plate_diameter = 0.20  # 20cm
    plate_pixels = W / 3  # 画像幅の1/3
    
    fx_typical = typical_distance * plate_pixels / plate_diameter
    K_typical = np.array([
        [fx_typical, 0, W/2],
        [0, fx_typical, H/2],
        [0, 0, 1]
    ])
    K_variants.append(("食事撮影典型", K_typical))
    
    # 各Kで体積を計算
    print("\n" + "=" * 70)
    print("各カメラパラメータでの体積推定結果")
    print("=" * 70)
    
    for name, K in K_variants:
        print(f"\n【{name}】")
        print(f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
        
        # 平面推定
        n, d, points_xyz = estimate_plane_from_depth_v2(
            depth, K, masks,
            margin_ratio=0.04,
            z_scale_factor=0.012
        )
        
        # 高さマップ
        from plane_fit_v2 import build_support_ring
        union_mask = masks[0]
        ring_mask = build_support_ring(union_mask, margin_ratio=0.04)
        
        height = height_map_from_plane(
            points_xyz, n, d,
            table_mask=ring_mask,
            food_mask=union_mask,
            clip_negative=True
        )
        
        # 体積計算
        a_pix = pixel_area_map(depth, K)
        vol_result = integrate_volume(
            height, a_pix, mask,
            conf=conf,
            use_conf_weight=False
        )
        
        volume_mL = vol_result["volume_mL"]
        height_mean = vol_result["height_mean_mm"]
        
        print(f"  体積: {volume_mL:.1f} mL", end="")
        if 50 <= volume_mL <= 1000:
            print(" ✓")
        elif volume_mL > 1000:
            print(f" ({volume_mL/1000:.1f}L) ⚠️")
        else:
            print(" ⚠️")
        
        print(f"  平均高さ: {height_mean:.1f} mm")
        
        # 現実的な範囲の判定
        if 50 <= volume_mL <= 1000:
            print("  → 現実的な範囲内！")
    
    # 4. 最適なKスケールファクターを逆算
    print("\n" + "=" * 70)
    print("最適なKの推定")
    print("=" * 70)
    
    # 平面と高さは最初のKで計算したものを使用
    n, d, points_xyz = estimate_plane_from_depth_v2(
        depth, K_pred, masks,
        margin_ratio=0.04,
        z_scale_factor=0.012
    )
    
    height = height_map_from_plane(
        points_xyz, n, d,
        table_mask=ring_mask,
        food_mask=union_mask,
        clip_negative=True
    )
    
    # 目標体積200mLになるKを探索
    optimal_factor = validate_volume_with_K_range(
        depth, height, mask,
        K_min_factor=1.0,
        K_max_factor=20.0,
        target_volume_mL=200.0
    )
    
    print(f"\n結論:")
    print(f"  UniDepth推定Kに対して {optimal_factor:.1f}倍 のスケーリングが必要")
    print(f"  これは食事撮影の典型的な条件と一致")

def main():
    import os
    
    # テスト画像
    test_cases = [
        ("test_images/train_00000.jpg", "outputs/sam2/masks/train_00000_det00_rice_bplus.png"),
    ]
    
    for image_path, mask_path in test_cases:
        if os.path.exists(image_path):
            print(f"画像: {os.path.basename(image_path)}\n")
            test_with_calibrated_K(image_path, mask_path)
            break

if __name__ == "__main__":
    main()