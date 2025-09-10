#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改良版実装のテスト
K_scale_factorを使わずに現実的な体積が得られるか確認
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

def test_volume_estimation(image_path: str, mask_path: str = None):
    """改良版実装で体積を推定"""
    
    print("=" * 70)
    print("改良版実装テスト（K_scale_factor撤廃）")
    print("=" * 70)
    
    # 1. UniDepth推論（改良版）
    print("\n1. UniDepth v2 推論...")
    engine = UniDepthEngineV2(
        model_repo="lpiccinelli/unidepth-v2-vitl14",
        device="cuda"
    )
    
    # 推論実行（K_sourceを変えてテスト）
    for k_source in ["predicted", "exif", "auto"]:
        print(f"\n--- K_source = {k_source} ---")
        
        try:
            result = engine.infer_image(image_path, K_source=k_source)
            
            depth = result["depth"]
            K = result["intrinsics"]
            K_pred = result["intrinsics_pred"]
            K_exif = result["intrinsics_exif"]
            xyz = result["points"]
            conf = result["confidence"]
            
            H, W = depth.shape
            
            print(f"深度範囲: {depth.min():.3f} - {depth.max():.3f} m")
            
            # 2. マスク読み込み
            if mask_path and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, 0) > 127
                masks = [mask]
                labels = ["food"]
            else:
                # テスト用：画像中央を食品と仮定
                cy, cx = H // 2, W // 2
                r = min(H, W) // 6
                yy, xx = np.ogrid[:H, :W]
                mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
                masks = [mask]
                labels = ["test_food"]
            
            print(f"マスクピクセル数: {mask.sum()}")
            
            # 3. 平面推定（改良版）
            print("\n2. 平面推定...")
            n, d, points_xyz = estimate_plane_from_depth_v2(
                depth, K, masks,
                margin_ratio=0.04,  # 画像最小辺の4%
                z_scale_factor=0.012  # 深度の1.2%
            )
            
            # 4. 高さマップ計算（改良版）
            # リング領域を再計算（符号決定用）
            from plane_fit_v2 import build_support_ring
            union_mask = masks[0]
            ring_mask = build_support_ring(union_mask, margin_ratio=0.04)
            
            height = height_map_from_plane(
                points_xyz, n, d,
                table_mask=ring_mask,
                food_mask=union_mask,
                clip_negative=True
            )
            
            # 5. 体積計算
            print("\n3. 体積計算...")
            a_pix = pixel_area_map(depth, K)
            
            # a_pixの統計
            a_pix_mean = a_pix[mask].mean()
            print(f"平均a_pix: {a_pix_mean:.2e} m²/px")
            
            # 体積積分
            vol_result = integrate_volume(
                height, a_pix, mask,
                conf=conf,
                use_conf_weight=False
            )
            
            volume_mL = vol_result["volume_mL"]
            height_mean = vol_result["height_mean_mm"]
            height_max = vol_result["height_max_mm"]
            
            print(f"\n結果:")
            print(f"  体積: {volume_mL:.1f} mL", end="")
            if volume_mL > 1000:
                print(f" ({volume_mL/1000:.2f}L) ⚠️")
            elif volume_mL < 10:
                print(f" ⚠️ (小さすぎ)")
            else:
                print(" ✓")
            
            print(f"  平均高さ: {height_mean:.1f} mm")
            print(f"  最大高さ: {height_max:.1f} mm")
            
            # 現実的な範囲のチェック
            if 50 <= volume_mL <= 1000:
                print("  → 現実的な範囲内 ✓")
            else:
                print(f"  → 異常な値（期待: 50-1000mL）")
                
                # デバッグ情報
                print("\nデバッグ:")
                print(f"  深度中央値: {np.median(depth):.3f} m")
                print(f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
                print(f"  高さ中央値: {np.median(height[mask])*1000:.1f} mm")
                
                # 目標体積から逆算
                target_mL = 200
                scale_needed = target_mL / volume_mL
                print(f"\n  200mLにするには:")
                print(f"    - 現在の{scale_needed:.3f}倍が必要")
                print(f"    - fx,fyを{1/np.sqrt(scale_needed):.2f}倍に")
                
        except Exception as e:
            print(f"エラー: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("まとめ:")
    print("K_scale_factorなしでの体積推定結果を確認してください。")
    print("現実的な範囲（50-1000mL）に収まれば成功です。")
    print("=" * 70)

def main():
    # テスト画像
    test_cases = [
        ("test_images/train_00000.jpg", "outputs/sam2/masks/train_00000_det00_rice_bplus.png"),
        ("test_images/train_00001.jpg", "outputs/sam2/masks/train_00001_det00_mashed_potatoes_bplus.png"),
        ("test_images/train_00002.jpg", "outputs/sam2/masks/train_00002_det00_French_toast_bplus.png"),
    ]
    
    for image_path, mask_path in test_cases:
        if os.path.exists(image_path):
            print(f"\n\n画像: {os.path.basename(image_path)}")
            test_volume_estimation(image_path, mask_path)
            break  # 最初の1枚だけテスト

if __name__ == "__main__":
    main()