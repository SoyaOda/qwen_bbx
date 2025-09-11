#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Pro実装のテスト
Apple Depth Proを使用した体積推定の検証
"""
import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import cv2
from PIL import Image
# Depth Proまたはシミュレーション版を使用
try:
    # Depth Proのパスを追加
    sys.path.insert(0, 'ml-depth-pro/src')
    import depth_pro
    from src.depthpro_runner import DepthProEngine
    print("実際のDepth Proを使用します")
except ImportError as e:
    print(f"Depth Proが利用できないため、シミュレーション版を使用します: {e}")
    from src.depthpro_runner_sim import DepthProEngineSimulated as DepthProEngine
from src.plane_fit_v2 import estimate_plane_from_depth_v2, height_map_from_plane
from src.volume_estimator import pixel_area_map, integrate_volume

def test_volume_estimation_depthpro(image_path: str, mask_path: str = None):
    """Depth Proを使用した体積推定"""
    
    print("=" * 70)
    print("Depth Pro実装テスト（絶対スケール深度推定）")
    print("=" * 70)
    
    # 1. Depth Pro推論
    print("\n1. Depth Pro 推論...")
    try:
        engine = DepthProEngine(device="cuda")
    except Exception as e:
        print(f"Depth Proエンジンの初期化に失敗: {e}")
        print("\nDepth Proのインストール手順:")
        print("1. git clone https://github.com/apple/ml-depth-pro.git")
        print("2. cd ml-depth-pro && pip install -e .")
        print("3. huggingface-cli download --local-dir checkpoints apple/DepthPro")
        return
    
    try:
        result = engine.infer_image(image_path)
        
        depth = result["depth"]
        K = result["intrinsics"]
        xyz = result["points"]
        
        H, W = depth.shape
        
        print(f"深度範囲: {depth.min():.3f} - {depth.max():.3f} m")
        print(f"深度中央値: {np.median(depth):.3f} m")
        
        # 2. マスク読み込み
        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0) > 127
            masks = [mask]
            labels = ["food"]
            print(f"マスク読み込み: {os.path.basename(mask_path)}")
        else:
            # テスト用：画像中央を食品と仮定
            cy, cx = H // 2, W // 2
            r = min(H, W) // 6
            yy, xx = np.ogrid[:H, :W]
            mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
            masks = [mask]
            labels = ["test_food"]
            print("テスト用中央円マスクを使用")
        
        print(f"マスクピクセル数: {mask.sum()}")
        
        # 3. 平面推定
        print("\n2. 平面推定...")
        n, d, points_xyz = estimate_plane_from_depth_v2(
            depth, K, masks,
            margin_ratio=0.04,  # 画像最小辺の4%
            z_scale_factor=0.012  # 深度の1.2%
        )
        
        # 4. 高さマップ計算
        # リング領域を再計算（符号決定用）
        from src.plane_fit_v2 import build_support_ring
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
        print(f"平均画素面積: {a_pix_mean:.2e} m²/px")
        print(f"  = {a_pix_mean * 1e6:.3f} mm²/px")
        
        # 体積積分
        vol_result = integrate_volume(
            height, a_pix, mask,
            conf=None,  # Depth Proは信頼度マップなし
            use_conf_weight=False
        )
        
        volume_mL = vol_result["volume_mL"]
        height_mean = vol_result["height_mean_mm"]
        height_max = vol_result["height_max_mm"]
        
        print(f"\n結果:")
        print(f"  体積: {volume_mL:.1f} mL", end="")
        if volume_mL > 1000:
            print(f" ({volume_mL/1000:.2f}L)")
        else:
            print()
        
        print(f"  平均高さ: {height_mean:.1f} mm")
        print(f"  最大高さ: {height_max:.1f} mm")
        
        # 現実的な範囲のチェック
        if 50 <= volume_mL <= 1000:
            print("  → ✓ 現実的な範囲内（50-1000mL）")
        elif volume_mL < 50:
            print("  → ⚠ 小さすぎる可能性")
        else:
            print(f"  → ⚠ 大きすぎる可能性（{volume_mL/1000:.2f}L）")
        
        # Depth Proの利点を確認
        print("\n【Depth Proの利点】")
        print("• K_scale補正が不要（絶対スケール推定）")
        print("• 焦点距離の自動推定または EXIF活用")
        print("• 高精度な深度マップ（細部まで保持）")
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)

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
            test_volume_estimation_depthpro(image_path, mask_path)
            break  # 最初の1枚だけテスト

if __name__ == "__main__":
    main()