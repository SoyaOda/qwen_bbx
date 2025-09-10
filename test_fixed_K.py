#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
固定カメラパラメータでの体積推定テスト
"""
import numpy as np
import torch
from PIL import Image
from unidepth.models import UniDepthV2
from torchvision import transforms
import cv2
import os

def test_with_fixed_K():
    """固定のカメラパラメータで体積を推定"""
    
    # テスト画像とマスク
    image_path = "test_images/train_00000.jpg"
    mask_path = "outputs/sam2/masks/train_00000_det00_rice_bplus.png"
    
    print("=" * 60)
    print("固定カメラパラメータテスト")
    print("=" * 60)
    
    # モデルロード
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(device)
    model.eval()
    
    # 画像読み込み
    rgb_pil = Image.open(image_path).convert("RGB")
    rgb_np = np.array(rgb_pil)
    H, W = rgb_np.shape[:2]
    
    # ImageNet正規化
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    rgb_tensor = torch.from_numpy(rgb_np).float() / 255.0
    rgb_tensor = rgb_tensor.permute(2, 0, 1)
    rgb_tensor = normalize(rgb_tensor)
    rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
    
    # 推論実行
    with torch.inference_mode():
        pred = model.infer(rgb_tensor)
        depth = pred["depth"].squeeze().cpu().numpy()
        K_estimated = pred["intrinsics"].squeeze().cpu().numpy()
    
    print(f"\n元画像サイズ: {W}x{H}")
    print(f"深度範囲: {depth.min():.3f} - {depth.max():.3f} m")
    
    # 推定されたKと固定Kの比較
    print("\n【推定されたK】")
    print(f"  fx={K_estimated[0,0]:.1f}, fy={K_estimated[1,1]:.1f}")
    print(f"  cx={K_estimated[0,2]:.1f}, cy={K_estimated[1,2]:.1f}")
    
    # 固定カメラパラメータ（スケールファクターを適用）
    scale_factor = 6.0  # デバッグ結果から得られた倍率
    K_fixed = K_estimated.copy()
    K_fixed[0, 0] *= scale_factor  # fx
    K_fixed[1, 1] *= scale_factor  # fy
    
    print("\n【固定K（スケール調整後）】")
    print(f"  fx={K_fixed[0,0]:.1f}, fy={K_fixed[1,1]:.1f}")
    print(f"  cx={K_fixed[0,2]:.1f}, cy={K_fixed[1,2]:.1f}")
    
    # マスク読み込み
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, 0) > 127
        print(f"\nマスク: rice")
        print(f"  ピクセル数: {mask.sum()}")
    else:
        # マスクがない場合は画像中央を使用
        cy, cx = H // 2, W // 2
        r = min(H, W) // 6
        yy, xx = np.ogrid[:H, :W]
        mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
        print(f"\nテストマスク（画像中央）")
        print(f"  ピクセル数: {mask.sum()}")
    
    # 両方のKで体積を計算
    for name, K in [("推定K", K_estimated), ("固定K", K_fixed)]:
        print(f"\n【{name}での体積計算】")
        
        # ピクセル面積
        fx, fy = K[0,0], K[1,1]
        a_pix_map = (depth ** 2) / (fx * fy)
        a_pix_mean = a_pix_map[mask].mean()
        
        print(f"  平均a_pix: {a_pix_mean:.2e} m²/px")
        
        # 仮定する高さ
        heights_cm = [1, 2, 3, 5]  # cm
        
        for h_cm in heights_cm:
            h_m = h_cm / 100.0
            volume_m3 = mask.sum() * h_m * a_pix_mean
            volume_mL = volume_m3 * 1e6
            
            print(f"  高さ{h_cm}cm → {volume_mL:.1f} mL", end="")
            
            if 100 <= volume_mL <= 500:
                print(" ✓")
            else:
                print(f" ({volume_mL/1000:.1f}L)")
    
    # 推奨値の計算
    print("\n" + "=" * 60)
    print("推奨カメラパラメータ")
    print("=" * 60)
    
    # 目標: 高さ2cmで200mLになるようなfx,fy
    target_volume_m3 = 200e-6  # 200mL = 0.0002 m³
    height_m = 0.02  # 2cm
    pixels = mask.sum()
    depth_mean = depth[mask].mean()
    
    # a_pix = target_volume / (pixels * height)
    target_a_pix = target_volume_m3 / (pixels * height_m)
    
    # a_pix = Z²/(fx·fy) より
    # fx·fy = Z²/a_pix
    fx_fy_product = (depth_mean ** 2) / target_a_pix
    
    # fx = fy と仮定
    fx_recommended = np.sqrt(fx_fy_product)
    fy_recommended = fx_recommended
    
    print(f"目標: {pixels}ピクセル、高さ{height_m*100}cm → 200mL")
    print(f"必要なa_pix: {target_a_pix:.2e} m²/px")
    print(f"推奨カメラパラメータ:")
    print(f"  fx = fy = {fx_recommended:.0f}")
    
    # 検証
    K_recommended = np.array([
        [fx_recommended, 0, W/2],
        [0, fy_recommended, H/2],
        [0, 0, 1]
    ])
    
    a_pix_check = (depth_mean ** 2) / (fx_recommended * fy_recommended)
    vol_check = pixels * height_m * a_pix_check * 1e6
    print(f"\n検証: fx={fx_recommended:.0f}で計算")
    print(f"  → {vol_check:.1f} mL")

if __name__ == "__main__":
    print("UniDepth v2 カメラパラメータ調整テスト")
    print("=" * 60)
    test_with_fixed_K()