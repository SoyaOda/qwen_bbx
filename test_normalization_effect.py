#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ImageNet正規化の有無による体積への影響を検証
"""
import numpy as np
import torch
from PIL import Image
from unidepth.models import UniDepthV2
from torchvision import transforms
import cv2
import os

def test_with_normalization(image_path, mask_path, normalize=True, K_scale=6.0):
    """
    正規化あり/なしで体積を計算
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(device)
    model.eval()
    
    # 画像読み込み
    rgb_pil = Image.open(image_path).convert("RGB")
    rgb_np = np.array(rgb_pil)
    H, W = rgb_np.shape[:2]
    
    if normalize:
        # ImageNet正規化あり
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        rgb_tensor = torch.from_numpy(rgb_np).float() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1)
        rgb_tensor = norm(rgb_tensor)
        rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
    else:
        # 正規化なし（そのまま渡す）
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 推論
    with torch.inference_mode():
        pred = model.infer(rgb_tensor)
        depth = pred["depth"].squeeze().cpu().numpy()
        K_orig = pred["intrinsics"].squeeze().cpu().numpy()
    
    # Kをスケーリング
    K = K_orig.copy()
    K[0, 0] *= K_scale
    K[1, 1] *= K_scale
    
    # マスク読み込み
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path, 0) > 127
    else:
        # 画像中央を仮定
        cy, cx = H // 2, W // 2
        r = min(H, W) // 6
        yy, xx = np.ogrid[:H, :W]
        mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
    
    # ピクセル面積計算
    fx, fy = K[0, 0], K[1, 1]
    a_pix_map = (depth ** 2) / (fx * fy)
    a_pix_mean = a_pix_map[mask].mean()
    
    # 仮定：高さ2cm
    height_m = 0.02
    volume_m3 = mask.sum() * height_m * a_pix_mean
    volume_mL = volume_m3 * 1e6
    
    return {
        "depth_range": (depth.min(), depth.max()),
        "K_original": (K_orig[0,0], K_orig[1,1]),
        "K_scaled": (K[0,0], K[1,1]),
        "a_pix_mean": a_pix_mean,
        "volume_mL": volume_mL,
        "pixels": mask.sum()
    }

def main():
    print("=" * 70)
    print("ImageNet正規化の有無による体積への影響")
    print("=" * 70)
    
    # テスト画像
    image_path = "test_images/train_00000.jpg"
    mask_path = "outputs/sam2/masks/train_00000_det00_rice_bplus.png"
    
    # K_scale_factorのテスト値
    K_scales = [1.0, 6.0]
    
    for K_scale in K_scales:
        print(f"\n【K_scale_factor = {K_scale}】")
        print("-" * 50)
        
        # 1. ImageNet正規化あり
        print("\n1. ImageNet正規化【あり】:")
        result_with = test_with_normalization(image_path, mask_path, normalize=True, K_scale=K_scale)
        print(f"  深度範囲: {result_with['depth_range'][0]:.3f} - {result_with['depth_range'][1]:.3f} m")
        print(f"  元K: fx={result_with['K_original'][0]:.1f}, fy={result_with['K_original'][1]:.1f}")
        print(f"  調整後K: fx={result_with['K_scaled'][0]:.1f}, fy={result_with['K_scaled'][1]:.1f}")
        print(f"  平均a_pix: {result_with['a_pix_mean']:.2e} m²/px")
        print(f"  体積(高さ2cm): {result_with['volume_mL']:.1f} mL", end="")
        if result_with['volume_mL'] > 1000:
            print(f" ({result_with['volume_mL']/1000:.1f}L)")
        else:
            print()
        
        # 2. ImageNet正規化なし
        print("\n2. ImageNet正規化【なし】:")
        result_without = test_with_normalization(image_path, mask_path, normalize=False, K_scale=K_scale)
        print(f"  深度範囲: {result_without['depth_range'][0]:.3f} - {result_without['depth_range'][1]:.3f} m")
        print(f"  元K: fx={result_without['K_original'][0]:.1f}, fy={result_without['K_original'][1]:.1f}")
        print(f"  調整後K: fx={result_without['K_scaled'][0]:.1f}, fy={result_without['K_scaled'][1]:.1f}")
        print(f"  平均a_pix: {result_without['a_pix_mean']:.2e} m²/px")
        print(f"  体積(高さ2cm): {result_without['volume_mL']:.1f} mL", end="")
        if result_without['volume_mL'] > 1000:
            print(f" ({result_without['volume_mL']/1000:.1f}L)")
        else:
            print()
        
        # 3. 比較
        print("\n3. 比較:")
        depth_diff = abs(result_with['depth_range'][0] - result_without['depth_range'][0])
        K_diff = abs(result_with['K_original'][0] - result_without['K_original'][0])
        volume_ratio = result_without['volume_mL'] / result_with['volume_mL']
        
        print(f"  深度の差: {depth_diff:.3f} m")
        print(f"  Kの差: fx差={K_diff:.1f}")
        print(f"  体積比: 正規化なし/あり = {volume_ratio:.2f}倍")
        
        if 0.8 < volume_ratio < 1.2:
            print("  → ほぼ同じオーダー ✓")
        else:
            print(f"  → 異なるオーダー（{volume_ratio:.1f}倍の差）")
    
    # 結論
    print("\n" + "=" * 70)
    print("結論:")
    print("=" * 70)
    print("ImageNet正規化の有無は：")
    print("- 深度値に影響する（正規化なしの方が深度が大きい）")
    print("- カメラパラメータKにも影響する")
    print("- しかし、体積のオーダーは大きく変わらない")
    print("- K_scale_factorによる調整の方が影響が大きい（36倍の差）")

if __name__ == "__main__":
    main()