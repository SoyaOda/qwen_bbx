#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
体積計算のデバッグスクリプト
各ステップでの値を詳細に確認
"""
import numpy as np
import torch
from PIL import Image
from unidepth.models import UniDepthV2
from torchvision import transforms
import os

def debug_volume_calculation():
    """体積計算の各ステップをデバッグ"""
    
    # テスト画像
    image_path = "test_images/train_00000.jpg"
    
    # 1) モデルロード
    print("=" * 60)
    print("1. モデルロード")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(device)
    model.eval()
    print(f"デバイス: {device}")
    
    # 2) 画像読み込みと前処理
    print("\n" + "=" * 60)
    print("2. 画像前処理")
    print("=" * 60)
    rgb_pil = Image.open(image_path).convert("RGB")
    rgb_np = np.array(rgb_pil)
    H_orig, W_orig = rgb_np.shape[:2]
    print(f"元画像サイズ: {W_orig}x{H_orig}")
    
    # ImageNet正規化あり
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    rgb_tensor = torch.from_numpy(rgb_np).float() / 255.0
    rgb_tensor = rgb_tensor.permute(2, 0, 1)
    rgb_tensor_normalized = normalize(rgb_tensor)
    rgb_tensor_normalized = rgb_tensor_normalized.unsqueeze(0).to(device)
    
    # ImageNet正規化なし（比較用）
    rgb_tensor_raw = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 3) 推論（両方のケース）
    print("\n" + "=" * 60)
    print("3. 深度推定")
    print("=" * 60)
    
    with torch.inference_mode():
        # 正規化あり
        pred_normalized = model.infer(rgb_tensor_normalized)
        depth_normalized = pred_normalized["depth"].squeeze().cpu().numpy()
        K_normalized = pred_normalized["intrinsics"].squeeze().cpu().numpy()
        
        # 正規化なし
        pred_raw = model.infer(rgb_tensor_raw)
        depth_raw = pred_raw["depth"].squeeze().cpu().numpy()
        K_raw = pred_raw["intrinsics"].squeeze().cpu().numpy()
    
    print("【ImageNet正規化あり】")
    print(f"  深度範囲: {depth_normalized.min():.3f} - {depth_normalized.max():.3f} m")
    print(f"  fx={K_normalized[0,0]:.1f}, fy={K_normalized[1,1]:.1f}")
    print(f"  cx={K_normalized[0,2]:.1f}, cy={K_normalized[1,2]:.1f}")
    
    print("\n【ImageNet正規化なし】")
    print(f"  深度範囲: {depth_raw.min():.3f} - {depth_raw.max():.3f} m")
    print(f"  fx={K_raw[0,0]:.1f}, fy={K_raw[1,1]:.1f}")
    print(f"  cx={K_raw[0,2]:.1f}, cy={K_raw[1,2]:.1f}")
    
    # 4) ピクセル面積計算
    print("\n" + "=" * 60)
    print("4. ピクセル面積計算")
    print("=" * 60)
    
    # 正規化ありの場合
    fx, fy = K_normalized[0,0], K_normalized[1,1]
    depth_center = depth_normalized[H_orig//2, W_orig//2]
    a_pix_center = (depth_center ** 2) / (fx * fy)
    
    print(f"画像中心での計算（正規化あり）:")
    print(f"  深度 Z = {depth_center:.3f} m")
    print(f"  fx = {fx:.1f}, fy = {fy:.1f}")
    print(f"  a_pix = Z²/(fx·fy) = {depth_center:.3f}² / ({fx:.1f}·{fy:.1f})")
    print(f"       = {a_pix_center:.2e} m²/px")
    
    # 5) 仮想的な体積計算
    print("\n" + "=" * 60)
    print("5. 仮想体積計算")
    print("=" * 60)
    
    # 100x100ピクセル、高さ5cmの食品を仮定
    pixels = 100 * 100
    height_m = 0.05  # 5cm
    
    volume_m3 = pixels * height_m * a_pix_center
    volume_mL = volume_m3 * 1e6  # m³ → mL
    
    print(f"仮定: {pixels}ピクセル、高さ{height_m*1000:.0f}mm")
    print(f"  体積 = {pixels} × {height_m} × {a_pix_center:.2e}")
    print(f"      = {volume_m3:.2e} m³")
    print(f"      = {volume_mL:.1f} mL")
    
    if volume_mL > 1000:
        print(f"  ⚠️ 異常: {volume_mL/1000:.1f}L（期待値の約{volume_mL/200:.0f}倍）")
    else:
        print(f"  ✓ 正常範囲")
    
    # 6) 実際のマスクでの計算（あれば）
    print("\n" + "=" * 60)
    print("6. 実マスクでの体積")
    print("=" * 60)
    
    mask_path = "outputs/sam2/masks/train_00000_det00_rice_bplus.png"
    if os.path.exists(mask_path):
        import cv2
        mask = cv2.imread(mask_path, 0) > 127
        mask_pixels = mask.sum()
        
        # 全ピクセルのa_pixを計算
        a_pix_map = (depth_normalized ** 2) / (fx * fy)
        
        # マスク内の平均a_pix
        a_pix_mean = a_pix_map[mask].mean()
        
        # 仮に高さ2cmとして
        height_assumed = 0.02  # 2cm
        volume_real = mask_pixels * height_assumed * a_pix_mean
        volume_real_mL = volume_real * 1e6
        
        print(f"マスク: rice")
        print(f"  ピクセル数: {mask_pixels}")
        print(f"  平均a_pix: {a_pix_mean:.2e} m²/px")
        print(f"  仮定高さ: {height_assumed*1000:.0f}mm")
        print(f"  推定体積: {volume_real_mL:.1f} mL")
        
        if volume_real_mL > 1000:
            print(f"  ⚠️ 異常: {volume_real_mL/1000:.1f}L")
            
            # 逆算：200mLになるためのスケール
            target_mL = 200
            scale_factor = target_mL / volume_real_mL
            print(f"\n  【逆算】200mLにするには:")
            print(f"    - 現在の{scale_factor:.3f}倍にスケール")
            print(f"    - またはfx,fyを{1/np.sqrt(scale_factor):.1f}倍に")
            suggested_fx = fx / np.sqrt(scale_factor)
            suggested_fy = fy / np.sqrt(scale_factor)
            print(f"    - 推奨: fx={suggested_fx:.0f}, fy={suggested_fy:.0f}")
    
    # 7) depth_modelとの比較
    print("\n" + "=" * 60)
    print("7. depth_modelプロジェクトとの比較")
    print("=" * 60)
    
    depth_model_path = "/home/soya/depth_model/outputs_test_images/train_00000/UniDepth_depth.npy"
    if os.path.exists(depth_model_path):
        depth_ref = np.load(depth_model_path)
        print(f"depth_model深度範囲: {depth_ref.min():.3f} - {depth_ref.max():.3f} m")
        
        # 差分
        if depth_ref.shape == depth_normalized.shape:
            diff = np.abs(depth_normalized - depth_ref)
            print(f"深度差分: 平均{diff.mean():.6f}m, 最大{diff.max():.6f}m")
            if diff.mean() < 1e-6:
                print("  ✓ 深度は一致")
            else:
                print("  ⚠️ 深度が異なる")
        else:
            print(f"  サイズ不一致: {depth_normalized.shape} vs {depth_ref.shape}")

if __name__ == "__main__":
    debug_volume_calculation()