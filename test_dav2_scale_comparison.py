#!/usr/bin/env python3
"""
Depth Anything V2: K_scale_factorを変えて最適な体積推定を探る
"""

import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# プロジェクトのsrcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from plane_fit import estimate_plane_from_depth
from volume_estimator import height_map_from_plane, pixel_area_map, integrate_volume


def load_finetuned_model(checkpoint_dir="checkpoints/dav2_metric_n5k"):
    """Finetuning済みのDepth Anything V2モデルをロード"""
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return processor, model, device


def infer_depth(image_path, processor, model, device):
    """深度推定を実行"""
    img = Image.open(image_path)
    orig_w, orig_h = img.size
    
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        depth_map = outputs.predicted_depth
    
    if depth_map.dim() == 3:
        depth_map = depth_map.squeeze(0)
    
    depth_map = depth_map.cpu().numpy()
    
    # 元の解像度にリサイズ
    depth_map = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    return depth_map


def load_masks(base_name, target_size=(480, 640)):
    """マスクファイルを読み込む"""
    mask_dir = "outputs/sam2/masks"
    masks = []
    labels = []
    
    for fname in sorted(os.listdir(mask_dir)):
        if fname.startswith(base_name.replace('.jpg', '')) and fname.endswith('_bplus.png'):
            mask_path = os.path.join(mask_dir, fname)
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if mask_img.shape != target_size:
                mask_img = cv2.resize(mask_img, (target_size[1], target_size[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            
            mask = mask_img > 127
            masks.append(mask)
            
            parts = os.path.basename(mask_path).split('_')
            if len(parts) > 3:
                food_name = '_'.join(parts[3:-1])
            else:
                food_name = "food"
            labels.append(food_name)
    
    return masks, labels


def estimate_intrinsics(image_shape, scale_factor=1.0):
    """内部パラメータ行列Kを推定"""
    H, W = image_shape[:2]
    
    # 基本的な焦点距離の推定 (水平視野角約60度を仮定)
    fx_base = W / (2 * 0.577)
    fx = fx_base * scale_factor
    fy = fx
    
    cx = W / 2.0
    cy = H / 2.0
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K


def test_with_scale(image_path, K_scale_factor=1.0):
    """指定されたK_scale_factorで体積推定"""
    
    # モデルロード
    processor, model, device = load_finetuned_model()
    
    # 深度推定
    depth_map = infer_depth(image_path, processor, model, device)
    H, W = depth_map.shape
    
    # 内部パラメータ
    K = estimate_intrinsics(depth_map.shape, scale_factor=K_scale_factor)
    
    # マスク読み込み
    img_name = os.path.basename(image_path)
    masks, labels = load_masks(img_name, target_size=(H, W))
    
    if not masks:
        return None
    
    # テーブル平面推定
    try:
        plane_normal, plane_distance, points_xyz = estimate_plane_from_depth(
            depth_map, K, masks,
            margin_px=40,
            dist_th=0.006,
            max_iters=2000
        )
    except Exception as e:
        return None
    
    # 高さマップと面積マップ
    height_map = height_map_from_plane(points_xyz, plane_normal, plane_distance, clip_negative=True)
    area_map = pixel_area_map(depth_map, K)
    
    # 体積計算
    total_volume = 0.0
    results = []
    
    for mask, label in zip(masks, labels):
        vol_result = integrate_volume(
            height_map, area_map, mask,
            conf=None, use_conf_weight=False
        )
        volume_mL = vol_result["volume_mL"]
        total_volume += volume_mL
        results.append({
            'label': label,
            'volume_mL': volume_mL,
            'height_mean_mm': vol_result["height_mean_mm"]
        })
    
    return {
        'total_volume_mL': total_volume,
        'results': results,
        'K_scale_factor': K_scale_factor,
        'depth_range': (float(depth_map.min()), float(depth_map.max()))
    }


def main():
    """メイン処理"""
    
    test_images = [
        "test_images/train_00000.jpg",
        "test_images/train_00001.jpg"
    ]
    
    # K_scale_factorの範囲をテスト
    scale_factors = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    
    for test_image in test_images:
        if not os.path.exists(test_image):
            continue
        
        print(f"\n{'='*70}")
        print(f"画像: {test_image}")
        print(f"{'='*70}")
        print(f"{'K_scale':>10s} {'Total Volume':>15s} {'料理別体積':>40s}")
        print("-" * 70)
        
        best_result = None
        best_scale = 1.0
        best_volume = float('inf')
        target_volume = 300  # 目標体積 (mL)
        
        for scale in scale_factors:
            result = test_with_scale(test_image, K_scale_factor=scale)
            if result:
                total_vol = result['total_volume_mL']
                food_vols = ', '.join([f"{r['label'][:8]}:{r['volume_mL']:.0f}" 
                                      for r in result['results'][:3]])
                print(f"{scale:10.1f} {total_vol:15.1f}mL {food_vols:>40s}")
                
                # 100-800mLの範囲で最も妥当な値を選択
                if 100 <= total_vol <= 800:
                    if best_result is None or abs(total_vol - target_volume) < abs(best_volume - target_volume):
                        best_result = result
                        best_scale = scale
                        best_volume = total_vol
        
        if best_result:
            print("-" * 70)
            print(f"最適なK_scale_factor: {best_scale}")
            print(f"合計体積: {best_volume:.1f}mL")
            print(f"深度範囲: {best_result['depth_range'][0]:.3f} - {best_result['depth_range'][1]:.3f}m")
            
            print("\n料理別詳細:")
            for r in best_result['results']:
                print(f"  {r['label']:20s}: {r['volume_mL']:7.1f}mL (高さ平均: {r['height_mean_mm']:.1f}mm)")


if __name__ == "__main__":
    main()