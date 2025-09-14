#!/usr/bin/env python3
"""
Depth Anything V2のFinetuning前後での体積推定比較テスト
Nutrition5kデータセットでの改善効果を定量評価
"""

import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import json
from datetime import datetime

# プロジェクトのsrcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from volume_estimator import pixel_area_map, integrate_volume
from plane_fit import estimate_plane_from_depth, build_support_ring
from volume_estimator import height_map_from_plane


def load_pretrained_model():
    """オリジナルのDepth Anything V2 Metricモデルをロード"""
    print("Loading original Depth Anything V2 (Metric-Indoor-Large) model...")
    
    processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    
    return processor, model, device


def load_finetuned_model(checkpoint_dir="checkpoints/dav2_metric_n5k"):
    """Finetuning済みのDepth Anything V2モデルをロード"""
    print(f"Loading finetuned model from {checkpoint_dir}...")
    
    processor = AutoImageProcessor.from_pretrained(checkpoint_dir)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint_dir)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    
    return processor, model, device


def infer_depth(image_path, processor, model, device):
    """深度推定を実行
    
    Returns:
        depth_map: 深度マップ (H, W) in meters
    """
    # 画像読み込み
    img = Image.open(image_path)
    orig_w, orig_h = img.size
    
    # 前処理
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 推論
    with torch.no_grad():
        outputs = model(**inputs)
        depth_map = outputs.predicted_depth
    
    # (1, H, W) or (H, W) -> (H, W)
    if depth_map.dim() == 3:
        depth_map = depth_map.squeeze(0)
    
    # numpy配列に変換
    depth_map = depth_map.cpu().numpy()
    
    # 元の解像度にリサイズ
    depth_map = cv2.resize(depth_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    return depth_map


def load_segmentation_mask(dish_id):
    """QwenVLで生成済みのセグメンテーションマスクをロード"""
    base_dir = "nutrition5k/nutrition5k_dataset/imagery/realsense_overhead"
    mask_path = os.path.join(base_dir, f"dish_{dish_id}", "qwen_mask.png")
    
    if not os.path.exists(mask_path):
        print(f"  Warning: Mask not found at {mask_path}")
        return None
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return mask > 127  # バイナリマスクに変換


def estimate_volume_from_depth(depth_map, mask, K):
    """深度マップとマスクから体積を推定
    
    Args:
        depth_map: 深度マップ (H, W) in meters
        mask: セグメンテーションマスク (H, W) bool
        K: カメラ内部パラメータ (3, 3)
    """
    
    # マスク内の深度値のみ抽出
    depth_masked = depth_map[mask]
    
    if len(depth_masked) == 0:
        return 0.0, {}
    
    # テーブル平面を推定（本来の関数を使用）
    try:
        # 単一マスクをリストとして渡す
        plane_normal, plane_distance, points_xyz = estimate_plane_from_depth(
            depth_map, K, [mask],
            margin_px=40,
            dist_th=0.006,
            max_iters=500  # 高速化のため削減
        )
    except Exception as e:
        # フォールバック：簡易推定
        print(f"    Plane estimation error: {e}")
        median_depth = np.median(depth_masked)
        plane_normal = np.array([0, 0, 1])
        plane_distance = -median_depth
        # 簡易3D点群
        H, W = depth_map.shape
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        us = np.arange(W)
        vs = np.arange(H)
        uu, vv = np.meshgrid(us, vs)
        Z = depth_map
        X = (uu - cx) / fx * Z
        Y = (vv - cy) / fy * Z
        points_xyz = np.stack([X, Y, Z], axis=0)
    
    # 高さマップ計算（本来の関数を使用）
    height_map = height_map_from_plane(points_xyz, plane_normal, plane_distance, clip_negative=True)
    
    # マスク内の高さ
    heights_masked = height_map[mask]
    
    # 負の高さ（テーブルより下）をゼロに
    heights_masked = np.maximum(heights_masked, 0)
    
    # ピクセル面積マップを正しく計算
    px_area_map = pixel_area_map(depth_map, K)
    
    # 体積積分
    volume_result = integrate_volume(height_map, px_area_map, mask)
    # integrate_volumeは辞書を返すので、体積値を取り出す
    if isinstance(volume_result, dict):
        volume_ml = volume_result.get('volume_mL', 0.0)
    else:
        volume_ml = volume_result * 1e6  # 古い形式の場合（m³ -> ml）
    
    stats = {
        'depth_min': float(depth_masked.min()),
        'depth_max': float(depth_masked.max()),
        'depth_mean': float(depth_masked.mean()),
        'height_mean': float(heights_masked.mean()),
        'height_max': float(heights_masked.max()),
        'num_pixels': int(mask.sum()),
        'plane_params': [float(plane_normal[0]), float(plane_normal[1]), float(plane_normal[2]), float(plane_distance)]
    }
    
    return volume_ml, stats


def compare_models_on_dish(dish_id, pretrained_model, finetuned_model, device):
    """1つの料理画像で両モデルを比較"""
    
    # パス設定
    base_dir = "nutrition5k/nutrition5k_dataset/imagery/realsense_overhead"
    dish_dir = os.path.join(base_dir, f"dish_{dish_id}")
    rgb_path = os.path.join(dish_dir, "rgb.png")
    depth_gt_path = os.path.join(dish_dir, "depth_raw.png")
    
    if not os.path.exists(rgb_path):
        return None
    
    print(f"\nProcessing dish_{dish_id}")
    
    # マスクをロード（なければ中央領域を使用）
    mask = load_segmentation_mask(dish_id)
    if mask is None:
        print(f"  Warning: No mask available, using center region")
        # 画像中央の小さい領域をマスクとして使用（テスト用）
        H, W = 640, 640  # 標準的なサイズ
        mask = np.zeros((H, W), dtype=bool)
        # 中央の200x200ピクセル領域を使用
        cy, cx = H // 2, W // 2
        size = 100
        mask[cy-size:cy+size, cx-size:cx+size] = True
    
    # Ground truth深度をロード（比較用）
    depth_gt_raw = cv2.imread(depth_gt_path, cv2.IMREAD_UNCHANGED)
    depth_gt = depth_gt_raw.astype(np.float32) * 1e-4  # meters
    
    # Pretrainedモデルで推定
    processor_pre, model_pre, _ = pretrained_model
    depth_pre = infer_depth(rgb_path, processor_pre, model_pre, device)
    
    # Finetunedモデルで推定
    processor_ft, model_ft, _ = finetuned_model
    depth_ft = infer_depth(rgb_path, processor_ft, model_ft, device)
    
    # カメラ内部パラメータを推定（Nutrition5kの標準的な値を使用）
    H, W = depth_gt.shape
    # Nutrition5kのRealSenseカメラの標準的な内部パラメータ
    fx = fy = W * 1.2  # 焦点距離（ピクセル単位）
    cx = W / 2.0
    cy = H / 2.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # 体積推定
    volume_pre, stats_pre = estimate_volume_from_depth(depth_pre, mask, K)
    volume_ft, stats_ft = estimate_volume_from_depth(depth_ft, mask, K)
    volume_gt, stats_gt = estimate_volume_from_depth(depth_gt, mask, K)
    
    # エラー計算（GT深度との比較）
    mask_indices = mask > 0
    
    # Pretrained model errors
    depth_error_pre = np.abs(depth_pre[mask_indices] - depth_gt[mask_indices])
    absrel_pre = np.mean(depth_error_pre / np.maximum(depth_gt[mask_indices], 1e-6))
    rmse_pre = np.sqrt(np.mean(depth_error_pre**2))
    
    # Finetuned model errors
    depth_error_ft = np.abs(depth_ft[mask_indices] - depth_gt[mask_indices])
    absrel_ft = np.mean(depth_error_ft / np.maximum(depth_gt[mask_indices], 1e-6))
    rmse_ft = np.sqrt(np.mean(depth_error_ft**2))
    
    result = {
        'dish_id': dish_id,
        'pretrained': {
            'volume_ml': volume_pre,
            'absrel': float(absrel_pre),
            'rmse_m': float(rmse_pre),
            'stats': stats_pre
        },
        'finetuned': {
            'volume_ml': volume_ft,
            'absrel': float(absrel_ft),
            'rmse_m': float(rmse_ft),
            'stats': stats_ft
        },
        'ground_truth': {
            'volume_ml': volume_gt,
            'stats': stats_gt
        },
        'improvements': {
            'absrel_reduction': float((absrel_pre - absrel_ft) / absrel_pre * 100) if absrel_pre > 0 else 0,
            'rmse_reduction': float((rmse_pre - rmse_ft) / rmse_pre * 100) if rmse_pre > 0 else 0,
            'volume_error_pre': float(abs(volume_pre - volume_gt)),
            'volume_error_ft': float(abs(volume_ft - volume_gt)),
            'volume_error_reduction': float((abs(volume_pre - volume_gt) - abs(volume_ft - volume_gt)) / abs(volume_pre - volume_gt) * 100) if abs(volume_pre - volume_gt) > 0 else 0
        }
    }
    
    print(f"  Pretrained - AbsRel: {absrel_pre:.4f}, RMSE: {rmse_pre:.4f}m, Volume: {volume_pre:.1f}ml")
    print(f"  Finetuned  - AbsRel: {absrel_ft:.4f}, RMSE: {rmse_ft:.4f}m, Volume: {volume_ft:.1f}ml")
    print(f"  Ground Truth Volume: {volume_gt:.1f}ml")
    print(f"  Improvements: AbsRel↓{result['improvements']['absrel_reduction']:.1f}%, RMSE↓{result['improvements']['rmse_reduction']:.1f}%")
    
    return result


def main():
    print("=" * 70)
    print("Depth Anything V2: Pretrained vs Finetuned Comparison")
    print("=" * 70)
    
    # モデルロード
    pretrained_model = load_pretrained_model()
    finetuned_model = load_finetuned_model()
    device = pretrained_model[2]
    
    # テスト用のdish IDリスト（実際に存在するディッシュ）
    test_dish_ids = [
        '1556572657',  # 最初のディッシュ
        '1556573514',  # 2番目のディッシュ
        '1556575014',  # 3番目のディッシュ
    ]
    
    results = []
    
    for dish_id in test_dish_ids:
        result = compare_models_on_dish(
            dish_id, 
            pretrained_model, 
            finetuned_model, 
            device
        )
        if result:
            results.append(result)
    
    # 統計サマリー
    if results:
        print("\n" + "=" * 70)
        print("Overall Statistics:")
        print("=" * 70)
        
        avg_absrel_reduction = np.mean([r['improvements']['absrel_reduction'] for r in results])
        avg_rmse_reduction = np.mean([r['improvements']['rmse_reduction'] for r in results])
        avg_volume_error_reduction = np.mean([r['improvements']['volume_error_reduction'] for r in results if r['improvements']['volume_error_reduction'] != float('inf')])
        
        print(f"Average AbsRel Reduction: {avg_absrel_reduction:.1f}%")
        print(f"Average RMSE Reduction: {avg_rmse_reduction:.1f}%")
        print(f"Average Volume Error Reduction: {avg_volume_error_reduction:.1f}%")
        
        # 結果をJSONファイルに保存
        output_file = f"dav2_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'test_dish_ids': test_dish_ids,
                'results': results,
                'summary': {
                    'avg_absrel_reduction': avg_absrel_reduction,
                    'avg_rmse_reduction': avg_rmse_reduction,
                    'avg_volume_error_reduction': avg_volume_error_reduction
                }
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    print("\n" + "=" * 70)
    print("Comparison completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()