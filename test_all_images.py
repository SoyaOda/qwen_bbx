#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全test_images画像で体積推定の汎用性をテスト
各画像に対して異なるK_scale_factorを試して最適値を分析
"""
import sys
import os
import glob
sys.path.insert(0, 'src')

import numpy as np
import cv2
from PIL import Image
from unidepth_runner import UniDepthEngine
from plane_fit import estimate_plane_from_depth
from volume_estimator import height_map_from_plane, pixel_area_map, integrate_volume
from vis_depth import apply_colormap
import json

def test_single_image(image_path, mask_paths, K_scale_factors=[1.0, 6.0, 8.0, 10.5, 12.0, 15.0]):
    """
    単一画像で複数のK_scale_factorをテスト
    """
    # UniDepth推論
    engine = UniDepthEngine(
        model_repo="lpiccinelli/unidepth-v2-vitl14",
        device="cuda"
    )
    
    # 画像名
    img_name = os.path.basename(image_path)
    
    results = {
        "image": img_name,
        "tests": []
    }
    
    print(f"\n{'='*70}")
    print(f"画像: {img_name}")
    print(f"{'='*70}")
    
    # 基本推論（K_scale=1.0で深度と元のKを取得）
    pred_base = engine.infer_image(image_path, K_scale_factor=1.0)
    depth = pred_base["depth"]
    K_orig = pred_base["intrinsics_original"]
    conf = pred_base["confidence"]
    
    H, W = depth.shape
    
    # 深度情報
    depth_stats = {
        "min": float(depth.min()),
        "max": float(depth.max()),
        "median": float(np.median(depth))
    }
    
    print(f"深度: {depth_stats['min']:.2f} - {depth_stats['max']:.2f}m (中央値: {depth_stats['median']:.2f}m)")
    print(f"元K: fx={K_orig[0,0]:.1f}, fy={K_orig[1,1]:.1f}")
    
    # マスクを読み込み
    masks = []
    labels = []
    for mask_path in mask_paths:
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0) > 127
            masks.append(mask)
            # ラベルをファイル名から抽出
            label = os.path.basename(mask_path).split('_')[2]  # det00_label_source.png
            labels.append(label)
    
    if not masks:
        # マスクがない場合は画像中央を使用
        cy, cx = H // 2, W // 2
        r = min(H, W) // 6
        yy, xx = np.ogrid[:H, :W]
        mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
        masks = [mask]
        labels = ["center_region"]
    
    print(f"マスク数: {len(masks)} ({', '.join(labels)})")
    
    # 各K_scale_factorでテスト
    print(f"\n{'K_scale':<10} {'体積(mL)':<40} {'評価':<10}")
    print("-" * 70)
    
    for K_scale in K_scale_factors:
        # Kを調整
        K = K_orig.copy()
        K[0, 0] *= K_scale
        K[1, 1] *= K_scale
        
        try:
            # 平面推定
            n, d, points_xyz = estimate_plane_from_depth(
                depth, K, masks,
                margin_px=40,
                dist_th=0.006,
                max_iters=2000
            )
            
            # 高さマップ
            height = height_map_from_plane(
                points_xyz, n, d,
                clip_negative=True
            )
            
            # ピクセル面積
            a_pix = pixel_area_map(depth, K)
            
            # 各マスクの体積を計算
            volumes = []
            for mask, label in zip(masks, labels):
                vol_result = integrate_volume(
                    height, a_pix, mask,
                    conf=conf,
                    use_conf_weight=False
                )
                volumes.append({
                    "label": label,
                    "volume_mL": vol_result["volume_mL"],
                    "height_mean_mm": vol_result["height_mean_mm"],
                    "height_max_mm": vol_result["height_max_mm"]
                })
            
            # 合計体積
            total_volume = sum(v["volume_mL"] for v in volumes)
            
            # 評価
            if 50 <= total_volume <= 1000:
                evaluation = "✓ 適切"
            elif total_volume < 50:
                evaluation = "⚠ 小さい"
            elif total_volume < 2000:
                evaluation = "△ やや大"
            else:
                evaluation = f"✗ 異常({total_volume/1000:.1f}L)"
            
            # 結果表示
            volume_str = ", ".join([f"{v['label'][:8]}:{v['volume_mL']:.0f}" for v in volumes])
            print(f"{K_scale:<10.1f} {volume_str:<40} {evaluation}")
            
            # 結果保存
            results["tests"].append({
                "K_scale": K_scale,
                "total_volume_mL": total_volume,
                "volumes": volumes,
                "evaluation": evaluation
            })
            
        except Exception as e:
            print(f"{K_scale:<10.1f} エラー: {str(e)[:40]}")
            results["tests"].append({
                "K_scale": K_scale,
                "error": str(e)
            })
    
    # 最適なK_scaleを推定
    valid_tests = [t for t in results["tests"] if "total_volume_mL" in t]
    if valid_tests:
        # 200-500mLの範囲に最も近いものを選択
        target_range = (200, 500)
        best = min(valid_tests, key=lambda t: 
                  abs(t["total_volume_mL"] - np.mean(target_range)))
        
        results["optimal_K_scale"] = best["K_scale"]
        results["optimal_volume_mL"] = best["total_volume_mL"]
        
        print(f"\n推奨K_scale: {best['K_scale']:.1f} (体積: {best['total_volume_mL']:.1f}mL)")
    
    results["depth_stats"] = depth_stats
    results["K_original"] = {
        "fx": float(K_orig[0, 0]),
        "fy": float(K_orig[1, 1])
    }
    
    return results

def main():
    """全test_images画像をテスト"""
    
    # test_images内の画像を取得
    test_images = sorted(glob.glob("test_images/*.jpg"))
    if not test_images:
        test_images = sorted(glob.glob("test_images/*.png"))
    
    print(f"テスト画像数: {len(test_images)}")
    
    # 各画像に対応するマスクを検索
    all_results = []
    
    for img_path in test_images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        
        # SAM2マスクを検索（複数ある場合）
        mask_pattern = f"outputs/sam2/masks/{stem}_det*_*_bplus.png"
        mask_paths = sorted(glob.glob(mask_pattern))
        
        if not mask_paths:
            # largeマスクも試す
            mask_pattern = f"outputs/sam2/masks/{stem}_det*_*_large.png"
            mask_paths = sorted(glob.glob(mask_pattern))
        
        # テスト実行
        result = test_single_image(img_path, mask_paths)
        all_results.append(result)
    
    # 統計分析
    print(f"\n{'='*70}")
    print("統計分析")
    print(f"{'='*70}")
    
    # 各K_scaleでの成功率を計算
    k_scale_stats = {}
    for k in [1.0, 6.0, 8.0, 10.5, 12.0, 15.0]:
        successes = 0
        total = 0
        volumes = []
        
        for result in all_results:
            for test in result["tests"]:
                if test.get("K_scale") == k and "total_volume_mL" in test:
                    total += 1
                    vol = test["total_volume_mL"]
                    volumes.append(vol)
                    if 50 <= vol <= 1000:
                        successes += 1
        
        if total > 0:
            success_rate = successes / total * 100
            avg_volume = np.mean(volumes) if volumes else 0
            
            k_scale_stats[k] = {
                "success_rate": success_rate,
                "avg_volume_mL": avg_volume,
                "count": total
            }
    
    print(f"\n{'K_scale':<10} {'成功率':<10} {'平均体積(mL)':<15} {'サンプル数'}")
    print("-" * 50)
    for k, stats in sorted(k_scale_stats.items()):
        print(f"{k:<10.1f} {stats['success_rate']:<10.1f}% {stats['avg_volume_mL']:<15.1f} {stats['count']}")
    
    # 最適なK_scaleの分布
    optimal_k_values = [r.get("optimal_K_scale", 0) for r in all_results if "optimal_K_scale" in r]
    if optimal_k_values:
        print(f"\n最適K_scaleの分布:")
        print(f"  平均: {np.mean(optimal_k_values):.1f}")
        print(f"  中央値: {np.median(optimal_k_values):.1f}")
        print(f"  範囲: {min(optimal_k_values):.1f} - {max(optimal_k_values):.1f}")
    
    # 深度と最適K_scaleの相関
    print(f"\n深度と最適K_scaleの関係:")
    depth_vs_k = []
    for r in all_results:
        if "optimal_K_scale" in r and "depth_stats" in r:
            depth_vs_k.append({
                "image": r["image"],
                "depth_median": r["depth_stats"]["median"],
                "optimal_k": r["optimal_K_scale"]
            })
    
    if depth_vs_k:
        depth_vs_k.sort(key=lambda x: x["depth_median"])
        for item in depth_vs_k:
            print(f"  {item['image']:<20} 深度中央値: {item['depth_median']:.2f}m → K_scale: {item['optimal_k']:.1f}")
    
    # 結果をJSONで保存
    with open("test_all_images_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果を test_all_images_results.json に保存しました")
    
    # 推奨値
    print(f"\n{'='*70}")
    print("推奨事項")
    print(f"{'='*70}")
    
    best_k = max(k_scale_stats.items(), key=lambda x: x[1]["success_rate"])[0]
    print(f"最も汎用性の高いK_scale_factor: {best_k}")
    print(f"成功率: {k_scale_stats[best_k]['success_rate']:.1f}%")
    print(f"平均体積: {k_scale_stats[best_k]['avg_volume_mL']:.1f}mL")

if __name__ == "__main__":
    main()