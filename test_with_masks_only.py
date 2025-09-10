#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2マスクがある画像のみで体積推定テスト
深度に基づく適応的K_scaleも検証
"""
import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import cv2
from src.unidepth_runner_final import UniDepthEngineFinal
from plane_fit import estimate_plane_from_depth
from volume_estimator import height_map_from_plane, pixel_area_map, integrate_volume

def test_with_adaptive_k(image_path, mask_paths):
    """適応的K_scaleでテスト"""
    
    engine = UniDepthEngineFinal(
        model_repo="lpiccinelli/unidepth-v2-vitl14",
        device="cuda"
    )
    
    img_name = os.path.basename(image_path)
    print(f"\n{'='*70}")
    print(f"画像: {img_name}")
    print(f"{'='*70}")
    
    # テスト設定
    test_modes = [
        ("raw", "fixed", 1.0),
        ("fixed_6", "fixed", 6.0),
        ("fixed_10.5", "fixed", 10.5),
        ("adaptive", "adaptive", None)
    ]
    
    results = []
    
    for mode_name, K_mode, K_scale in test_modes:
        print(f"\n【{mode_name}】")
        
        try:
            # UniDepth推論
            if K_mode == "adaptive":
                pred = engine.infer_image(image_path, K_mode="adaptive")
            else:
                pred = engine.infer_image(image_path, K_mode="fixed", fixed_K_scale=K_scale)
            
            depth = pred["depth"]
            K = pred["intrinsics"]
            K_scale_used = pred["K_scale_factor"]
            conf = pred["confidence"]
            
            H, W = depth.shape
            
            print(f"深度範囲: {depth.min():.2f} - {depth.max():.2f}m (中央値: {np.median(depth):.2f}m)")
            print(f"K_scale使用値: {K_scale_used:.1f}")
            print(f"調整後K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
            
            # マスク読み込み
            masks = []
            labels = []
            for mask_path in mask_paths:
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, 0) > 127
                    masks.append(mask)
                    # ラベル抽出
                    parts = os.path.basename(mask_path).split('_')
                    label = parts[2] if len(parts) > 2 else "mask"
                    labels.append(label)
            
            if not masks:
                print("マスクが見つかりません")
                continue
            
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
            
            # 体積計算
            total_volume = 0
            for mask, label in zip(masks, labels):
                vol_result = integrate_volume(
                    height, a_pix, mask,
                    conf=conf,
                    use_conf_weight=False
                )
                volume_mL = vol_result["volume_mL"]
                height_mean = vol_result["height_mean_mm"]
                total_volume += volume_mL
                
                print(f"  {label}: {volume_mL:.1f}mL (平均高さ: {height_mean:.1f}mm)")
            
            # 評価
            if 50 <= total_volume <= 1000:
                evaluation = "✓ 適切"
            elif total_volume < 50:
                evaluation = "⚠ 小さすぎ"
            elif total_volume < 2000:
                evaluation = "△ やや大きい"
            else:
                evaluation = f"✗ 異常({total_volume/1000:.1f}L)"
            
            print(f"合計体積: {total_volume:.1f}mL {evaluation}")
            
            results.append({
                "mode": mode_name,
                "K_scale": K_scale_used,
                "total_volume_mL": total_volume,
                "evaluation": evaluation
            })
            
        except Exception as e:
            print(f"エラー: {e}")
            results.append({
                "mode": mode_name,
                "error": str(e)
            })
    
    # 最適なモードを選択
    valid_results = [r for r in results if "total_volume_mL" in r and 50 <= r["total_volume_mL"] <= 1000]
    if valid_results:
        best = min(valid_results, key=lambda r: abs(r["total_volume_mL"] - 350))
        print(f"\n推奨: {best['mode']} (体積: {best['total_volume_mL']:.1f}mL)")
    
    return results

def main():
    """SAM2マスクがある画像のみテスト"""
    
    # マスクがある画像を特定
    test_cases = [
        ("test_images/train_00000.jpg", [
            "outputs/sam2/masks/train_00000_det00_rice_bplus.png",
            "outputs/sam2/masks/train_00000_det01_snow_peas_bplus.png",
            "outputs/sam2/masks/train_00000_det02_chicken_with_sauce_bplus.png"
        ]),
        ("test_images/train_00001.jpg", [
            "outputs/sam2/masks/train_00001_det00_mashed_potatoes_bplus.png",
            "outputs/sam2/masks/train_00001_det01_zucchini_slices_bplus.png",
            "outputs/sam2/masks/train_00001_det02_stewed_meat_with_tomato_sauce_bplus.png"
        ]),
        ("test_images/train_00002.jpg", [
            "outputs/sam2/masks/train_00002_det00_French_toast_bplus.png",
            "outputs/sam2/masks/train_00002_det01_powdered_sugar_bplus.png"
        ])
    ]
    
    all_results = {}
    
    for img_path, mask_paths in test_cases:
        if os.path.exists(img_path):
            results = test_with_adaptive_k(img_path, mask_paths)
            all_results[os.path.basename(img_path)] = results
    
    # 統計
    print(f"\n{'='*70}")
    print("統計分析")
    print(f"{'='*70}")
    
    mode_stats = {}
    
    for img_name, results in all_results.items():
        for r in results:
            if "total_volume_mL" in r:
                mode = r["mode"]
                if mode not in mode_stats:
                    mode_stats[mode] = {"volumes": [], "successes": 0, "total": 0}
                
                vol = r["total_volume_mL"]
                mode_stats[mode]["volumes"].append(vol)
                mode_stats[mode]["total"] += 1
                
                if 50 <= vol <= 1000:
                    mode_stats[mode]["successes"] += 1
    
    print(f"\n{'モード':<15} {'成功率':<10} {'平均体積(mL)':<15} {'中央値(mL)'}")
    print("-" * 60)
    
    for mode, stats in mode_stats.items():
        if stats["total"] > 0:
            success_rate = stats["successes"] / stats["total"] * 100
            avg_vol = np.mean(stats["volumes"])
            median_vol = np.median(stats["volumes"])
            
            print(f"{mode:<15} {success_rate:<10.1f}% {avg_vol:<15.1f} {median_vol:.1f}")
    
    # 深度別の適応的K_scale
    print(f"\n適応的K_scaleの動作:")
    for img_name, results in all_results.items():
        for r in results:
            if r["mode"] == "adaptive" and "K_scale" in r:
                print(f"  {img_name}: K_scale = {r['K_scale']:.1f}")

if __name__ == "__main__":
    main()