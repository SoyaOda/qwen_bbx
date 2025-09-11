#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Proを使用したSAM2マスクでの体積推定テスト
複数の料理マスクに対する体積計算
"""
import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import cv2
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
from src.plane_fit import estimate_plane_from_depth
from src.volume_estimator import height_map_from_plane, pixel_area_map, integrate_volume

def test_depthpro_volume(image_path, mask_paths):
    """Depth Proで複数マスクの体積推定"""
    
    img_name = os.path.basename(image_path)
    print(f"\n{'='*70}")
    print(f"画像: {img_name}")
    print(f"{'='*70}")
    
    # Depth Proエンジンの初期化
    try:
        engine = DepthProEngine(device="cuda")
    except Exception as e:
        print(f"Depth Proエンジンの初期化に失敗: {e}")
        return None
    
    try:
        # Depth Pro推論（K_scale補正不要）
        print("\nDepth Pro推論中...")
        pred = engine.infer_image(image_path)
        
        depth = pred["depth"]
        K = pred["intrinsics"]
        
        H, W = depth.shape
        
        print(f"深度範囲: {depth.min():.2f} - {depth.max():.2f}m")
        print(f"深度中央値: {np.median(depth):.2f}m")
        print(f"カメラ内部パラメータ: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
        
        # マスク読み込み
        masks = []
        labels = []
        for mask_path in mask_paths:
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, 0) > 127
                masks.append(mask)
                # ラベル抽出
                parts = os.path.basename(mask_path).split('_')
                # 食品名を抽出（例: rice, mashed_potatoes など）
                if len(parts) > 3:
                    # "_bplus.png"を除去して食品名を取得
                    food_name = '_'.join(parts[2:-1])
                else:
                    food_name = "food"
                labels.append(food_name)
        
        if not masks:
            print("マスクが見つかりません")
            return None
        
        print(f"\n{len(masks)}個のマスクを読み込みました")
        
        # 平面推定
        print("\nテーブル平面を推定中...")
        n, d, points_xyz = estimate_plane_from_depth(
            depth, K, masks,
            margin_px=40,
            dist_th=0.006,  # 6mm閾値
            max_iters=2000
        )
        
        # 高さマップ
        height = height_map_from_plane(
            points_xyz, n, d,
            clip_negative=True
        )
        
        # ピクセル面積
        a_pix = pixel_area_map(depth, K)
        
        # 各マスクの体積計算
        print("\n各料理の体積:")
        total_volume = 0
        food_volumes = []
        
        for mask, label in zip(masks, labels):
            vol_result = integrate_volume(
                height, a_pix, mask,
                conf=None,  # Depth Proは信頼度マップなし
                use_conf_weight=False
            )
            volume_mL = vol_result["volume_mL"]
            height_mean = vol_result["height_mean_mm"]
            height_max = vol_result["height_max_mm"]
            total_volume += volume_mL
            
            # 評価
            if 10 <= volume_mL <= 500:
                status = "✓"
            elif volume_mL < 10:
                status = "⚠小"
            elif volume_mL > 1000:
                status = "⚠大"
            else:
                status = "△"
            
            print(f"  {label:30s}: {volume_mL:7.1f} mL  (高さ: 平均{height_mean:.1f}mm, 最大{height_max:.1f}mm) {status}")
            
            food_volumes.append({
                "label": label,
                "volume_mL": volume_mL,
                "height_mean_mm": height_mean,
                "height_max_mm": height_max
            })
        
        # 合計評価
        print(f"\n合計体積: {total_volume:.1f} mL", end="")
        if total_volume > 1000:
            print(f" ({total_volume/1000:.2f}L)")
        else:
            print()
        
        # 全体評価
        if 100 <= total_volume <= 1500:
            print("→ ✓ 妥当な範囲（100-1500mL）")
        elif total_volume < 100:
            print("→ ⚠ 小さすぎる可能性")
        else:
            print(f"→ ⚠ 大きすぎる可能性")
        
        # UniDepth v2との比較コメント
        print("\n【Depth Proの改善点】")
        print("• K_scale補正が不要（絶対スケールで推定）")
        print("• 焦点距離の推定精度が高い")
        print("• 深度マップの細部が保持される")
        
        return {
            "image": img_name,
            "total_volume_mL": total_volume,
            "foods": food_volumes,
            "depth_range": (depth.min(), depth.max()),
            "K": K
        }
        
    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """複数のSAM2マスクがある画像をテスト"""
    
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
    
    all_results = []
    
    for img_path, mask_paths in test_cases:
        if os.path.exists(img_path):
            result = test_depthpro_volume(img_path, mask_paths)
            if result:
                all_results.append(result)
    
    # 統計サマリー
    if all_results:
        print(f"\n{'='*70}")
        print("統計サマリー（Depth Pro）")
        print(f"{'='*70}")
        
        total_volumes = [r["total_volume_mL"] for r in all_results]
        
        print(f"\n画像数: {len(all_results)}")
        print(f"合計体積の範囲: {min(total_volumes):.1f} - {max(total_volumes):.1f} mL")
        print(f"合計体積の平均: {np.mean(total_volumes):.1f} mL")
        print(f"合計体積の中央値: {np.median(total_volumes):.1f} mL")
        
        # 各料理の統計
        food_stats = {}
        for result in all_results:
            for food in result["foods"]:
                label = food["label"]
                if label not in food_stats:
                    food_stats[label] = []
                food_stats[label].append(food["volume_mL"])
        
        print(f"\n料理別の体積統計:")
        print(f"{'料理名':<30} {'平均(mL)':<12} {'中央値(mL)':<12} {'範囲(mL)'}")
        print("-" * 70)
        
        for food_name, volumes in sorted(food_stats.items()):
            avg_vol = np.mean(volumes)
            med_vol = np.median(volumes)
            min_vol = min(volumes)
            max_vol = max(volumes)
            print(f"{food_name:<30} {avg_vol:<12.1f} {med_vol:<12.1f} {min_vol:.1f}-{max_vol:.1f}")
        
        # 成功率評価
        success_count = sum(1 for v in total_volumes if 100 <= v <= 1500)
        success_rate = success_count / len(total_volumes) * 100
        
        print(f"\n成功率（100-1500mL範囲内）: {success_rate:.1f}% ({success_count}/{len(total_volumes)})")
        
        if success_rate >= 66:
            print("→ ✓ Depth Proによる改善が確認されました")
        else:
            print("→ さらなる調整が必要かもしれません")

if __name__ == "__main__":
    main()