#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Pro最適化版のテスト
水平性事前分布と焦点距離最適化を使用した改善版
"""
import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'ml-depth-pro/src')

import numpy as np
import cv2
from PIL import Image

# Depth Proエンジン
from src.depthpro_runner import DepthProEngine, read_f35_from_exif

# 新しいモジュール
from src.plane_fit_depthpro import estimate_table_plane
from src.volume_depthpro import pixel_area_map, height_from_plane, integrate_volume, sanity_check_volume
from src.fpx_refine import refine_fpx_by_flatness, validate_fpx

def test_volume_estimation_optimized(image_path: str, mask_path: str = None):
    """Depth Pro最適化版での体積推定"""
    
    print("=" * 70)
    print("Depth Pro最適化版テスト（水平性事前分布 + 焦点距離最適化）")
    print("=" * 70)
    
    # 1. Depth Proエンジンの初期化
    print("\n1. Depth Proエンジンの初期化...")
    try:
        engine = DepthProEngine(device="cuda")
    except Exception as e:
        print(f"Depth Proエンジンの初期化に失敗: {e}")
        return
    
    # 2. 第1パス: 初回推論（EXIFまたはモデル推定）
    print("\n2. 第1パス: 初回推論...")
    result1 = engine.infer_image(image_path)
    
    depth = result1["depth"]
    K = result1["intrinsics"]
    fx = result1["fx"]
    fy = result1["fy"]
    fpx_pred = result1.get("fpx_pred")
    f35 = result1.get("f35mm")
    H, W = depth.shape
    
    print(f"\n初回推論結果:")
    print(f"  深度範囲: {depth.min():.3f} - {depth.max():.3f} m")
    print(f"  深度中央値: {np.median(depth):.3f} m")
    print(f"  焦点距離: fx={fx:.1f}, fy={fy:.1f} pixels")
    
    # 焦点距離の検証
    is_valid, msg = validate_fpx(fx, fy, (H, W), verbose=True)
    
    # 3. マスク読み込み
    print("\n3. マスク読み込み...")
    if mask_path and os.path.exists(mask_path):
        mask = cv2.imread(mask_path, 0) > 127
        print(f"  マスク: {os.path.basename(mask_path)}")
    else:
        # テスト用：画像中央を食品と仮定
        cy, cx = H // 2, W // 2
        r = min(H, W) // 6
        yy, xx = np.ogrid[:H, :W]
        mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
        print("  テスト用中央円マスクを使用")
    
    print(f"  マスクピクセル数: {mask.sum()}")
    
    # 4. 焦点距離の最適化（EXIFがない場合のみ）
    need_refinement = (f35 is None and fpx_pred is not None)
    
    if need_refinement:
        print("\n4. 焦点距離の最適化...")
        print("  EXIFがないため、平面の水平性で焦点距離を最適化します")
        
        # 焦点距離の探索
        best_fx = refine_fpx_by_flatness(
            depth, K, mask, 
            fpx0=fpx_pred, 
            size_hw=(H, W),
            try_scales=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5],
            verbose=True
        )
        
        # 最適化された焦点距離で再推論
        if abs(best_fx - fpx_pred) > 10:  # 有意な変化がある場合のみ
            print(f"\n  再推論: fx={fpx_pred:.1f} → {best_fx:.1f}")
            result2 = engine.infer_image(image_path, force_fpx=best_fx)
            depth = result2["depth"]
            K = result2["intrinsics"]
            fx = result2["fx"]
            fy = result2["fy"]
            print(f"  再推論後の深度範囲: {depth.min():.3f} - {depth.max():.3f} m")
    else:
        if f35 is not None:
            print("\n4. 焦点距離の最適化をスキップ（EXIFあり）")
        else:
            print("\n4. 焦点距離の最適化をスキップ")
    
    # 5. 堅牢な平面推定（水平性事前分布）
    print("\n5. テーブル平面推定（水平性強化版）...")
    try:
        n, d = estimate_table_plane(
            depth, K, mask,
            z_med=np.median(depth),
            z_grad_thr=0.004,
            horiz_deg=15.0,  # 15度以内を優先
            verbose=True
        )
    except Exception as e:
        print(f"  平面推定エラー: {e}")
        print("  デフォルト水平面を使用")
        n = np.array([0, 0, 1])
        d = np.median(depth)
    
    # 6. 高さマップと体積計算
    print("\n6. 体積計算...")
    
    # 高さマップ
    height = height_from_plane(depth, K, n, d, clip_negative=True)
    
    # 画素面積
    a_pix = pixel_area_map(depth, K)
    
    # 体積積分
    vol_result = integrate_volume(
        height, a_pix, mask,
        conf=None,
        use_conf_weight=False
    )
    
    volume_mL = vol_result["volume_mL"]
    height_mean = vol_result["height_mean_mm"]
    height_max = vol_result["height_max_mm"]
    height_median = vol_result["height_median_mm"]
    area_cm2 = vol_result["area_m2"] * 1e4
    
    print(f"\n結果:")
    print(f"  体積: {volume_mL:.1f} mL", end="")
    if volume_mL > 1000:
        print(f" ({volume_mL/1000:.2f}L)")
    else:
        print()
    
    print(f"  平均高さ: {height_mean:.1f} mm")
    print(f"  中央値高さ: {height_median:.1f} mm")
    print(f"  最大高さ: {height_max:.1f} mm")
    print(f"  投影面積: {area_cm2:.1f} cm²")
    
    # 妥当性チェック
    is_sane = sanity_check_volume(vol_result, food_type="rice", verbose=True)
    
    # 改善度の評価
    print("\n【改善ポイント】")
    print("• 水平性事前分布による平面推定の安定化")
    if need_refinement:
        print(f"• 焦点距離の最適化: {fpx_pred:.1f} → {fx:.1f} pixels")
    print("• リング領域の自動拡張")
    print("• 深度勾配によるフラット領域の選択")
    
    # デバッグ情報
    print("\n【デバッグ情報】")
    print(f"深度範囲: {depth.min():.3f} - {depth.max():.3f} m")
    print(f"K行列: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"平面法線: n=[{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")
    print(f"水平からの傾き: {np.rad2deg(np.arccos(abs(n[2]))):.1f}度")
    
    # 画素面積の統計
    a_pix_masked = a_pix[mask]
    a_pix_mean = a_pix_masked.mean()
    a_pix_median = np.median(a_pix_masked)
    print(f"画素面積: 平均={a_pix_mean*1e6:.3f}mm², 中央値={a_pix_median*1e6:.3f}mm²")
    
    print("\n" + "=" * 70)
    
    return {
        "volume_mL": volume_mL,
        "height_mean_mm": height_mean,
        "height_max_mm": height_max,
        "is_sane": is_sane,
        "n": n,
        "fx": fx,
        "fy": fy
    }

def main():
    # テスト画像
    test_cases = [
        ("test_images/train_00000.jpg", "outputs/sam2/masks/train_00000_det00_rice_bplus.png"),
        ("test_images/train_00001.jpg", "outputs/sam2/masks/train_00001_det00_mashed_potatoes_bplus.png"),
        ("test_images/train_00002.jpg", "outputs/sam2/masks/train_00002_det00_French_toast_bplus.png"),
    ]
    
    results = []
    
    for image_path, mask_path in test_cases:
        if os.path.exists(image_path):
            print(f"\n\n画像: {os.path.basename(image_path)}")
            result = test_volume_estimation_optimized(image_path, mask_path)
            if result:
                results.append(result)
            break  # 最初の1枚だけテスト
    
    # サマリー
    if results:
        print("\n" + "=" * 70)
        print("最適化版サマリー")
        print("=" * 70)
        
        for i, r in enumerate(results):
            print(f"\nケース{i+1}:")
            print(f"  体積: {r['volume_mL']:.1f} mL")
            print(f"  高さ: {r['height_mean_mm']:.1f} mm")
            print(f"  焦点距離: fx={r['fx']:.1f}")
            print(f"  水平度: {np.rad2deg(np.arccos(abs(r['n'][2]))):.1f}度")
            print(f"  妥当性: {'✓' if r['is_sane'] else '✗'}")

if __name__ == "__main__":
    main()