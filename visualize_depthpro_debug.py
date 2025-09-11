#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Pro デバッグ用可視化スクリプト
深度マップ、高さマップ、マスク、PLYファイルを出力
"""
import sys
import os
sys.path.insert(0, 'src')
sys.path.insert(0, 'ml-depth-pro/src')

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm

# Depth Proエンジン
from src.depthpro_runner import DepthProEngine
from src.plane_fit_depthpro import estimate_table_plane
from src.volume_depthpro import pixel_area_map, height_from_plane, integrate_volume

def save_ply(filename, points, colors=None):
    """
    3D点群をPLYファイルとして保存
    
    Args:
        filename: 保存先ファイル名
        points: 3D点群 (N, 3)
        colors: RGB色 (N, 3) [0-255]
    """
    with open(filename, 'w') as f:
        # ヘッダー
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # データ
        for i, p in enumerate(points):
            if colors is not None:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}\n")
            else:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

def visualize_depth(depth, filename, title="Depth Map"):
    """深度マップの可視化"""
    plt.figure(figsize=(10, 8))
    
    # カラーマップ適用（viridis）
    im = plt.imshow(depth, cmap='viridis')
    plt.colorbar(im, label='Depth [m]')
    plt.title(title)
    
    # 統計情報を追加
    plt.text(0.02, 0.98, f"Min: {depth.min():.3f}m\nMax: {depth.max():.3f}m\nMedian: {np.median(depth):.3f}m",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  深度マップを保存: {filename}")

def visualize_height(height, mask, filename, title="Height Map"):
    """高さマップの可視化"""
    plt.figure(figsize=(12, 5))
    
    # サブプロット1: 全体の高さマップ
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(height * 1000, cmap='jet', vmin=0)  # mmに変換
    plt.colorbar(im1, label='Height [mm]')
    plt.title(f"{title} - Full")
    
    # サブプロット2: マスク内のみ
    plt.subplot(1, 2, 2)
    height_masked = np.where(mask, height * 1000, np.nan)
    im2 = plt.imshow(height_masked, cmap='jet', vmin=0)
    plt.colorbar(im2, label='Height [mm]')
    plt.title(f"{title} - Masked")
    
    # 統計情報
    h_in_mask = height[mask] * 1000  # mm
    if len(h_in_mask) > 0:
        stats_text = (f"Mean: {h_in_mask.mean():.2f}mm\n"
                     f"Median: {np.median(h_in_mask):.2f}mm\n"
                     f"Max: {h_in_mask.max():.2f}mm\n"
                     f"Positive pixels: {(h_in_mask > 0).sum()}/{len(h_in_mask)}")
        plt.text(1.15, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  高さマップを保存: {filename}")

def analyze_mask(mask_path, image_path):
    """マスクの精度を分析"""
    # マスクと元画像を読み込み
    mask = cv2.imread(mask_path, 0) > 127
    image = cv2.imread(image_path)
    
    # マスクの統計
    H, W = mask.shape
    total_pixels = H * W
    mask_pixels = mask.sum()
    
    print(f"\nマスク分析: {os.path.basename(mask_path)}")
    print(f"  画像サイズ: {W}x{H}")
    print(f"  マスクピクセル: {mask_pixels} ({100*mask_pixels/total_pixels:.1f}%)")
    
    # マスクの外接矩形
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        print(f"  外接矩形: ({x},{y}) - ({x+w},{y+h}), サイズ: {w}x{h}")
        print(f"  充填率: {100*mask_pixels/(w*h):.1f}%")
    
    # マスクのオーバーレイ画像を保存
    overlay = image.copy()
    overlay[mask] = overlay[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
    
    output_path = mask_path.replace('.png', '_overlay.jpg')
    cv2.imwrite(output_path, overlay)
    print(f"  オーバーレイ画像を保存: {output_path}")
    
    return mask

def debug_volume_estimation(image_path: str, mask_path: str):
    """詳細なデバッグ情報付き体積推定"""
    
    print("=" * 70)
    print("Depth Pro デバッグ可視化")
    print("=" * 70)
    
    # 出力ディレクトリ作成
    output_dir = "debug_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. マスク分析
    mask = analyze_mask(mask_path, image_path)
    
    # 2. Depth Pro推論
    print("\nDepth Pro推論...")
    engine = DepthProEngine(device="cuda")
    result = engine.infer_image(image_path)
    
    depth = result["depth"]
    K = result["intrinsics"]
    xyz = result["points"]
    H, W = depth.shape
    
    # 3. 深度マップ可視化
    visualize_depth(depth, f"{output_dir}/depth_map.png")
    
    # 4. 平面推定
    print("\n平面推定...")
    try:
        n, d = estimate_table_plane(depth, K, mask, verbose=True)
    except Exception as e:
        print(f"  エラー: {e}")
        n = np.array([0, 0, 1])
        d = np.median(depth)
    
    # 5. 高さマップ計算と可視化
    print("\n高さマップ計算...")
    height = height_from_plane(depth, K, n, d, clip_negative=True)
    visualize_height(height, mask, f"{output_dir}/height_map.png")
    
    # 高さマップの詳細分析
    print("\n高さマップ分析:")
    h_all = height.flatten()
    h_mask = height[mask]
    
    print(f"  全体:")
    print(f"    範囲: {h_all.min()*1000:.2f} - {h_all.max()*1000:.2f} mm")
    print(f"    ゼロピクセル: {(h_all == 0).sum()} / {len(h_all)} ({100*(h_all == 0).sum()/len(h_all):.1f}%)")
    print(f"    正のピクセル: {(h_all > 0).sum()} / {len(h_all)} ({100*(h_all > 0).sum()/len(h_all):.1f}%)")
    
    print(f"  マスク内:")
    print(f"    範囲: {h_mask.min()*1000:.2f} - {h_mask.max()*1000:.2f} mm")
    print(f"    ゼロピクセル: {(h_mask == 0).sum()} / {len(h_mask)} ({100*(h_mask == 0).sum()/len(h_mask):.1f}%)")
    print(f"    正のピクセル: {(h_mask > 0).sum()} / {len(h_mask)} ({100*(h_mask > 0).sum()/len(h_mask):.1f}%)")
    
    if (h_mask > 0).sum() > 0:
        h_positive = h_mask[h_mask > 0]
        print(f"    正の値の統計: 平均={h_positive.mean()*1000:.2f}mm, 中央値={np.median(h_positive)*1000:.2f}mm")
    
    # 6. PLYファイル出力
    print("\nPLYファイル生成...")
    
    # 全点群
    points_all = xyz.reshape(-1, 3)
    colors_all = np.tile([128, 128, 128], (len(points_all), 1))  # グレー
    
    # マスク内の点を赤に
    mask_flat = mask.flatten()
    colors_all[mask_flat] = [255, 0, 0]
    
    # 高さが正の点を色分け
    height_flat = height.flatten()
    positive_mask = height_flat > 0
    
    # 高さに応じた色（青→緑→黄→赤）
    for i in np.where(positive_mask)[0]:
        h_norm = np.clip(height_flat[i] / 0.05, 0, 1)  # 50mmで正規化
        rgba = np.array(cm.jet(h_norm))  # RGBAをnumpy配列に変換
        rgb = (rgba[:3] * 255).astype(np.int32)  # RGBのみ取り出して整数化
        colors_all[i] = rgb
    
    save_ply(f"{output_dir}/pointcloud_all.ply", points_all, colors_all)
    
    # マスク内のみ
    points_mask = xyz[mask]
    height_mask_values = height[mask]
    colors_mask = np.zeros((len(points_mask), 3), dtype=int)
    
    for i, h in enumerate(height_mask_values):
        h_norm = np.clip(h / 0.05, 0, 1)
        rgba = np.array(cm.jet(h_norm))  # RGBAをnumpy配列に変換
        rgb = (rgba[:3] * 255).astype(np.int32)  # RGBのみ取り出して整数化
        colors_mask[i] = rgb
    
    save_ply(f"{output_dir}/pointcloud_masked.ply", points_mask, colors_mask)
    
    # 7. 体積計算
    print("\n体積計算...")
    a_pix = pixel_area_map(depth, K)
    vol_result = integrate_volume(height, a_pix, mask)
    
    print(f"  体積: {vol_result['volume_mL']:.1f} mL")
    print(f"  平均高さ: {vol_result['height_mean_mm']:.1f} mm")
    print(f"  最大高さ: {vol_result['height_max_mm']:.1f} mm")
    
    # 8. 平面の可視化
    print("\n平面情報:")
    print(f"  法線: n=[{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")
    print(f"  d: {d:.3f}")
    print(f"  水平からの傾き: {np.rad2deg(np.arccos(abs(n[2]))):.1f}度")
    
    # テーブル面の点を生成して可視化
    xx, yy = np.meshgrid(np.linspace(-0.2, 0.2, 50), np.linspace(-0.2, 0.2, 50))
    zz = (d - n[0]*xx - n[1]*yy) / n[2]
    plane_points = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)
    plane_colors = np.tile([0, 255, 0], (len(plane_points), 1))  # 緑
    save_ply(f"{output_dir}/plane.ply", plane_points, plane_colors)
    
    print("\n" + "=" * 70)
    print(f"デバッグファイルは {output_dir}/ に保存されました")
    print("=" * 70)

def main():
    # テストケース
    image_path = "test_images/train_00000.jpg"
    mask_path = "outputs/sam2/masks/train_00000_det00_rice_bplus.png"
    
    if os.path.exists(image_path) and os.path.exists(mask_path):
        debug_volume_estimation(image_path, mask_path)
    else:
        print(f"ファイルが見つかりません:")
        print(f"  画像: {image_path} - {'存在' if os.path.exists(image_path) else '不在'}")
        print(f"  マスク: {mask_path} - {'存在' if os.path.exists(mask_path) else '不在'}")

if __name__ == "__main__":
    main()