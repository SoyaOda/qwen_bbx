#!/usr/bin/env python3
"""
Depth Anything V2 (Metric)を使用した料理の体積推定テスト
仕様書 (md_files/DA_v2_spec.md) に基づいた実装
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


def load_depth_anything_model():
    """Depth Anything V2 Metricモデルをロード"""
    print("Depth Anything V2 (Metric-Indoor-Large) モデルをロード中...")
    
    # Transformers経由でモデルとプロセッサをロード
    processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    )
    
    # GPUが利用可能ならCUDAを使用
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")
    model = model.to(device).eval()
    
    return processor, model, device


def infer_depth_anything(image_path, processor, model, device):
    """Depth Anything V2で画像から深度マップ推定
    
    Returns:
        depth_map: 深度マップ (H, W) in meters
    """
    # 画像読み込み
    img = Image.open(image_path)
    orig_w, orig_h = img.size  # PIL: size=(W,H)
    
    # プロセッサで前処理
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 推論実行
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 深度マップを取得
    # outputsの構造を確認
    if hasattr(outputs, 'predicted_depth'):
        pred_depth = outputs.predicted_depth
    else:
        # 別の形式の場合
        pred_depth = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    
    # shapeを確認して適切に処理
    if len(pred_depth.shape) == 3:  # [batch, H, W]の場合
        pred_depth = pred_depth.unsqueeze(1)  # [batch, 1, H, W]に変換
    elif len(pred_depth.shape) == 2:  # [H, W]の場合
        pred_depth = pred_depth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]に変換
    
    # 元画像サイズにリサイズ
    if pred_depth.shape[-2:] != (orig_h, orig_w):
        pred_depth = torch.nn.functional.interpolate(
            pred_depth, size=(orig_h, orig_w),
            mode="bicubic", align_corners=False
        )
    
    # numpy配列に変換 (H, W) in meters
    if len(pred_depth.shape) == 4:
        depth_map = pred_depth[0, 0].cpu().numpy()
    else:
        depth_map = pred_depth.squeeze().cpu().numpy()
    
    return depth_map


def load_masks(base_name):
    """指定された画像のマスクファイルを読み込む"""
    mask_dir = "outputs/sam2/masks"
    masks = []
    labels = []
    mask_paths = []
    
    # マスクファイルを検索
    for fname in os.listdir(mask_dir):
        if fname.startswith(base_name.replace('.jpg', '')) and fname.endswith('_bplus.png'):
            mask_path = os.path.join(mask_dir, fname)
            mask_paths.append(mask_path)
    
    # ソートして順番を保持
    mask_paths.sort()
    
    for mask_path in mask_paths:
        if os.path.exists(mask_path):
            # マスク画像を読み込み
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask_img > 127
            masks.append(mask)
            
            # ファイル名から料理名を抽出
            # 例: train_00000_det00_rice_bplus.png -> "rice"
            parts = os.path.basename(mask_path).split('_')
            if len(parts) > 3:
                # det番号を除いて料理名を抽出
                food_name = '_'.join(parts[3:-1])  # -1はbplusを除外
            else:
                food_name = "food"
            labels.append(food_name)
            print(f"  マスク読み込み: {food_name}")
    
    return masks, labels


def estimate_intrinsics(image_shape, scale_factor=10.5):
    """画像サイズから内部パラメータ行列Kを推定
    
    iPhone等のスマートフォンカメラを想定した典型的な値を使用
    scale_factor: UniDepth v2の経験から、約10.5倍のスケーリングが必要
    """
    H, W = image_shape[:2]
    
    # 基本的な焦点距離の推定 (水平視野角約60度を仮定)
    # fx = W / (2 * tan(FOV_h/2))
    # FOV_h = 60度の場合、tan(30度) ≈ 0.577
    fx_base = W / (2 * 0.577)
    
    # スケーリングファクターを適用
    # UniDepth v2やDepth Anything V2の経験から、
    # 実際の焦点距離は基本推定値の約10倍程度が必要
    fx = fx_base * scale_factor
    fy = fx  # 正方形ピクセルを仮定
    
    # 主点は画像中心
    cx = W / 2.0
    cy = H / 2.0
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K


def test_volume_estimation(image_path, processor, model, device, K_scale_factor=10.5):
    """単一画像で体積推定テストを実行"""
    
    img_name = os.path.basename(image_path)
    print(f"\n{'='*70}")
    print(f"画像: {img_name}")
    print(f"{'='*70}")
    
    # 深度推定
    print("\nDepth Anything V2で深度推定中...")
    depth_map = infer_depth_anything(image_path, processor, model, device)
    H, W = depth_map.shape
    print(f"深度マップサイズ: {W}x{H}")
    print(f"深度範囲: {depth_map.min():.3f} - {depth_map.max():.3f} m")
    
    # 内部パラメータ推定
    K = estimate_intrinsics(depth_map.shape, scale_factor=K_scale_factor)
    print(f"\n内部パラメータ:")
    print(f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"  cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    
    # マスク読み込み
    print("\nマスクを読み込み中...")
    masks, labels = load_masks(img_name)
    
    if not masks:
        print("マスクが見つかりません")
        return None
    
    print(f"{len(masks)}個のマスクを読み込みました")
    
    # テーブル平面推定
    print("\nテーブル平面を推定中...")
    try:
        plane_normal, plane_distance, points_xyz = estimate_plane_from_depth(
            depth_map, K, masks,
            margin_px=40,
            dist_th=0.006,  # 6mm
            max_iters=2000
        )
        print(f"平面法線: [{plane_normal[0]:.3f}, {plane_normal[1]:.3f}, {plane_normal[2]:.3f}]")
        print(f"平面距離: {plane_distance:.3f} m")
    except Exception as e:
        print(f"平面推定エラー: {e}")
        return None
    
    # 高さマップ計算
    print("\n高さマップを計算中...")
    height_map = height_map_from_plane(points_xyz, plane_normal, plane_distance, clip_negative=True)
    
    # ピクセル面積マップ計算
    area_map = pixel_area_map(depth_map, K)
    
    # 各マスクの体積計算
    print("\n料理ごとの体積を計算中...")
    print("-" * 60)
    
    total_volume = 0.0
    results = []
    
    for mask, label in zip(masks, labels):
        # 体積計算 (confidence重み付けなし)
        vol_result = integrate_volume(
            height_map, area_map, mask,
            conf=None, use_conf_weight=False
        )
        
        volume_mL = vol_result["volume_mL"]
        height_mean = vol_result["height_mean_mm"]
        height_max = vol_result["height_max_mm"]
        
        total_volume += volume_mL
        
        # 適切性評価
        if 10 <= volume_mL <= 1000:
            status = "✓"
        elif volume_mL < 10:
            status = "⚠小"
        elif volume_mL > 1500:
            status = "⚠大"
        else:
            status = "△"
        
        # 結果表示
        print(f"  {label:30s}: {volume_mL:7.1f} mL  "
              f"(高さ: 平均{height_mean:.1f}mm, 最大{height_max:.1f}mm) {status}")
        
        results.append({
            'label': label,
            'volume_mL': volume_mL,
            'height_mean_mm': height_mean,
            'height_max_mm': height_max
        })
    
    # 合計体積の表示
    print("-" * 60)
    print(f"合計体積: {total_volume:.1f} mL", end="")
    if total_volume > 1000:
        print(f"  ({total_volume/1000:.2f} L)")
    else:
        print("")
    
    # 合計体積の評価
    if 100 <= total_volume <= 1500:
        print("→ ✓ 妥当な範囲（100-1500 mL）")
    elif total_volume < 100:
        print("→ ⚠ 小さすぎる可能性")
    else:
        print("→ ⚠ 大きすぎる可能性")
    
    return {
        'image': img_name,
        'total_volume_mL': total_volume,
        'results': results
    }


def main():
    """メイン処理"""
    
    # モデルロード
    processor, model, device = load_depth_anything_model()
    print("モデルロード完了\n")
    
    # テスト画像（train_00000.jpg）で実行
    test_image = "test_images/train_00000.jpg"
    
    if not os.path.exists(test_image):
        print(f"エラー: テスト画像が見つかりません: {test_image}")
        return
    
    # 異なるK_scale_factorでテスト
    print("\n" + "="*70)
    print("K_scale_factorの調整テスト")
    print("="*70)
    
    # まず、デフォルト値（過大推定の問題確認）
    print("\n[1] K_scale_factor = 1.0 (デフォルト、問題あり)")
    result1 = test_volume_estimation(test_image, processor, model, device, K_scale_factor=1.0)
    
    # 次に、UniDepth v2の経験値を適用
    print("\n" + "="*70)
    print("\n[2] K_scale_factor = 10.5 (UniDepth v2の経験値)")
    result2 = test_volume_estimation(test_image, processor, model, device, K_scale_factor=10.5)
    
    # 結果の比較
    if result1 and result2:
        print(f"\n{'='*70}")
        print("結果比較")
        print(f"{'='*70}")
        print(f"K_scale_factor = 1.0:  {result1['total_volume_mL']:.1f} mL")
        print(f"K_scale_factor = 10.5: {result2['total_volume_mL']:.1f} mL")
        print(f"体積比: {result1['total_volume_mL'] / result2['total_volume_mL']:.1f}倍")
        
        print("\n※ Depth Anything V2 (Metric)による絶対尺度推定")
        print("※ 内部パラメータKの適切な設定により現実的な体積を算出")


if __name__ == "__main__":
    main()