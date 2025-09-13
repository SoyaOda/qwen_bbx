#!/usr/bin/env python3
"""
Nutrition5kデータセットでDepth Anything V2を使用した体積推定テスト
事前生成されたSAM2.1マスクを使用
"""

import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image
from pathlib import Path
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# UniDepth用のパスとインポート
sys.path.append(os.path.join(os.path.dirname(__file__), 'UniDepth'))
try:
    from unidepth.models import UniDepthV2
    UNIDEPTH_AVAILABLE = True
except ImportError:
    print("警告: UniDepth v2が利用できません。UniDepthディレクトリを確認してください。")
    UNIDEPTH_AVAILABLE = False

# プロジェクトのsrcディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from plane_fit import estimate_plane_from_depth
from volume_estimator import height_map_from_plane, pixel_area_map, integrate_volume


def load_unidepth_model():
    """UniDepth v2モデルをロード"""
    if not UNIDEPTH_AVAILABLE:
        return None, None
    
    print("UniDepth v2 モデルをロード中...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")
    
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(device)
    model.eval()
    
    return model, device


def infer_unidepth(image_path, model, device):
    """UniDepth v2で画像から深度マップとK行列を推定"""
    if model is None:
        return None, None
    
    # 画像読み込み（正規化はモデル内部で実施）
    rgb_pil = Image.open(image_path).convert("RGB")
    rgb_np = np.array(rgb_pil)
    
    # RGBテンソルに変換（uint8のまま）
    rgb = torch.from_numpy(rgb_np).permute(2,0,1).unsqueeze(0).to(device)
    
    # 推論実行（Kは推定させる）
    with torch.inference_mode():
        pred = model.infer(rgb)  # 正規化はモデル内部で行われる
    
    # 深度とKを取得
    depth = pred["depth"].squeeze().detach().cpu().numpy()   # [m]
    K = pred["intrinsics"].squeeze().detach().cpu().numpy()  # (3,3)
    
    return depth, K


def load_depth_anything_model():
    """Depth Anything V2 Metricモデルをロード"""
    print("Depth Anything V2 (Metric-Indoor-Large) モデルをロード中...")
    
    processor = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    )
    model = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用デバイス: {device}")
    model = model.to(device).eval()
    
    return processor, model, device


def infer_depth_anything(image_path, processor, model, device):
    """Depth Anything V2で画像から深度マップ推定"""
    img = Image.open(image_path)
    orig_w, orig_h = img.size
    
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    if hasattr(outputs, 'predicted_depth'):
        pred_depth = outputs.predicted_depth
    else:
        pred_depth = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    
    if len(pred_depth.shape) == 3:
        pred_depth = pred_depth.unsqueeze(1)
    elif len(pred_depth.shape) == 2:
        pred_depth = pred_depth.unsqueeze(0).unsqueeze(0)
    
    if pred_depth.shape[-2:] != (orig_h, orig_w):
        pred_depth = torch.nn.functional.interpolate(
            pred_depth, size=(orig_h, orig_w),
            mode="bicubic", align_corners=False
        )
    
    if len(pred_depth.shape) == 4:
        depth_map = pred_depth[0, 0].cpu().numpy()
    else:
        depth_map = pred_depth.squeeze().cpu().numpy()
    
    return depth_map


def load_nutrition5k_gt_depth(dish_id):
    """Nutrition5kのGT深度マップを読み込み"""
    depth_path = f"nutrition5k/nutrition5k_dataset/imagery/realsense_overhead/{dish_id}/depth_raw.png"
    
    if not os.path.exists(depth_path):
        return None
    
    # 16bit深度画像を読み込み
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    # Nutrition5kのGT深度は848x480、test_imagesは640x480にリサイズ済み
    # 640x480にリサイズ
    if depth_raw.shape != (480, 640):
        depth_raw = cv2.resize(depth_raw, (640, 480), interpolation=cv2.INTER_NEAREST)
    
    # 深度値の単位を判定（mm or 0.1mm）
    depth_float = depth_raw.astype(np.float32)
    valid_mask = depth_raw > 0
    if valid_mask.any():
        median_val = np.median(depth_raw[valid_mask])
        print(f"  GT深度の中央値: {median_val:.1f}")
        if median_val < 1000:
            # mm単位と推定
            depth_m = depth_float * 0.001
            print(f"  単位: mm → メートルに変換")
        else:
            # 0.1mm単位と推定  
            depth_m = depth_float / 10000.0
            print(f"  単位: 0.1mm → メートルに変換")
    else:
        depth_m = depth_float * 0.001
    
    return depth_m


def load_masks(base_name):
    """事前生成されたSAM2.1マスクを読み込む"""
    mask_dir = "outputs/sam2/masks"
    masks = []
    labels = []
    mask_paths = []
    
    # bplusマスクファイルを検索
    for fname in os.listdir(mask_dir):
        if fname.startswith(base_name) and fname.endswith('_bplus.png'):
            mask_path = os.path.join(mask_dir, fname)
            mask_paths.append(mask_path)
    
    mask_paths.sort()
    
    for mask_path in mask_paths:
        if os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # マスクが異なるサイズの場合、リサイズ
            if mask_img.shape != (480, 640):
                mask_img = cv2.resize(mask_img, (640, 480), interpolation=cv2.INTER_NEAREST)
            mask = mask_img > 127
            masks.append(mask)
            
            # ファイル名から料理名を抽出
            parts = os.path.basename(mask_path).split('_')
            if len(parts) > 3:
                food_name = '_'.join(parts[3:-1])  # -1はbplusを除外
            else:
                food_name = "food"
            labels.append(food_name)
            print(f"  マスク読み込み: {food_name}")
    
    return masks, labels


def get_nutrition5k_K_matrix():
    """Nutrition5kの内部パラメータ行列Kを取得"""
    # 論文から: RealSense D415, 元は848x480解像度
    # test_imagesは640x480にリサイズ済みなので、それに合わせて調整
    W, H = 640, 480
    
    # 元の解像度でのパラメータ
    W_orig = 848
    hfov_deg = 65.0
    hfov_rad = np.deg2rad(hfov_deg)
    fx_orig = W_orig / (2 * np.tan(hfov_rad / 2))
    
    # リサイズ比率に合わせて調整
    scale = W / W_orig
    fx = fx_orig * scale
    fy = fx  # 正方形ピクセルを仮定
    cx = W / 2.0
    cy = H / 2.0
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K


def process_nutrition5k_sample(sample_id, da_processor, da_model, da_device, 
                               uni_model=None, uni_device=None):
    """Nutrition5kサンプルを処理（複数の深度推定手法を比較）"""
    print(f"\n{'='*60}")
    print(f"処理中: {sample_id}")
    print(f"{'='*60}")
    
    # Nutrition5kのdish IDを取得
    dish_mapping = {
        'train_00000': 'dish_1556572657',
        'train_00001': 'dish_1556573514',
        'train_00002': 'dish_1556575014',
        'train_00003': 'dish_1556575083',
        'train_00004': 'dish_1556575124',
        'train_00005': 'dish_1556575273',
        'train_00006': 'dish_1556575327',
        'train_00007': 'dish_1556575386',
        'train_00008': 'dish_1556575446',
        'train_00009': 'dish_1556575499',
    }
    
    dish_id = dish_mapping.get(sample_id)
    if not dish_id:
        print(f"  警告: {sample_id}のdish IDが見つかりません")
        return None
    
    # RGB画像パス
    image_path = f"test_images/{sample_id}.jpg"
    if not os.path.exists(image_path):
        print(f"  エラー: 画像ファイルが見つかりません: {image_path}")
        return None
    
    # マスクを読み込み
    print("マスクを読み込み中...")
    masks, labels = load_masks(sample_id)
    if not masks:
        print("  警告: マスクが見つかりません")
        return None
    
    # GT深度マップを読み込み
    print("GT深度マップを読み込み中...")
    gt_depth = load_nutrition5k_gt_depth(dish_id)
    
    # Depth Anything V2で深度推定
    print("Depth Anything V2で深度推定中...")
    da_depth = infer_depth_anything(image_path, da_processor, da_model, da_device)
    
    # UniDepth v2で深度推定
    uni_depth = None
    uni_K = None
    if uni_model is not None:
        print("UniDepth v2で深度推定中...")
        uni_depth, uni_K = infer_unidepth(image_path, uni_model, uni_device)
    
    # K行列を取得
    K = get_nutrition5k_K_matrix()
    # Nutrition5kは実際のカメラパラメータがあるので、スケーリングを小さくする
    K_scale_factor = 1.0  # GT深度の場合はスケーリング不要
    K_scaled_gt = K.copy()
    K_scaled_gt[0, 0] *= K_scale_factor
    K_scaled_gt[1, 1] *= K_scale_factor
    
    # Depth Anything V2の場合は別のスケール係数
    K_scale_factor_da = 3.0  # Depth Anything V2用のスケール調整
    K_scaled_da = K.copy()
    K_scaled_da[0, 0] *= K_scale_factor_da
    K_scaled_da[1, 1] *= K_scale_factor_da
    
    # UniDepth v2は推定されたKを使用
    K_scale_factor_uni = 10.5  # UniDepth v2の経験値
    if uni_K is not None:
        K_scaled_uni = uni_K.copy()
        K_scaled_uni[0, 0] *= K_scale_factor_uni
        K_scaled_uni[1, 1] *= K_scale_factor_uni
    else:
        K_scaled_uni = None
    
    results = {}
    
    # GT深度での体積計算
    if gt_depth is not None:
        print("\n--- GT深度マップを使用した体積計算 ---")
        results['gt'] = calculate_volumes(gt_depth, K_scaled_gt, masks, labels)
    
    # 予測深度での体積計算
    print("\n--- Depth Anything V2予測深度を使用した体積計算 ---")
    results['da_v2'] = calculate_volumes(da_depth, K_scaled_da, masks, labels)
    
    # UniDepth v2での体積計算
    if uni_depth is not None:
        print("\n--- UniDepth v2予測深度を使用した体積計算 ---")
        results['unidepth'] = calculate_volumes(uni_depth, K_scaled_uni, masks, labels)
    
    # 結果の比較
    if 'gt' in results and ('da_v2' in results or 'unidepth' in results):
        print("\n--- 体積比較 ---")
        for i, label in enumerate(labels):
            gt_vol = results['gt']['volumes'][i]
            print(f"  {label:30s}:")
            print(f"    GT: {gt_vol:7.1f}mL")
            
            if 'da_v2' in results:
                da_vol = results['da_v2']['volumes'][i]
                da_error = abs(da_vol - gt_vol) / gt_vol * 100 if gt_vol > 0 else 0
                print(f"    Depth Anything V2: {da_vol:7.1f}mL (誤差={da_error:5.1f}%)")
            
            if 'unidepth' in results:
                uni_vol = results['unidepth']['volumes'][i]
                uni_error = abs(uni_vol - gt_vol) / gt_vol * 100 if gt_vol > 0 else 0
                print(f"    UniDepth v2: {uni_vol:7.1f}mL (誤差={uni_error:5.1f}%)")
    
    return results


def calculate_volumes(depth_map, K, masks, labels):
    """深度マップとマスクから体積を計算"""
    H, W = depth_map.shape
    print(f"  深度マップサイズ: {H}x{W}")
    if masks:
        print(f"  マスクサイズ: {masks[0].shape}")
    
    # 平面推定（points_xyzも取得）
    try:
        plane_normal, plane_distance, points_xyz = estimate_plane_from_depth(
            depth_map, K, masks,
            margin_px=40,
            dist_th=0.006,
            max_iters=2000,
            min_support=2000
        )
    except Exception as e:
        print(f"平面推定エラー: {e}")
        return {'volumes': [0.0] * len(masks), 'error': str(e)}
    
    # 高さマップとピクセル面積マップ
    # points_xyzは既に(3, H, W)形式なのでそのまま使用
    height_map = height_map_from_plane(points_xyz, plane_normal, plane_distance, clip_negative=True)
    area_map = pixel_area_map(depth_map, K)
    
    print(f"  height_map形状: {height_map.shape}, area_map形状: {area_map.shape}")
    
    # 各マスクの体積計算
    volumes = []
    for mask, label in zip(masks, labels):
        vol_result = integrate_volume(
            height_map, area_map, mask,
            conf=None, use_conf_weight=False
        )
        volume_mL = vol_result["volume_mL"]
        volumes.append(volume_mL)
        print(f"  {label:30s}: {volume_mL:7.1f} mL")
    
    return {'volumes': volumes, 'labels': labels}


def main():
    """メイン処理"""
    # Depth Anything V2モデルをロード
    da_processor, da_model, da_device = load_depth_anything_model()
    
    # UniDepth v2モデルをロード
    uni_model, uni_device = load_unidepth_model()
    
    # テストするサンプル
    samples = ['train_00000', 'train_00001', 'train_00002']
    
    all_results = {}
    for sample_id in samples:
        results = process_nutrition5k_sample(sample_id, da_processor, da_model, da_device,
                                              uni_model, uni_device)
        if results:
            all_results[sample_id] = results
    
    # 全体の統計を表示
    if all_results:
        print("\n" + "="*60)
        print("全体統計")
        print("="*60)
        
        da_errors = []
        uni_errors = []
        
        for sample_id, results in all_results.items():
            if 'gt' in results:
                for i in range(len(results['gt']['volumes'])):
                    gt_vol = results['gt']['volumes'][i]
                    
                    if 'da_v2' in results and gt_vol > 0:
                        da_vol = results['da_v2']['volumes'][i]
                        da_error = abs(da_vol - gt_vol) / gt_vol * 100
                        da_errors.append(da_error)
                    
                    if 'unidepth' in results and gt_vol > 0:
                        uni_vol = results['unidepth']['volumes'][i]
                        uni_error = abs(uni_vol - gt_vol) / gt_vol * 100
                        uni_errors.append(uni_error)
        
        if da_errors:
            print("\nDepth Anything V2統計:")
            print(f"  平均相対誤差: {np.mean(da_errors):.1f}%")
            print(f"  中央値相対誤差: {np.median(da_errors):.1f}%")
            print(f"  最小誤差: {np.min(da_errors):.1f}%")
            print(f"  最大誤差: {np.max(da_errors):.1f}%")
        
        if uni_errors:
            print("\nUniDepth v2統計:")
            print(f"  平均相対誤差: {np.mean(uni_errors):.1f}%")
            print(f"  中央値相対誤差: {np.median(uni_errors):.1f}%")
            print(f"  最小誤差: {np.min(uni_errors):.1f}%")
            print(f"  最大誤差: {np.max(uni_errors):.1f}%")


if __name__ == "__main__":
    main()