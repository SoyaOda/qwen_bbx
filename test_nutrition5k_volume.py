#!/usr/bin/env python3
"""
Nutrition5kデータセットでの体積予測テスト
- GT深度マップからの体積計算
- Depth Anything V2での予測体積計算
- 両者の比較評価
"""

import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# プロジェクトのパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'nutrition5k'))

from plane_fit import estimate_plane_from_depth
from volume_estimator import height_map_from_plane, pixel_area_map, integrate_volume
from preprocess_nutrition5k import depth_raw_to_meters, get_K_matrix, process_dish


class Nutrition5kVolumeTest:
    """Nutrition5kデータセットでの体積予測テスト"""
    
    def __init__(self, root_dir="nutrition5k/nutrition5k_dataset"):
        self.root_dir = root_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Depth Anything V2モデルをロード
        print("Depth Anything V2モデルをロード中...")
        self.processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
        ).to(self.device).eval()
        print(f"モデルロード完了 (device: {self.device})")
    
    def predict_depth_anything(self, rgb_image):
        """Depth Anything V2で深度予測"""
        # PIL Image形式に変換
        if isinstance(rgb_image, np.ndarray):
            img_pil = Image.fromarray(rgb_image)
        else:
            img_pil = rgb_image
        
        # 元のサイズを記録
        orig_w, orig_h = img_pil.size
        
        # モデル入力を準備
        inputs = self.processor(images=img_pil, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推論実行
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 深度マップを取得
        if hasattr(outputs, 'predicted_depth'):
            pred_depth = outputs.predicted_depth
        else:
            pred_depth = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # 適切な形状に変換
        if len(pred_depth.shape) == 3:
            pred_depth = pred_depth.unsqueeze(1)
        elif len(pred_depth.shape) == 2:
            pred_depth = pred_depth.unsqueeze(0).unsqueeze(0)
        
        # 元のサイズにリサイズ
        if pred_depth.shape[-2:] != (orig_h, orig_w):
            pred_depth = torch.nn.functional.interpolate(
                pred_depth, size=(orig_h, orig_w),
                mode="bicubic", align_corners=False
            )
        
        # numpy配列に変換
        depth_map = pred_depth[0, 0].cpu().numpy()
        
        return depth_map
    
    def generate_simple_mask(self, depth_map, method="center_region"):
        """簡易的なマスク生成（テスト用）"""
        H, W = depth_map.shape
        
        if method == "center_region":
            # 中央領域を食品と仮定
            mask = np.zeros((H, W), dtype=bool)
            center_y, center_x = H // 2, W // 2
            radius = min(H, W) // 3
            
            yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
            distance = np.sqrt((yy - center_y)**2 + (xx - center_x)**2)
            mask = distance < radius
            
        elif method == "depth_threshold":
            # 深度の閾値でマスク生成
            valid_depths = depth_map[depth_map > 0]
            if len(valid_depths) > 0:
                threshold = np.percentile(valid_depths, 30)  # 下位30%を食品と仮定
                mask = (depth_map > 0) & (depth_map < threshold)
            else:
                mask = np.zeros((H, W), dtype=bool)
        
        return mask
    
    def calculate_volume(self, depth_map, K, mask):
        """深度マップから体積を計算"""
        
        # マスクがない場合は簡易生成
        if mask is None:
            mask = self.generate_simple_mask(depth_map, method="depth_threshold")
        
        # マスクをリスト形式に変換（estimate_plane_from_depthの要求形式）
        masks = [mask]
        
        try:
            # テーブル平面を推定
            plane_normal, plane_distance, points_xyz = estimate_plane_from_depth(
                depth_map, K, masks,
                margin_px=40,
                dist_th=0.006,
                max_iters=2000
            )
            
            # 高さマップを計算
            height_map = height_map_from_plane(
                points_xyz, plane_normal, plane_distance, 
                clip_negative=True
            )
            
            # ピクセル面積マップを計算
            area_map = pixel_area_map(depth_map, K)
            
            # 体積を積分
            vol_result = integrate_volume(
                height_map, area_map, mask,
                conf=None, use_conf_weight=False
            )
            
            return vol_result
            
        except Exception as e:
            print(f"  体積計算エラー: {e}")
            return None
    
    def test_single_dish(self, dish_id):
        """単一のdishで体積予測をテスト"""
        
        print(f"\n{'='*70}")
        print(f"Dish: {dish_id}")
        print(f"{'='*70}")
        
        # データ読み込み
        data = process_dish(dish_id, self.root_dir, verbose=False)
        if data is None:
            print(f"エラー: {dish_id}のデータ読み込み失敗")
            return None
        
        rgb = data['rgb']
        gt_depth = data['depth_m']
        K = data['K']
        mask = data['mask']
        
        print(f"\n1. データ情報:")
        print(f"  画像サイズ: {rgb.shape[1]}x{rgb.shape[0]}")
        print(f"  K行列: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
        
        # GT深度の統計
        valid_gt = gt_depth[gt_depth > 0]
        if len(valid_gt) > 0:
            print(f"  GT深度: {valid_gt.min():.3f} - {valid_gt.max():.3f} m")
            print(f"  GT深度中央値: {np.median(valid_gt):.3f} m")
        
        # マスクが無い場合は簡易生成
        if mask is None:
            print("  マスク: 自動生成（深度閾値法）")
            mask = self.generate_simple_mask(gt_depth, method="depth_threshold")
        else:
            print("  マスク: 読み込み済み")
        
        mask_pixels = np.sum(mask)
        print(f"  マスクピクセル数: {mask_pixels} ({mask_pixels/(mask.shape[0]*mask.shape[1])*100:.1f}%)")
        
        # 1. GT深度からの体積計算
        print(f"\n2. GT深度からの体積計算:")
        gt_volume_result = self.calculate_volume(gt_depth, K, mask)
        
        if gt_volume_result:
            gt_volume_ml = gt_volume_result['volume_mL']
            gt_height_mean = gt_volume_result['height_mean_mm']
            gt_height_max = gt_volume_result['height_max_mm']
            
            print(f"  体積: {gt_volume_ml:.1f} mL")
            print(f"  平均高さ: {gt_height_mean:.1f} mm")
            print(f"  最大高さ: {gt_height_max:.1f} mm")
        else:
            gt_volume_ml = None
            print("  計算失敗")
        
        # 2. Depth Anything V2での予測
        print(f"\n3. Depth Anything V2での深度予測:")
        pred_depth = self.predict_depth_anything(rgb)
        
        # 予測深度の統計
        valid_pred = pred_depth[pred_depth > 0]
        if len(valid_pred) > 0:
            print(f"  予測深度: {valid_pred.min():.3f} - {valid_pred.max():.3f} m")
            print(f"  予測深度中央値: {np.median(valid_pred):.3f} m")
        
        # スケール調整（Depth Anything V2は相対深度の可能性）
        # GT深度の中央値に合わせてスケーリング
        if len(valid_gt) > 0 and len(valid_pred) > 0:
            scale_factor = np.median(valid_gt) / np.median(valid_pred)
            print(f"  スケール調整: ×{scale_factor:.2f}")
            pred_depth_scaled = pred_depth * scale_factor
        else:
            pred_depth_scaled = pred_depth
            scale_factor = 1.0
        
        # 3. 予測深度からの体積計算
        print(f"\n4. 予測深度からの体積計算:")
        
        # 内部パラメータの調整（必要に応じて）
        K_pred = K.copy()
        if scale_factor > 10:  # 極端なスケール差がある場合
            K_scale = 10.5  # UniDepth v2の経験値
            K_pred[0,0] *= K_scale
            K_pred[1,1] *= K_scale
            print(f"  K_scale調整: ×{K_scale}")
        
        pred_volume_result = self.calculate_volume(pred_depth_scaled, K_pred, mask)
        
        if pred_volume_result:
            pred_volume_ml = pred_volume_result['volume_mL']
            pred_height_mean = pred_volume_result['height_mean_mm']
            pred_height_max = pred_volume_result['height_max_mm']
            
            print(f"  体積: {pred_volume_ml:.1f} mL")
            print(f"  平均高さ: {pred_height_mean:.1f} mm")
            print(f"  最大高さ: {pred_height_max:.1f} mm")
        else:
            pred_volume_ml = None
            print("  計算失敗")
        
        # 4. 比較評価
        print(f"\n5. 比較評価:")
        if gt_volume_ml and pred_volume_ml:
            error = abs(gt_volume_ml - pred_volume_ml)
            error_pct = (error / gt_volume_ml) * 100
            
            print(f"  GT体積: {gt_volume_ml:.1f} mL")
            print(f"  予測体積: {pred_volume_ml:.1f} mL")
            print(f"  絶対誤差: {error:.1f} mL")
            print(f"  相対誤差: {error_pct:.1f}%")
            
            if error_pct < 20:
                print("  → ✓ 良好（誤差20%以内）")
            elif error_pct < 50:
                print("  → △ 許容範囲（誤差50%以内）")
            else:
                print("  → ✗ 要改善（誤差50%超）")
        else:
            print("  比較不可")
        
        return {
            'dish_id': dish_id,
            'gt_volume_ml': gt_volume_ml,
            'pred_volume_ml': pred_volume_ml,
            'scale_factor': scale_factor
        }
    
    def test_multiple_dishes(self, dish_ids=None):
        """複数のdishでテスト"""
        
        if dish_ids is None:
            # デフォルトのテストセット
            dish_ids = [
                "dish_1556572657",  # mm単位
                "dish_1556573514",  # 0.1mm単位
                "dish_1556575014",
            ]
        
        results = []
        
        print("\n" + "="*70)
        print("Nutrition5k 体積予測テスト")
        print("="*70)
        
        for dish_id in dish_ids:
            result = self.test_single_dish(dish_id)
            if result:
                results.append(result)
        
        # 統計サマリー
        if results:
            print("\n" + "="*70)
            print("統計サマリー")
            print("="*70)
            
            gt_volumes = [r['gt_volume_ml'] for r in results if r['gt_volume_ml']]
            pred_volumes = [r['pred_volume_ml'] for r in results if r['pred_volume_ml']]
            
            if gt_volumes and pred_volumes:
                print(f"\nGT体積:")
                print(f"  平均: {np.mean(gt_volumes):.1f} mL")
                print(f"  中央値: {np.median(gt_volumes):.1f} mL")
                print(f"  範囲: {min(gt_volumes):.1f} - {max(gt_volumes):.1f} mL")
                
                print(f"\n予測体積:")
                print(f"  平均: {np.mean(pred_volumes):.1f} mL")
                print(f"  中央値: {np.median(pred_volumes):.1f} mL")
                print(f"  範囲: {min(pred_volumes):.1f} - {max(pred_volumes):.1f} mL")
                
                # ペアごとの誤差
                errors = []
                for r in results:
                    if r['gt_volume_ml'] and r['pred_volume_ml']:
                        error_pct = abs(r['gt_volume_ml'] - r['pred_volume_ml']) / r['gt_volume_ml'] * 100
                        errors.append(error_pct)
                
                if errors:
                    print(f"\n相対誤差:")
                    print(f"  平均: {np.mean(errors):.1f}%")
                    print(f"  中央値: {np.median(errors):.1f}%")
        
        return results


def main():
    """メイン処理"""
    
    # テストクラスのインスタンス化
    tester = Nutrition5kVolumeTest()
    
    # 複数のdishでテスト実行
    results = tester.test_multiple_dishes()
    
    print("\n" + "="*70)
    print("テスト完了")
    print("="*70)
    print("\n重要な知見:")
    print("1. GT深度マップが利用可能なため、正確な体積GT計算が可能")
    print("2. Depth Anything V2の予測にはスケール調整が必要")
    print("3. セグメンテーションマスクの品質が体積精度に大きく影響")
    print("4. Fine-tuningによりスケール問題と精度改善が期待できる")


if __name__ == "__main__":
    main()