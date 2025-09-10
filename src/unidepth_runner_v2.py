# -*- coding: utf-8 -*-
"""
UniDepth v2 推論エンジン（改良版）
K_scale_factorを撤廃し、推定Kをそのまま使用
EXIFからのK計算と妥当性チェック機能付き
"""
import numpy as np
import torch
from PIL import Image
from PIL.ExifTags import TAGS
from typing import Dict, Any, Optional, Tuple
import warnings

class UniDepthEngineV2:
    """UniDepth v2 モデルの推論エンジン（改良版）"""
    
    def __init__(self, model_repo: str, device: str = "cuda"):
        """
        Args:
            model_repo: Hugging Faceのモデルリポジトリ
            device: 実行デバイス ("cuda" or "cpu")
        """
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        print(f"UniDepth v2 デバイス: {self.device}")
        
        # UniDepthモデルのロード
        try:
            from unidepth.models import UniDepthV2
            self.model = UniDepthV2.from_pretrained(model_repo)
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"UniDepth v2 モデルロード完了: {model_repo}")
        except ImportError as e:
            raise RuntimeError(
                "UniDepthがインストールされていません。\n"
                "git clone https://github.com/lpiccinelli-eth/UniDepth.git\n"
                "cd UniDepth\n"
                "pip install -e ."
            ) from e

    def compute_K_from_exif(self, image_path: str, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        EXIFデータからカメラ内部パラメータを計算
        
        Args:
            image_path: 画像パス
            image_shape: (H, W)
            
        Returns:
            K行列またはNone
        """
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()
            
            if exif_data is None:
                return None
                
            # EXIFタグを解析
            exif = {}
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                exif[tag] = value
            
            H, W = image_shape
            
            # 35mm換算焦点距離を取得
            focal_35mm = exif.get('FocalLengthIn35mmFilm')
            focal_length = exif.get('FocalLength')
            
            if focal_35mm:
                # 35mm換算から計算: f_pixels = W * f_35 / 36
                fx = W * float(focal_35mm) / 36.0
                fy = fx  # アスペクト比1:1と仮定
                
            elif focal_length:
                # 実焦点距離から計算（センサーサイズが必要）
                # 一般的なスマートフォンのセンサー幅を仮定（約7mm）
                sensor_width_mm = 7.0  # デフォルト値
                
                # メーカー/モデルから推定（簡易版）
                make = exif.get('Make', '')
                model = exif.get('Model', '')
                
                if 'iPhone' in str(make) or 'iPhone' in str(model):
                    sensor_width_mm = 4.8  # iPhone標準
                elif 'Samsung' in str(make):
                    sensor_width_mm = 6.4  # Samsung標準
                    
                fx = W * float(focal_length) / sensor_width_mm
                fy = fx
            else:
                return None
                
            cx = W / 2.0
            cy = H / 2.0
            
            K = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
            
            print(f"EXIF由来K: fx={fx:.1f}, fy={fy:.1f}")
            return K
            
        except Exception as e:
            print(f"EXIF読み取りエラー: {e}")
            return None
    
    def sanity_check_K(self, K_pred: np.ndarray, K_exif: Optional[np.ndarray], 
                      depth: np.ndarray, xyz: np.ndarray) -> Dict[str, Any]:
        """
        カメラパラメータと深度の妥当性チェック
        
        Returns:
            チェック結果の辞書
        """
        checks = {}
        
        # 1. Z成分の一致確認
        z_error = np.abs(xyz[:, :, 2] - depth).mean()
        checks['z_consistency'] = {
            'mean_error': float(z_error),
            'passed': z_error < 1e-6
        }
        
        # 2. fx, fyの妥当性（一般的な範囲）
        fx_pred, fy_pred = K_pred[0, 0], K_pred[1, 1]
        H, W = depth.shape
        
        # 画像幅に対する相対的な焦点距離
        fx_relative = fx_pred / W
        typical_range = (0.5, 3.0)  # 典型的な範囲
        
        checks['K_range'] = {
            'fx': float(fx_pred),
            'fy': float(fy_pred),
            'fx_relative': float(fx_relative),
            'in_typical_range': typical_range[0] <= fx_relative <= typical_range[1]
        }
        
        # 3. EXIFとの比較（あれば）
        if K_exif is not None:
            fx_exif = K_exif[0, 0]
            ratio = fx_pred / fx_exif
            checks['exif_comparison'] = {
                'fx_exif': float(fx_exif),
                'fx_pred': float(fx_pred),
                'ratio': float(ratio),
                'within_50_percent': 0.5 <= ratio <= 1.5
            }
        
        # 4. a_pixのオーダー確認
        fx, fy = K_pred[0, 0], K_pred[1, 1]
        z_median = np.median(depth)
        a_pix_typical = (z_median ** 2) / (fx * fy)
        
        checks['pixel_area'] = {
            'z_median': float(z_median),
            'a_pix_typical': float(a_pix_typical),
            'order_of_magnitude': int(np.log10(a_pix_typical))
        }
        
        return checks
    
    def infer_image(self, image_path: str, K_source: str = "predicted") -> Dict[str, Any]:
        """
        RGB画像から深度、内部パラメータ、3D点群、信頼度を推定
        
        Args:
            image_path: 入力画像のパス
            K_source: "predicted"（推定）, "exif"（EXIF優先）, "auto"（自動選択）
            
        Returns:
            推論結果の辞書
        """
        # 画像読み込み（正規化はモデル内部で実施）
        rgb_pil = Image.open(image_path).convert("RGB")
        rgb_np = np.array(rgb_pil)
        H, W = rgb_np.shape[:2]
        
        # RGBテンソルに変換（正規化なし、uint8のまま）
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)  # (C,H,W)
        
        # READMEに従い、バッチ次元は付けない
        if rgb_tensor.ndim == 4:
            rgb_tensor = rgb_tensor.squeeze(0)
            
        rgb_tensor = rgb_tensor.to(self.device)
        
        # 推論実行
        with torch.inference_mode():
            pred = self.model.infer(rgb_tensor)
        
        # 深度マップ取得
        depth_t = pred.get("depth")
        if depth_t.ndim >= 3:
            depth_t = depth_t.squeeze()
        depth = depth_t.detach().cpu().numpy()
        
        # カメラ内部パラメータ取得
        K_t = pred.get("intrinsics")
        if K_t.ndim >= 3:
            K_t = K_t.squeeze()
        K_pred = K_t.detach().cpu().numpy()
        
        # EXIFからKを計算（参考用）
        K_exif = self.compute_K_from_exif(image_path, (H, W))
        
        # K選択
        if K_source == "exif" and K_exif is not None:
            K = K_exif
            print("EXIFのKを使用")
        elif K_source == "auto":
            # 自動選択：EXIFがあり、推定値と大きく異なる場合は警告
            if K_exif is not None:
                ratio = K_pred[0, 0] / K_exif[0, 0]
                if not (0.5 <= ratio <= 1.5):
                    print(f"警告: 推定KとEXIF Kが大きく異なります（比率: {ratio:.2f}）")
            K = K_pred  # デフォルトは推定値
        else:
            K = K_pred
            
        print(f"使用K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
        
        # 3D点群を計算（正しい逆投影）
        xyz = self._unproject_depth_to_xyz(depth, K)
        
        # Sanity check
        checks = self.sanity_check_K(K_pred, K_exif, depth, xyz)
        self._print_sanity_check(checks)
        
        # 信頼度マップ
        conf_t = pred.get("confidence")
        if conf_t is not None:
            if conf_t.ndim >= 3:
                conf_t = conf_t.squeeze()
            conf = conf_t.detach().cpu().numpy()
        else:
            conf = None
        
        return {
            "depth": depth,
            "intrinsics": K,
            "intrinsics_pred": K_pred,
            "intrinsics_exif": K_exif,
            "points": xyz,
            "confidence": conf,
            "sanity_check": checks
        }
    
    def _unproject_depth_to_xyz(self, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        """正しい逆投影（K^{-1}を使用）"""
        H, W = depth.shape
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        Z = depth
        X = (uu - cx) * Z / fx
        Y = (vv - cy) * Z / fy
        
        return np.stack([X, Y, Z], axis=-1)
    
    def _print_sanity_check(self, checks: Dict[str, Any]):
        """Sanity checkの結果を表示"""
        print("\n=== Sanity Check ===")
        
        # Z一致
        z_check = checks['z_consistency']
        status = "✓" if z_check['passed'] else "✗"
        print(f"{status} Z一致: 誤差={z_check['mean_error']:.2e}m")
        
        # K範囲
        k_check = checks['K_range']
        status = "✓" if k_check['in_typical_range'] else "⚠"
        print(f"{status} K範囲: fx={k_check['fx']:.1f} (相対値={k_check['fx_relative']:.2f})")
        
        # EXIF比較
        if 'exif_comparison' in checks:
            e_check = checks['exif_comparison']
            status = "✓" if e_check['within_50_percent'] else "⚠"
            print(f"{status} EXIF比較: 推定/EXIF = {e_check['ratio']:.2f}")
        
        # ピクセル面積
        p_check = checks['pixel_area']
        print(f"   a_pix: {p_check['a_pix_typical']:.2e} m² (10^{p_check['order_of_magnitude']})")
        
        print("==================\n")