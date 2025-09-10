# -*- coding: utf-8 -*-
"""
UniDepth v2 推論エンジン（最終版）
食事撮影に適したK推定を含む実用的な実装
"""
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, Optional

class UniDepthEngineFinal:
    """UniDepth v2 モデルの推論エンジン（最終版）"""
    
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

    def estimate_K_scale_for_food(self, depth: np.ndarray) -> float:
        """
        食事撮影に適したKスケールファクターを推定
        
        Args:
            depth: 深度マップ
            
        Returns:
            推奨スケールファクター
        """
        median_depth = np.median(depth)
        
        # 深度に基づく適応的スケーリング
        if median_depth < 0.8:  # 近接撮影（<80cm）
            scale = 12.0
        elif median_depth < 1.2:  # 標準的な食事撮影（80-120cm）
            scale = 10.5
        elif median_depth < 1.8:  # やや遠い（120-180cm）
            scale = 8.0
        else:  # 遠距離（>180cm）
            scale = 6.0
        
        print(f"深度中央値: {median_depth:.2f}m → K_scale_factor: {scale}")
        return scale
    
    def infer_image(self, 
                   image_path: str, 
                   K_mode: str = "adaptive",
                   fixed_K_scale: float = 10.5) -> Dict[str, Any]:
        """
        RGB画像から深度、内部パラメータ、3D点群、信頼度を推定
        
        Args:
            image_path: 入力画像のパス
            K_mode: "adaptive"（深度に基づく自動調整）, "fixed"（固定スケール）, "raw"（調整なし）
            fixed_K_scale: K_mode="fixed"時のスケールファクター
            
        Returns:
            推論結果の辞書
        """
        # 画像読み込み（正規化はモデル内部で実施）
        rgb_pil = Image.open(image_path).convert("RGB")
        rgb_np = np.array(rgb_pil)
        H, W = rgb_np.shape[:2]
        
        # RGBテンソルに変換（uint8のまま）
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)  # (C,H,W)
        
        # バッチ次元は付けない（README準拠）
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
        K_raw = K_t.detach().cpu().numpy()
        
        # Kスケーリング
        if K_mode == "adaptive":
            # 深度に基づく自動調整
            K_scale = self.estimate_K_scale_for_food(depth)
        elif K_mode == "fixed":
            # 固定スケール
            K_scale = fixed_K_scale
        else:  # "raw"
            K_scale = 1.0
        
        # Kを調整
        K = K_raw.copy()
        K[0, 0] *= K_scale  # fx
        K[1, 1] *= K_scale  # fy
        
        print(f"K調整: fx={K_raw[0,0]:.1f} → {K[0,0]:.1f} (×{K_scale})")
        
        # 3D点群を計算
        xyz = self._unproject_depth_to_xyz(depth, K)
        
        # Sanity check
        self._sanity_check(depth, K, xyz)
        
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
            "intrinsics_raw": K_raw,
            "K_scale_factor": K_scale,
            "points": xyz,
            "confidence": conf
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
    
    def _sanity_check(self, depth: np.ndarray, K: np.ndarray, xyz: np.ndarray):
        """簡易的な妥当性チェック"""
        # Z一致確認
        z_error = np.abs(xyz[:, :, 2] - depth).mean()
        
        # a_pixのオーダー確認
        fx, fy = K[0, 0], K[1, 1]
        z_median = np.median(depth)
        a_pix_typical = (z_median ** 2) / (fx * fy)
        
        print(f"Sanity Check:")
        print(f"  Z誤差: {z_error:.2e}m")
        print(f"  典型a_pix: {a_pix_typical:.2e} m²")
        
        # 警告
        if a_pix_typical > 1e-5:
            print("  ⚠ a_pixが大きすぎます（体積が過大になる可能性）")
        elif a_pix_typical < 1e-8:
            print("  ⚠ a_pixが小さすぎます（体積が過小になる可能性）")