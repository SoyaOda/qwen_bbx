"""
Depth Pro推論エンジン（シミュレーション版）
実際のDepth Proがインストールされるまでの暫定実装
UniDepth v2を内部で使用し、Depth Pro相当の出力を生成
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from PIL import Image
import os
import sys

# UniDepth v2を使用（Depth Proの代替として）
from unidepth_runner_final import UniDepthEngineFinal


class DepthProEngineSimulated:
    """
    Depth Proモデルをシミュレートする推論エンジン
    実際のDepth Proが利用可能になるまでの暫定実装
    """
    
    def __init__(self, device: str = "cuda", checkpoint_path: Optional[str] = None):
        """
        Depth Proエンジンの初期化（シミュレーション版）
        
        Args:
            device: 実行デバイス ("cuda" or "cpu")
            checkpoint_path: モデルチェックポイントのパス（未使用）
        """
        print("=" * 70)
        print("Depth Pro シミュレーションモード")
        print("実際のDepth Proの代わりにUniDepth v2を使用します")
        print("=" * 70)
        
        # UniDepth v2エンジンを内部で使用
        self.unidepth_engine = UniDepthEngineFinal(
            model_repo="lpiccinelli/unidepth-v2-vitl14",
            device=device
        )
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"デバイス: {self.device}")
    
    def infer_image(self, image_path: str) -> Dict[str, Any]:
        """
        画像から深度推定を実行（Depth Proをシミュレート）
        
        Args:
            image_path: 入力画像のパス
            
        Returns:
            Depth Pro互換の出力辞書
        """
        print(f"\n画像を処理中: {image_path}")
        
        # UniDepth v2で推論（適応的K_scaleを使用）
        unidepth_result = self.unidepth_engine.infer_image(
            image_path, 
            K_mode="adaptive"
        )
        
        depth = unidepth_result["depth"]
        K = unidepth_result["intrinsics"]
        xyz = unidepth_result["points"]
        K_scale = unidepth_result["K_scale_factor"]
        
        # Depth Proをシミュレート：絶対スケールに調整
        # K_scaleが適用されているので、理論的には正しいスケール
        print(f"K_scale適用済み: {K_scale:.2f}")
        
        # 深度統計
        H, W = depth.shape
        depth_median = np.median(depth)
        depth_mean = np.mean(depth)
        
        print(f"深度マップサイズ: {depth.shape}")
        print(f"深度範囲: {depth.min():.3f}m - {depth.max():.3f}m")
        print(f"深度中央値: {depth_median:.3f}m")
        
        # 焦点距離の取得
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        print(f"カメラ内部パラメータ: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        
        # サニティチェック
        self._sanity_check(depth, K, xyz)
        
        # Depth Pro互換の結果を返す
        return {
            "depth": depth,
            "intrinsics": K,
            "intrinsics_raw": K.copy(),  # シミュレーションでは同じ
            "points": xyz,
            "confidence": unidepth_result.get("confidence")  # UniDepthの信頼度を流用
        }
    
    def _sanity_check(self, depth: np.ndarray, K: np.ndarray, xyz: np.ndarray):
        """
        推定結果のサニティチェック
        """
        depth_median = np.median(depth)
        fx, fy = K[0, 0], K[1, 1]
        
        # 典型的な画素面積の計算
        a_pix_typical = (depth_median ** 2) / (fx * fy)
        
        print("\n=== サニティチェック ===")
        print(f"典型的な画素面積 (深度中央値): {a_pix_typical:.9f} m^2")
        print(f"  = {a_pix_typical * 1e6:.3f} mm^2")
        
        if a_pix_typical > 1e-5:
            print("注意: 画素面積が大きめです")
        elif a_pix_typical < 1e-9:
            print("注意: 画素面積が非常に小さいです")
        else:
            print("画素面積は妥当な範囲内です")
        
        # 点群の範囲確認
        xyz_min = xyz.min(axis=(0, 1))
        xyz_max = xyz.max(axis=(0, 1))
        xyz_range = xyz_max - xyz_min
        print(f"\n3D点群の範囲:")
        print(f"  X: {xyz_min[0]:.3f}m ~ {xyz_max[0]:.3f}m (幅: {xyz_range[0]:.3f}m)")
        print(f"  Y: {xyz_min[1]:.3f}m ~ {xyz_max[1]:.3f}m (高さ: {xyz_range[1]:.3f}m)")
        print(f"  Z: {xyz_min[2]:.3f}m ~ {xyz_max[2]:.3f}m (奥行き: {xyz_range[2]:.3f}m)")
        print("========================\n")


# 実際のDepth Proが利用不可の場合、シミュレーション版を使用
try:
    import depth_pro
    from depthpro_runner import DepthProEngine
except ImportError:
    print("Depth Proが利用できないため、シミュレーション版を使用します")
    DepthProEngine = DepthProEngineSimulated