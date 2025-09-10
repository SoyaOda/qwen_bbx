# -*- coding: utf-8 -*-
"""
UniDepth v2 推論エンジン
深度推定、カメラ内部パラメータ、信頼度マップを取得
"""
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, Optional

class UniDepthEngine:
    """UniDepth v2 モデルの推論エンジン"""
    
    def __init__(self, model_repo: str, device: str = "cuda"):
        """
        Args:
            model_repo: Hugging Faceのモデルリポジトリ (例: "lpiccinelli/unidepth-v2-vitl14")
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
                "以下のコマンドでインストールしてください:\n"
                "git clone https://github.com/lpiccinelli-eth/UniDepth.git\n"
                "cd UniDepth\n"
                "pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118"
            ) from e

    def infer_image(self, image_path: str, use_estimated_K: bool = True, K_scale_factor: float = 6.0, use_normalization: bool = False) -> Dict[str, Any]:
        """
        RGB画像から深度、内部パラメータ、3D点群、信頼度を推定
        
        注意: ImageNet正規化の有無をテスト可能。
        正規化なしでも体積のオーダーはほぼ同じ（0.8倍程度）。
        
        Args:
            image_path: 入力画像のパス
            use_estimated_K: Trueの場合、モデルが推定するKを使用。
            K_scale_factor: カメラパラメータのスケーリング係数（デフォルト6.0）
                          体積が大きすぎる場合は増やし、小さすぎる場合は減らす
            use_normalization: ImageNet正規化を使用するか（デフォルトFalse）
            
        Returns:
            dict: {
                "depth": np.ndarray (H,W) - 深度マップ[m]
                "intrinsics": np.ndarray (3,3) - カメラ内部パラメータ（スケール調整済み）
                "intrinsics_original": np.ndarray (3,3) - 元のカメラ内部パラメータ
                "points": np.ndarray (H,W,3) - 3D点群（カメラ座標系）
                "confidence": np.ndarray (H,W) - 信頼度マップ（V2のみ）
            }
        """
        # 画像読み込み
        rgb_pil = Image.open(image_path).convert("RGB")
        rgb_np = np.array(rgb_pil)
        
        if use_normalization:
            # ImageNet正規化あり（元の実装）
            from torchvision import transforms
            
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
            
            # RGB tensorを作成 (H, W, C) -> (C, H, W)
            rgb_tensor = torch.from_numpy(rgb_np).float() / 255.0
            rgb_tensor = rgb_tensor.permute(2, 0, 1)
            rgb_tensor = normalize(rgb_tensor)
        else:
            # 正規化なし（uint8のまま渡す）
            rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1)
        
        # バッチ次元を追加
        if rgb_tensor.ndim == 3:
            rgb_tensor = rgb_tensor.unsqueeze(0)  # (1,C,H,W)
        
        rgb_tensor = rgb_tensor.to(self.device)
        
        # 推論実行
        with torch.inference_mode():
            if use_estimated_K:
                # Kを指定せず、モデルに推定させる
                pred = self.model.infer(rgb_tensor)
            else:
                # TODO: 手動でKを指定する場合の実装
                raise NotImplementedError("手動K指定はまだ実装されていません")
        
        # 深度マップ (メートル単位)
        depth_t = pred.get("depth")  # (B,H,W) or (H,W)
        if depth_t.ndim == 4:
            depth_t = depth_t[0, 0]  # (B,C,H,W) -> (H,W)
        elif depth_t.ndim == 3:
            depth_t = depth_t[0]  # (B,H,W) -> (H,W)
        depth = depth_t.detach().cpu().numpy()
        
        # カメラ内部パラメータ（モデルが推定）
        K_t = pred.get("intrinsics")  # (B,3,3) or (3,3)
        if K_t.ndim == 4:
            K_t = K_t[0, 0]  # (B,C,3,3) -> (3,3)
        elif K_t.ndim == 3 and K_t.shape[0] != 3:
            K_t = K_t[0]  # (B,3,3) -> (3,3)
        K_original = K_t.detach().cpu().numpy()
        
        # カメラパラメータのスケーリング
        # UniDepthの推定値は小さすぎることがあるため調整
        K = K_original.copy()
        K[0, 0] *= K_scale_factor  # fx
        K[1, 1] *= K_scale_factor  # fy
        # cx, cyはそのまま
        
        # 3D点群を正しい逆投影で計算（調整済みKを使用）
        H, W = depth.shape
        points = self._unproject_depth_to_xyz(depth, K)
        
        # 信頼度マップ（V2で追加）
        conf_t = pred.get("confidence", None)
        if conf_t is not None:
            if conf_t.ndim == 4:
                conf_t = conf_t[0, 0]  # (B,C,H,W) -> (H,W)
            elif conf_t.ndim == 3:
                conf_t = conf_t[0]  # (B,H,W) -> (H,W)
            conf = conf_t.detach().cpu().numpy()
        else:
            conf = None
        
        return {
            "depth": depth,
            "intrinsics": K,  # スケール調整済み
            "intrinsics_original": K_original,  # 元の値（参考用）
            "points": points,
            "confidence": conf
        }
    
    def _unproject_depth_to_xyz(self, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        深度マップから3D点群を生成（正しい逆投影式を使用）
        
        Args:
            depth: (H,W) 深度マップ[m]
            K: (3,3) カメラ内部パラメータ
            
        Returns:
            (H,W,3) 3D点群（カメラ座標系）
        """
        H, W = depth.shape
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)  # 画素座標
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # 正しい逆投影式（K^{-1}を使用）
        Z = depth
        X = (uu - cx) * Z / fx
        Y = (vv - cy) * Z / fy
        
        # (H,W,3)形状で返す
        xyz = np.stack([X, Y, Z], axis=-1)
        return xyz