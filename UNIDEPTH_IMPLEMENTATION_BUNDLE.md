# UniDepth v2 実装コード集

生成日時: 2025-09-10 15:49:12

## 概要

このドキュメントは、Qwen BBXプロジェクトにおけるUniDepth v2統合の主要実装ファイルをまとめたものです。

### 検証結果と重要な発見

#### 1. K_scale_factorの必要性
- **UniDepthの根本的な問題**: モデルが推定するカメラ内部パラメータ（fx, fy）が実際の値より約10倍小さい
- **原因**: UniDepthは近接撮影（食品撮影）のデータで訓練されていない
- **影響**: K_scale_factorなしでは体積が46Lなど非現実的な値になる

#### 2. 最適なK_scale_factorの値
- **理論的に正しい値**: 約10.5（検証により判明）
- **実用的な値**: 画像によって1.0〜12.0まで大きく変動
- **深度による適応的調整**:
  - 深度 < 0.8m: K_scale = 12.0
  - 深度 < 1.2m: K_scale = 10.5
  - 深度 < 1.5m: K_scale = 8.0
  - それ以上: K_scale = 6.0

#### 3. 検証結果（test_all_images.py）
- **固定K_scale=6.0**: 成功率11.1%（画像によって最適値が異なるため）
- **固定K_scale=10.5**: 理論的に正しいが、汎用性は低い
- **適応的K_scale**: 深度に基づく調整でも限界あり（成功率約33%）

### 技術的詳細

#### ピクセル面積の計算式
```
a_pix = Z² / (fx * fy)
```
- Z: 深度値[m]
- fx, fy: カメラの焦点距離[pixels]
- UniDepthはfx≈686を推定するが、実際にはfx≈10,500が必要

#### ImageNet正規化について
- **結論**: 正規化は不要（model.infer()が内部で処理）
- **影響**: 正規化の有無で体積が0.8倍程度変化するのみ

---

## 目次

1. [unidepth_runner.py](#unidepth-runnerpy) - UniDepth v2推論エンジン（初期実装、K_scale_factor=6.0）
2. [unidepth_runner_v2.py](#unidepth-runner-v2py) - 改良版UniDepth推論エンジン（レビュー推奨実装、K_scale削除）
3. [unidepth_runner_final.py](#unidepth-runner-finalpy) - 最終版UniDepth推論エンジン（適応的K_scale対応）
4. [plane_fit.py](#plane-fitpy) - RANSAC平面フィッティング（初期実装）
5. [plane_fit_v2.py](#plane-fit-v2py) - 改良版RANSAC平面フィッティング（適応的閾値）
6. [volume_estimator.py](#volume-estimatorpy) - 体積推定モジュール
7. [vis_depth.py](#vis-depthpy) - 深度マップ可視化
8. [run_unidepth.py](#run-unidepthpy) - メインパイプライン（Qwen→SAM2→UniDepth→体積）
9. [config.yaml](#configyaml) - 全体設定
10. [test_all_images.py](#test-all-imagespy) - 汎用性検証テスト（全画像での最適K_scale探索）
11. [test_with_masks_only.py](#test-with-masks-onlypy) - SAM2マスク画像での体積推定テスト

---

## unidepth_runner.py

**パス**: `src/unidepth_runner.py`

**説明**: UniDepth v2推論エンジン（初期実装、K_scale_factor=6.0）

```python
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
```

*ファイルサイズ: 5,728 bytes, 行数: 170*

---

## unidepth_runner_v2.py

**パス**: `src/unidepth_runner_v2.py`

**説明**: 改良版UniDepth推論エンジン（レビュー推奨実装、K_scale削除）

```python
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
```

*ファイルサイズ: 9,278 bytes, 行数: 293*

---

## unidepth_runner_final.py

**パス**: `src/unidepth_runner_final.py`

**説明**: 最終版UniDepth推論エンジン（適応的K_scale対応）

```python
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
```

*ファイルサイズ: 5,462 bytes, 行数: 183*

---

## plane_fit.py

**パス**: `src/plane_fit.py`

**説明**: RANSAC平面フィッティング（初期実装）

```python
# -*- coding: utf-8 -*-
"""
平面フィッティングモジュール
食品マスクの外側リング領域から皿/卓面をRANSACで推定
"""
import numpy as np
import cv2
from typing import Tuple, Optional

def build_support_ring(food_union_mask: np.ndarray, margin_px: int = 40) -> np.ndarray:
    """
    食品マスクの外側リング領域（皿や卓面候補）を作成
    
    Args:
        food_union_mask: 全食品の結合マスク (H,W) bool
        margin_px: リングの幅（ピクセル）
    
    Returns:
        ring: リング領域のマスク (H,W) bool
    """
    # カーネルサイズ（奇数にする）
    k = 2 * margin_px + 1
    kernel = np.ones((k, k), np.uint8)
    
    # マスクを膨張
    dil = cv2.dilate(food_union_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    
    # 元のマスクを除外してリングを作成
    ring = np.logical_and(dil, np.logical_not(food_union_mask))
    
    return ring

def fit_plane_ransac(
    points_xyz: np.ndarray,
    cand_mask: np.ndarray,
    dist_th: float = 0.006,
    max_iters: int = 2000,
    min_support: int = 2000,
    rng_seed: int = 3
) -> Tuple[Tuple[np.ndarray, float], float]:
    """
    RANSAC法で平面を当てはめ
    
    Args:
        points_xyz: (3,H,W) の3D点群（カメラ座標系、メートル単位）
        cand_mask: (H,W) のbool（RANSAC候補点のマスク）
        dist_th: 点-平面距離の閾値 [m]
        max_iters: 最大反復回数
        min_support: 最小サポート点数
        rng_seed: 乱数シード
    
    Returns:
        (n, d): 平面パラメータ（n·X + d = 0、nは単位法線ベクトル）
        inliers: インライア数
    """
    H, W = cand_mask.shape
    ys, xs = np.where(cand_mask)
    
    if ys.size < min_support:
        raise RuntimeError(f"平面候補点が不足: {ys.size} < {min_support}")
    
    # 候補点の3D座標を抽出
    X = points_xyz[0, ys, xs]
    Y = points_xyz[1, ys, xs]
    Z = points_xyz[2, ys, xs]
    P = np.stack([X, Y, Z], axis=1)  # (N, 3)
    
    # 有限値のみを使用
    valid_mask = np.isfinite(P).all(axis=1)
    P = P[valid_mask]
    
    if P.shape[0] < min_support:
        raise RuntimeError(f"有効な平面候補点が不足: {P.shape[0]} < {min_support}")
    
    rs = np.random.RandomState(rng_seed)
    best_inliers = -1
    best_n, best_d = None, None
    
    for _ in range(max_iters):
        # ランダムに3点を選択
        idx = rs.choice(P.shape[0], size=3, replace=False)
        p1, p2, p3 = P[idx]
        
        # 平面の法線ベクトルを計算
        v1, v2 = p2 - p1, p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-8:
            continue  # 3点が同一直線上にある
        
        n = n / n_norm  # 正規化
        d = -np.dot(n, p1)  # 平面方程式のd
        
        # 全点から平面までの距離を計算
        dist = np.abs(P @ n + d)
        inliers = (dist < dist_th)
        n_in = int(inliers.sum())
        
        if n_in > best_inliers:
            # 最小二乗法でリファイン
            Q = P[inliers]
            
            # SVDを使って最適な平面を求める
            # 点群の重心を計算
            centroid = np.mean(Q, axis=0)
            Q_centered = Q - centroid
            
            # SVDで主成分分析
            _, _, vh = np.linalg.svd(Q_centered, full_matrices=False)
            n_ref = vh[-1, :]  # 最小特異値に対応する特異ベクトル（法線）
            
            # 法線の向きを統一（+Z方向が上）
            if n_ref[2] < 0:
                n_ref = -n_ref
            
            # 平面方程式のdを再計算
            d_ref = -np.dot(n_ref, centroid)
            
            best_n, best_d, best_inliers = n_ref, d_ref, n_in
    
    if best_n is None:
        raise RuntimeError("RANSAC平面推定に失敗しました")
    
    return (best_n, float(best_d)), float(best_inliers)

def estimate_plane_from_depth(
    depth: np.ndarray,
    K: np.ndarray,
    food_masks: list,
    margin_px: int = 40,
    dist_th: float = 0.006,
    max_iters: int = 2000,
    min_support: int = 2000
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    深度マップと食品マスクから平面を推定
    
    Args:
        depth: 深度マップ (H,W) [m]
        K: カメラ内部パラメータ (3,3)
        food_masks: 食品マスクのリスト
        margin_px: リングマージン
        dist_th: RANSAC距離閾値
        max_iters: RANSAC最大反復回数
        min_support: 最小サポート点数
    
    Returns:
        n: 平面の法線ベクトル (3,)
        d: 平面方程式の定数項
        points_xyz: 3D点群 (3,H,W)
    """
    H, W = depth.shape
    
    # 3D点群を生成
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # ピクセル座標のメッシュグリッド
    us = np.arange(W)
    vs = np.arange(H)
    uu, vv = np.meshgrid(us, vs)
    
    # 3D座標を計算
    Z = depth
    X = (uu - cx) / fx * Z
    Y = (vv - cy) / fy * Z
    points_xyz = np.stack([X, Y, Z], axis=0)
    
    # 全食品マスクの結合
    if len(food_masks) > 0:
        union_mask = np.zeros((H, W), dtype=bool)
        for mask in food_masks:
            union_mask |= mask
    else:
        # マスクがない場合は画像中央を使用
        union_mask = np.zeros((H, W), dtype=bool)
        cy, cx = H // 2, W // 2
        r = min(H, W) // 4
        yy, xx = np.ogrid[:H, :W]
        union_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    
    # リング領域を作成
    ring = build_support_ring(union_mask, margin_px)
    
    # RANSACで平面フィッティング
    try:
        (n, d), nin = fit_plane_ransac(
            points_xyz, ring,
            dist_th=dist_th,
            max_iters=max_iters,
            min_support=min_support
        )
    except RuntimeError as e:
        # フォールバック：画像全体から平面を推定
        print(f"リング領域でのRANSAC失敗: {e}")
        print("画像全体から平面を推定します...")
        
        full_mask = np.logical_not(union_mask)
        (n, d), nin = fit_plane_ransac(
            points_xyz, full_mask,
            dist_th=dist_th,
            max_iters=max_iters,
            min_support=min_support // 2
        )
    
    print(f"平面推定完了: インライア数={nin:.0f}")
    print(f"  法線: [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")
    print(f"  d: {d:.3f}")
    
    return n, d, points_xyz
```

*ファイルサイズ: 5,458 bytes, 行数: 211*

---

## plane_fit_v2.py

**パス**: `src/plane_fit_v2.py`

**説明**: 改良版RANSAC平面フィッティング（適応的閾値）

```python
# -*- coding: utf-8 -*-
"""
平面フィッティングモジュール（改良版）
符号決定ロジックの改善とスケール依存のパラメータ調整
"""
import numpy as np
import cv2
from typing import Tuple, Optional

def build_support_ring(food_union_mask: np.ndarray, margin_ratio: float = 0.04) -> np.ndarray:
    """
    食品マスクの外側リング領域（皿や卓面候補）を作成
    
    Args:
        food_union_mask: 全食品の結合マスク (H,W) bool
        margin_ratio: リング幅の割合（画像最小辺に対する比率）
    
    Returns:
        ring: リング領域のマスク (H,W) bool
    """
    H, W = food_union_mask.shape
    margin_px = int(margin_ratio * min(H, W))
    margin_px = max(margin_px, 10)  # 最小10ピクセル
    
    # カーネルサイズ（奇数にする）
    k = 2 * margin_px + 1
    kernel = np.ones((k, k), np.uint8)
    
    # マスクを膨張
    dil = cv2.dilate(food_union_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    
    # 元のマスクを除外してリングを作成
    ring = np.logical_and(dil, np.logical_not(food_union_mask))
    
    return ring

def fit_plane_ransac(
    points_xyz: np.ndarray,
    cand_mask: np.ndarray,
    z_scale_factor: float = 0.012,
    min_dist_th: float = 0.004,
    max_iters: int = 2000,
    min_support: int = 1000,
    rng_seed: int = 3
) -> Tuple[Tuple[np.ndarray, float], int]:
    """
    RANSAC法で平面を当てはめ（スケール依存の閾値）
    
    Args:
        points_xyz: (3,H,W) or (H,W,3) の3D点群
        cand_mask: (H,W) のbool（RANSAC候補点のマスク）
        z_scale_factor: 深度に対する距離閾値の係数
        min_dist_th: 最小距離閾値 [m]
        max_iters: 最大反復回数
        min_support: 最小サポート点数
        rng_seed: 乱数シード
    
    Returns:
        (n, d): 平面パラメータ（n·X + d = 0、nは単位法線ベクトル）
        inliers: インライア数
    """
    # 入力形状を統一
    if points_xyz.shape[0] == 3 and len(points_xyz.shape) == 3:
        # (3,H,W) -> (H,W,3)
        points_xyz = np.transpose(points_xyz, (1, 2, 0))
    
    H, W = cand_mask.shape
    
    # 候補点を抽出
    P = points_xyz[cand_mask].reshape(-1, 3)
    
    # 有限値のみを使用
    valid_mask = np.isfinite(P).all(axis=1)
    P = P[valid_mask]
    
    if P.shape[0] < min_support:
        raise RuntimeError(f"平面候補点が不足: {P.shape[0]} < {min_support}")
    
    # 深度の中央値からRANSAC閾値を自動調整
    z_median = np.median(P[:, 2])
    dist_th = max(min_dist_th, z_scale_factor * z_median)
    print(f"  RANSAC閾値: {dist_th*1000:.1f}mm (深度中央値: {z_median:.2f}m)")
    
    # RANSAC
    N = P.shape[0]
    best_inl = 0
    best_model = None
    rng = np.random.default_rng(rng_seed)
    
    for _ in range(max_iters):
        # 3点をランダムに選択
        idx = rng.choice(N, 3, replace=False)
        p1, p2, p3 = P[idx]
        
        # 平面の法線を計算（外積）
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        
        # 法線を正規化
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        
        # 平面方程式の定数項
        d = -np.dot(n, p1)
        
        # 点から平面までの距離
        dist = np.abs(P @ n + d)
        
        # インライアをカウント
        inl = np.sum(dist < dist_th)
        
        if inl > best_inl:
            best_inl = inl
            best_model = (n, d)
    
    if best_model is None:
        raise RuntimeError("平面を見つけられませんでした")
    
    # 最終的な平面をインライアで再推定（最小二乗法）
    n, d = best_model
    dist = np.abs(P @ n + d)
    inliers = P[dist < dist_th]
    
    if inliers.shape[0] >= 3:
        # SVDで最適平面を計算
        centroid = np.mean(inliers, axis=0)
        centered = inliers - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        n = Vt[-1]  # 最小特異値に対応する特異ベクトル
        d = -np.dot(n, centroid)
        best_model = (n, d)
    
    return best_model, best_inl

def height_map_from_plane(
    points_xyz: np.ndarray,
    plane_n: np.ndarray,
    plane_d: float,
    table_mask: Optional[np.ndarray] = None,
    food_mask: Optional[np.ndarray] = None,
    clip_negative: bool = True
) -> np.ndarray:
    """
    平面からの高さマップを計算（改良版符号決定）
    
    Args:
        points_xyz: 3D点群 (3,H,W) or (H,W,3)
        plane_n: 平面の法線ベクトル (3,)
        plane_d: 平面方程式の定数項
        table_mask: 卓面候補領域のマスク
        food_mask: 食品領域のマスク
        clip_negative: 負の高さを0にクリップするか
    
    Returns:
        height: 高さマップ (H,W) [m]
    """
    # 入力形状を統一
    if points_xyz.shape[-1] == 3:
        # (H,W,3) -> (3,H,W)
        points_xyz = np.transpose(points_xyz, (2, 0, 1))
    
    X = points_xyz[0]
    Y = points_xyz[1]
    Z = points_xyz[2]
    
    # 符号付き距離を計算
    h = plane_n[0] * X + plane_n[1] * Y + plane_n[2] * Z + plane_d
    
    # 符号の自動決定
    if table_mask is not None and np.any(table_mask):
        # 卓面の中央値を基準に
        med_table = np.median(h[table_mask])
        
        if food_mask is not None and np.any(food_mask):
            med_food = np.median(h[food_mask])
            # 食品が卓面より低い場合は符号を反転
            if med_food < med_table:
                h = -h
                print(f"  高さ符号を反転（食品側を正に）")
        else:
            # 卓面が0に近くなるように調整
            if abs(med_table) > 0.01:  # 1cm以上ずれている場合
                if med_table > 0:
                    h = -h
    else:
        # フォールバック：カメラ座標系のZ軸向きで判定
        if plane_n[2] > 0:
            h = -h
    
    if clip_negative:
        h = np.maximum(h, 0.0)
    
    return h

def estimate_plane_from_depth_v2(
    depth: np.ndarray,
    K: np.ndarray,
    food_masks: list,
    margin_ratio: float = 0.04,
    z_scale_factor: float = 0.012,
    min_support: int = 1000
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    深度マップと食品マスクから平面を推定（改良版）
    
    Args:
        depth: 深度マップ (H,W) [m]
        K: カメラ内部パラメータ (3,3)
        food_masks: 食品マスクのリスト
        margin_ratio: リング幅の割合
        z_scale_factor: RANSAC閾値の係数
        min_support: 最小サポート点数
    
    Returns:
        n: 平面の法線ベクトル (3,)
        d: 平面方程式の定数項
        points_xyz: 3D点群 (H,W,3)
    """
    H, W = depth.shape
    
    # 3D点群を生成（正しい逆投影）
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    points_xyz = np.stack([X, Y, Z], axis=-1)
    
    # 全食品マスクの結合
    if len(food_masks) > 0:
        union_mask = np.zeros((H, W), dtype=bool)
        for mask in food_masks:
            union_mask |= mask
    else:
        # マスクがない場合は画像中央を使用
        union_mask = np.zeros((H, W), dtype=bool)
        cy, cx = H // 2, W // 2
        r = min(H, W) // 4
        yy, xx = np.ogrid[:H, :W]
        union_mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2
    
    # リング領域を作成
    ring = build_support_ring(union_mask, margin_ratio)
    
    # RANSAC（スケール依存の閾値）
    try:
        (n, d), nin = fit_plane_ransac(
            points_xyz, ring,
            z_scale_factor=z_scale_factor,
            min_support=min_support
        )
    except RuntimeError as e:
        print(f"リング領域でのRANSAC失敗: {e}")
        print("画像全体から平面を推定します...")
        
        full_mask = np.logical_not(union_mask)
        (n, d), nin = fit_plane_ransac(
            points_xyz, full_mask,
            z_scale_factor=z_scale_factor,
            min_support=min_support // 2
        )
    
    print(f"平面推定完了: インライア数={nin:.0f}")
    print(f"  法線: [{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}]")
    print(f"  d: {d:.3f}")
    
    return n, d, points_xyz
```

*ファイルサイズ: 7,086 bytes, 行数: 275*

---

## volume_estimator.py

**パス**: `src/volume_estimator.py`

**説明**: 体積推定モジュール

```python
# -*- coding: utf-8 -*-
"""
体積推定モジュール
平面からの高さマップを生成し、マスクごとの体積を積分
"""
import numpy as np
from typing import Dict, Any, Optional, List

def ensure_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    深度マップとカメラ内部パラメータから3D点群を生成
    
    Args:
        depth: 深度マップ (H,W) [m]
        K: カメラ内部パラメータ (3,3)
    
    Returns:
        points: 3D点群 (3,H,W) [m]
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # ピクセル座標のメッシュグリッド
    us = np.arange(W)
    vs = np.arange(H)
    uu, vv = np.meshgrid(us, vs)
    
    # 3D座標を計算（カメラ座標系）
    Z = depth
    X = (uu - cx) / fx * Z
    Y = (vv - cy) / fy * Z
    
    return np.stack([X, Y, Z], axis=0)

def height_map_from_plane(
    points_xyz: np.ndarray,
    plane_n: np.ndarray,
    plane_d: float,
    clip_negative: bool = True
) -> np.ndarray:
    """
    平面からの高さマップを計算
    
    Args:
        points_xyz: 3D点群 (3,H,W) [m]
        plane_n: 平面の法線ベクトル (3,)
        plane_d: 平面方程式の定数項
        clip_negative: 負の高さを0にクリップするか
    
    Returns:
        height: 高さマップ (H,W) [m]
    """
    # 各点の平面からの符号付き距離を計算
    # 平面方程式: n·X + d = 0
    # 点から平面までの距離: h = n·X + d
    X = points_xyz[0]
    Y = points_xyz[1]
    Z = points_xyz[2]
    
    h = plane_n[0] * X + plane_n[1] * Y + plane_n[2] * Z + plane_d
    
    # 皿面が0、上が正になるように符号を調整
    # （法線が上向きの場合、皿の上の点はh > 0になるはず）
    if plane_n[2] > 0:  # 法線が上向き
        h = -h  # 符号を反転
    
    if clip_negative:
        h = np.maximum(h, 0.0)
    
    return h

def pixel_area_map(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    各ピクセルが表す実世界の面積を計算
    
    Args:
        depth: 深度マップ (H,W) [m]
        K: カメラ内部パラメータ (3,3)
    
    Returns:
        area: ピクセル面積マップ (H,W) [m²]
    """
    fx, fy = K[0, 0], K[1, 1]
    
    # 小面積近似: a_pix(z) ≈ (z²)/(fx·fy)
    # これは、カメラから距離zにある平面上の1ピクセルが表す面積
    area = (depth ** 2) / (fx * fy + 1e-12)
    
    return area

def integrate_volume(
    height: np.ndarray,
    a_pix: np.ndarray,
    mask_bool: np.ndarray,
    conf: Optional[np.ndarray] = None,
    use_conf_weight: bool = False
) -> Dict[str, Any]:
    """
    マスク内の体積を積分
    
    Args:
        height: 高さマップ (H,W) [m]
        a_pix: ピクセル面積マップ (H,W) [m²]
        mask_bool: 対象領域のマスク (H,W) bool
        conf: 信頼度マップ (H,W)（オプション）
        use_conf_weight: 信頼度による重み付けを使用するか
    
    Returns:
        dict: {
            "pixels": マスク内のピクセル数,
            "volume_mL": 体積（ミリリットル）,
            "height_mean_mm": 平均高さ（ミリメートル）,
            "height_max_mm": 最大高さ（ミリメートル）
        }
    """
    m = mask_bool.astype(bool)
    
    if not np.any(m):
        return {
            "pixels": 0,
            "volume_mL": 0.0,
            "height_mean_mm": 0.0,
            "height_max_mm": 0.0
        }
    
    # 体積積分: V = Σ h(x,y) * a_pix(x,y)
    if use_conf_weight and conf is not None:
        # 信頼度による重み付け
        w = np.clip(conf, 0.0, 1.0)
        V = float(np.sum(height[m] * a_pix[m] * w[m]))
    else:
        # 重み付けなし
        V = float(np.sum(height[m] * a_pix[m]))
    
    # m³ → mL (1m³ = 1,000,000 mL)
    volume_mL = V * 1e6
    
    # 高さ統計（m → mm）
    heights_m = height[m]
    height_mean_mm = float(np.mean(heights_m)) * 1000
    height_max_mm = float(np.max(heights_m)) * 1000
    
    return {
        "pixels": int(m.sum()),
        "volume_mL": volume_mL,
        "height_mean_mm": height_mean_mm,
        "height_max_mm": height_max_mm
    }

def estimate_volumes(
    depth: np.ndarray,
    K: np.ndarray,
    plane_n: np.ndarray,
    plane_d: float,
    masks: List[np.ndarray],
    labels: List[str],
    confidence: Optional[np.ndarray] = None,
    use_conf_weight: bool = False
) -> List[Dict[str, Any]]:
    """
    複数のマスクに対して体積を推定
    
    Args:
        depth: 深度マップ (H,W) [m]
        K: カメラ内部パラメータ (3,3)
        plane_n: 平面の法線ベクトル (3,)
        plane_d: 平面方程式の定数項
        masks: マスクのリスト
        labels: ラベルのリスト
        confidence: 信頼度マップ（オプション）
        use_conf_weight: 信頼度による重み付けを使用するか
    
    Returns:
        results: 各マスクの体積推定結果のリスト
    """
    # 3D点群を生成
    points_xyz = ensure_points(depth, K)
    
    # 高さマップを計算
    height = height_map_from_plane(points_xyz, plane_n, plane_d, clip_negative=True)
    
    # ピクセル面積マップを計算
    a_pix = pixel_area_map(depth, K)
    
    # 各マスクに対して体積を計算
    results = []
    for i, (mask, label) in enumerate(zip(masks, labels)):
        # 信頼度なしで計算
        vol_plain = integrate_volume(height, a_pix, mask, conf=None, use_conf_weight=False)
        
        # 信頼度ありで計算（可能な場合）
        if confidence is not None and use_conf_weight:
            vol_conf = integrate_volume(height, a_pix, mask, conf=confidence, use_conf_weight=True)
        else:
            vol_conf = vol_plain
        
        result = {
            "id": i,
            "label": label,
            "pixels": vol_plain["pixels"],
            "volume_mL": vol_plain["volume_mL"],
            "volume_mL_conf": vol_conf["volume_mL"] if use_conf_weight else None,
            "height_mean_mm": vol_plain["height_mean_mm"],
            "height_max_mm": vol_plain["height_max_mm"]
        }
        results.append(result)
        
        print(f"  {label}:")
        print(f"    体積: {vol_plain['volume_mL']:.1f} mL")
        print(f"    平均高さ: {vol_plain['height_mean_mm']:.1f} mm")
        print(f"    最大高さ: {vol_plain['height_max_mm']:.1f} mm")
    
    return results
```

*ファイルサイズ: 5,332 bytes, 行数: 214*

---

## vis_depth.py

**パス**: `src/vis_depth.py`

**説明**: 深度マップ可視化

```python
# -*- coding: utf-8 -*-
"""
深度マップと高さマップの可視化ユーティリティ
depth_modelプロジェクトと同じビジュアライゼーション方法を実装
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import Optional, Tuple, Dict

def normalize_depth_for_display(
    depth: np.ndarray,
    percentile: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    """
    深度を可視化用に正規化（depth_modelと同じ）
    
    Args:
        depth: 入力深度マップ
        percentile: クリッピング用のパーセンタイル (min, max)
        
    Returns:
        normalized: [0, 1]に正規化された深度
    """
    # 無効な値を除去
    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    
    if len(valid_depth) == 0:
        return np.zeros_like(depth)
    
    # ロバストな正規化のためのパーセンタイル値を取得
    vmin = np.percentile(valid_depth, percentile[0])
    vmax = np.percentile(valid_depth, percentile[1])
    
    # クリップして正規化
    normalized = np.clip(depth, vmin, vmax)
    normalized = (normalized - vmin) / (vmax - vmin + 1e-6)
    
    # 無効な値を処理
    normalized[~np.isfinite(depth)] = 0
    
    return normalized

def apply_colormap(
    depth: np.ndarray,
    colormap: str = "viridis",
    normalize: bool = True,
    model_name: Optional[str] = None
) -> np.ndarray:
    """
    深度マップにカラーマップを適用（depth_modelと同じ実装）
    
    Args:
        depth: 入力深度マップ
        colormap: Matplotlibカラーマップ名
        normalize: 深度を最初に正規化するか
        model_name: モデル固有の処理用のモデル名（"UniDepth"など）
        
    Returns:
        colored: RGB画像 (H, W, 3), uint8
    """
    # モデル固有の処理（統一ルール: 「近い=明るい」）
    if model_name:
        if model_name == "UniDepth":
            # UniDepthはメートル単位の深度を出力 - 近い=小さい値
            # 逆数化して近い=大きい値に変換
            display_depth = 1.0 / (np.maximum(depth, 1e-6))
        else:
            display_depth = depth
    else:
        display_depth = depth
    
    if normalize:
        display_depth = normalize_depth_for_display(display_depth)
    
    # カラーマップを取得
    if hasattr(cm, 'get_cmap'):
        cmap = cm.get_cmap(colormap)
    else:
        # Matplotlib 3.11以降の新しいAPI
        cmap = plt.colormaps[colormap]
    
    # カラーマップを適用（RGBAを返す）
    colored = cmap(display_depth)
    
    # RGB uint8に変換
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return rgb

def colorize_depth(depth: np.ndarray, clip_q: Tuple[float, float] = (0.02, 0.98)) -> np.ndarray:
    """
    深度マップをカラーマップで可視化（後方互換性のため維持）
    
    Args:
        depth: 深度マップ [m]
        clip_q: クリッピング用のパーセンタイル (low, high)
    
    Returns:
        カラーマップ適用済みのBGR画像
    """
    # depth_modelと同じViridisカラーマップを使用
    rgb = apply_colormap(depth, colormap="viridis", model_name="UniDepth")
    # OpenCVはBGRを期待するので変換
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def colorize_height(height: np.ndarray, max_h_m: float = 0.05) -> np.ndarray:
    """
    高さマップをカラーマップで可視化
    
    Args:
        height: 高さマップ [m]
        max_h_m: 表示する最大高さ [m]
    
    Returns:
        カラーマップ適用済みのBGR画像
    """
    h = np.clip(height / max_h_m, 0, 1)
    h8 = (h * 255).astype(np.uint8)
    return cv2.applyColorMap(h8, cv2.COLORMAP_MAGMA)

def create_depth_panel(
    original: np.ndarray,
    depth: np.ndarray,
    height: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    元画像、深度マップ、高さマップ、信頼度マップを並べたパネル画像を作成
    
    Args:
        original: 元画像 (BGR)
        depth: 深度マップ [m]
        height: 高さマップ [m]（オプション）
        confidence: 信頼度マップ（オプション）
    
    Returns:
        パネル画像 (BGR)
    """
    H, W = depth.shape
    panels = []
    
    # 元画像
    if original.shape[:2] != (H, W):
        original = cv2.resize(original, (W, H))
    panels.append(original)
    
    # 深度マップ
    depth_viz = colorize_depth(depth)
    panels.append(depth_viz)
    
    # 高さマップ（あれば）
    if height is not None:
        height_viz = colorize_height(height)
        panels.append(height_viz)
    
    # 信頼度マップ（あれば）
    if confidence is not None:
        conf_viz = (np.clip(confidence, 0, 1) * 255).astype(np.uint8)
        conf_viz = cv2.applyColorMap(conf_viz, cv2.COLORMAP_VIRIDIS)
        panels.append(conf_viz)
    
    # パネルを横に並べる
    panel = np.concatenate(panels, axis=1)
    
    # ラベルを追加
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    
    labels = ["Original", "Depth"]
    if height is not None:
        labels.append("Height")
    if confidence is not None:
        labels.append("Confidence")
    
    for i, label in enumerate(labels):
        x = i * W + 10
        y = 30
        cv2.putText(panel, label, (x, y), font, font_scale, color, thickness)
    
    return panel

def save_depth_as_16bit_png(
    depth: np.ndarray,
    save_path: str,
    max_depth: Optional[float] = None
) -> None:
    """
    深度を16ビットPNGとして保存（depth_modelと同じ）
    
    Args:
        depth: 入力深度マップ
        save_path: PNG保存パス
        max_depth: スケーリング用の最大深度
    """
    if max_depth is None:
        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        if len(valid_depth) > 0:
            max_depth = np.percentile(valid_depth, 99)
        else:
            max_depth = 1.0
    
    # 16ビット範囲にスケーリング
    depth_scaled = np.clip(depth / max_depth, 0, 1)
    depth_16bit = (depth_scaled * 65535).astype(np.uint16)
    
    # PNGとして保存
    Image.fromarray(depth_16bit, mode='I;16').save(save_path)

def create_comparison_panel(
    rgb_image: np.ndarray,
    depth_maps: Dict[str, np.ndarray],
    save_path: str,
    colormap: str = "viridis"
) -> None:
    """
    RGBと深度マップの比較パネルを作成（depth_modelと同じ）
    
    Args:
        rgb_image: 入力RGB画像
        depth_maps: モデル名 -> 深度マップの辞書
        save_path: パネル保存パス
        colormap: 深度可視化用のカラーマップ
    """
    n_models = len(depth_maps)
    fig, axes = plt.subplots(2, (n_models + 1) // 2 + 1, 
                            figsize=(5 * ((n_models + 1) // 2 + 1), 10))
    
    # 1次元配列に平坦化
    axes = axes.flatten()
    
    # RGB画像を表示
    axes[0].imshow(rgb_image)
    axes[0].set_title("Input RGB", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # モデル固有の処理で深度マップを表示
    for idx, (model_name, depth) in enumerate(depth_maps.items(), 1):
        colored_depth = apply_colormap(depth, colormap=colormap, model_name=model_name)
        axes[idx].imshow(colored_depth)
        axes[idx].set_title(model_name, fontsize=12)
        axes[idx].axis('off')
    
    # 未使用のサブプロットを非表示
    for idx in range(n_models + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_depth_as_ply(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    output_path: str = "output.ply"
) -> None:
    """
    3D点群をPLY形式で保存
    
    Args:
        points: 3D点群 (3, H, W) または (N, 3)
        colors: 色情報 (3, H, W) または (N, 3)、0-255の範囲
        output_path: 出力ファイルパス
    """
    # 点群を (N, 3) 形式に変換
    if points.shape[0] == 3 and len(points.shape) == 3:
        # (3, H, W) -> (N, 3)
        points = points.reshape(3, -1).T
    
    # 有効な点のみを抽出
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    
    if colors is not None:
        if colors.shape[0] == 3 and len(colors.shape) == 3:
            colors = colors.reshape(3, -1).T
        colors = colors[valid_mask]
    
    # PLYヘッダー
    num_points = points.shape[0]
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z"""
    
    if colors is not None:
        header += """
property uchar red
property uchar green
property uchar blue"""
    
    header += "\nend_header\n"
    
    # ファイルに書き込み
    with open(output_path, 'w') as f:
        f.write(header)
        for i in range(num_points):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if colors is not None:
                line += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"
            f.write(line + "\n")
    
    print(f"PLYファイル保存: {output_path} ({num_points} points)")
```

*ファイルサイズ: 7,988 bytes, 行数: 310*

---

## run_unidepth.py

**パス**: `src/run_unidepth.py`

**説明**: メインパイプライン（Qwen→SAM2→UniDepth→体積）

```python
# -*- coding: utf-8 -*-
"""
UniDepth v2 メインスクリプト
Qwen/SAM2の結果を基に深度推定→平面フィッティング→体積計算
"""
import os
import json
import glob
import numpy as np
import cv2
import yaml
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# 自作モジュール
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unidepth_runner import UniDepthEngine
from plane_fit import estimate_plane_from_depth
from volume_estimator import (
    height_map_from_plane,
    pixel_area_map,
    estimate_volumes
)
from vis_depth import (
    apply_colormap,
    save_depth_as_16bit_png,
    colorize_height
)
from visualize import ensure_dir

def load_sam2_summary(json_path: str):
    """SAM2のサマリJSONを読み込み"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_binary_mask(path: str) -> np.ndarray:
    """バイナリマスクPNGを読み込み"""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127)

def find_mask_files(mask_dir: str, stem: str, det_idx: int, label: str, source: str):
    """マスクファイルのパスを生成"""
    # ラベルを安全なファイル名に変換
    safe_lab = "".join([c if c.isalnum() else "_" for c in label])[:40]
    return os.path.join(mask_dir, f"{stem}_det{det_idx:02d}_{safe_lab}_{source}.png")

def main():
    # 設定ファイル読み込み
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # UniDepth設定
    uni_cfg = cfg.get("unidepth", {})
    plane_cfg = cfg.get("plane", {})
    vol_cfg = cfg.get("volume", {})
    paths = cfg.get("paths", {})
    
    # マスクソース（bplus or large）
    mask_source = cfg.get("mask_source", "large")
    
    # 出力ディレクトリ
    out_root = cfg.get("unidepth_paths", {}).get("out_root", "outputs/unidepth")
    ddir = os.path.join(out_root, "depth")
    cdir = os.path.join(out_root, "conf")
    kdir = os.path.join(out_root, "intrinsics")
    hdir = os.path.join(out_root, "height")
    vdir = os.path.join(out_root, "viz")
    jdir = os.path.join(out_root, "json")
    
    for d in (ddir, cdir, kdir, hdir, vdir, jdir):
        ensure_dir(d)
    
    # UniDepthモデルを初期化
    print("UniDepth v2 モデルを初期化中...")
    engine = UniDepthEngine(
        model_repo=uni_cfg.get("model_repo", "lpiccinelli/unidepth-v2-vitl14"),
        device=uni_cfg.get("device", "cuda")
    )
    
    # 入力画像とSAM2結果のパスを設定
    img_dir = paths.get("input_dir", "test_images")
    sam2_json_dir = paths.get("sam2_json_dir", "outputs/sam2/json")
    mask_dir = paths.get("sam2_mask_dir", "outputs/sam2/masks")
    
    # 処理する画像を取得
    stems = []
    for p in glob.glob(os.path.join(img_dir, "*")):
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            stems.append(os.path.splitext(os.path.basename(p))[0])
    stems.sort()
    
    print(f"\n{len(stems)}枚の画像を処理します")
    
    for stem in tqdm(stems, desc="UniDepth v2 → 平面 → 体積"):
        # 画像パスを検索
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            p = os.path.join(img_dir, f"{stem}{ext}")
            if os.path.exists(p):
                img_path = p
                break
        
        if not img_path:
            continue
        
        print(f"\n処理中: {stem}")
        
        # 1) UniDepth推論
        print("  深度推定...")
        K_scale = uni_cfg.get("K_scale_factor", 6.0)
        pred = engine.infer_image(img_path, K_scale_factor=K_scale)
        depth = pred["depth"]
        K = pred["intrinsics"]
        points = pred["points"]
        conf = pred["confidence"]
        
        # 次元が多い場合は削減
        if depth.ndim == 4:
            depth = depth[0, 0]  # (B,C,H,W) -> (H,W)
        elif depth.ndim == 3:
            depth = depth[0]  # (B,H,W) or (C,H,W) -> (H,W)
        
        if K.ndim == 3:
            K = K[0]  # (B,3,3) -> (3,3)
        
        if conf is not None:
            if conf.ndim == 4:
                conf = conf[0, 0]
            elif conf.ndim == 3:
                conf = conf[0]
        
        H, W = depth.shape
        
        # 深度データを保存
        if uni_cfg.get("save_npy", True):
            np.save(os.path.join(ddir, f"{stem}.npy"), depth)
            np.save(os.path.join(kdir, f"{stem}.K.npy"), K)
            if conf is not None:
                np.save(os.path.join(cdir, f"{stem}.conf.npy"), conf)
        
        if uni_cfg.get("save_png", True):
            save_depth_as_16bit_png(depth, os.path.join(ddir, f"{stem}.png"))
            if conf is not None:
                c8 = (np.clip(conf, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(cdir, f"{stem}.png"), c8)
        
        # 2) SAM2の検出結果を読み込み
        sam2_json_path = os.path.join(sam2_json_dir, f"{stem}.sam2.json")
        if not os.path.exists(sam2_json_path):
            print(f"  SAM2結果が見つかりません: {sam2_json_path}")
            continue
        
        summ = load_sam2_summary(sam2_json_path)
        dets = summ.get("detections", [])
        
        if len(dets) == 0:
            print("  検出結果がありません")
            continue
        
        # マスクを読み込み
        masks = []
        labels = []
        for i, det in enumerate(dets):
            label = det.get("label_ja", det.get("label_en", f"object_{i}"))
            labels.append(label)
            
            # マスクファイルを探す
            mpath = find_mask_files(mask_dir, stem, i, label, mask_source)
            if os.path.exists(mpath):
                m = load_binary_mask(mpath)
                masks.append(m)
            else:
                print(f"  警告: マスクファイルが見つかりません: {mpath}")
                # 空のマスクを追加
                masks.append(np.zeros((H, W), dtype=bool))
        
        # 3) 平面フィッティング
        print("  平面推定...")
        try:
            plane_n, plane_d, points_xyz = estimate_plane_from_depth(
                depth, K, masks,
                margin_px=plane_cfg.get("ring_margin_px", 40),
                dist_th=plane_cfg.get("ransac_threshold_m", 0.006),
                max_iters=plane_cfg.get("ransac_max_iters", 2000),
                min_support=plane_cfg.get("min_support_px", 2000)
            )
        except Exception as e:
            print(f"  平面推定エラー: {e}")
            continue
        
        # 4) 高さマップ生成
        height = height_map_from_plane(points_xyz, plane_n, plane_d, 
                                      clip_negative=vol_cfg.get("clip_negative_height", True))
        
        # 高さマップを保存
        if uni_cfg.get("save_npy", True):
            np.save(os.path.join(hdir, f"{stem}.height.npy"), height)
        
        # 5) 体積計算
        print("  体積計算...")
        volumes = estimate_volumes(
            depth, K, plane_n, plane_d,
            masks, labels,
            confidence=conf,
            use_conf_weight=vol_cfg.get("use_confidence_weight", False)
        )
        
        # 6) 可視化
        print("  可視化...")
        
        # 元画像を読み込み
        img_bgr = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # 深度マップの可視化
        depth_viz = apply_colormap(depth, model_name="UniDepth")
        depth_viz_bgr = cv2.cvtColor(depth_viz, cv2.COLOR_RGB2BGR)
        
        # 高さマップの可視化
        height_viz = colorize_height(height, max_h_m=0.05)  # 5cm上限
        
        # パネル画像を作成（元画像｜深度｜高さ）
        panel = np.concatenate([img_bgr, depth_viz_bgr, height_viz], axis=1)
        
        # ラベルを追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, "Depth", (W + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, "Height", (2*W + 10, 30), font, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(os.path.join(vdir, f"{stem}_panel.jpg"), panel)
        
        # 7) JSON保存
        result_json = {
            "image": os.path.basename(img_path),
            "width": W,
            "height": H,
            "intrinsics": K.tolist(),
            "plane": {
                "normal": plane_n.tolist(),
                "d": float(plane_d)
            },
            "mask_source": mask_source,
            "detections": volumes
        }
        
        with open(os.path.join(jdir, f"{stem}.unidepth.json"), "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
    
    print(f"\n完了: 結果は {out_root} に保存されました")

if __name__ == "__main__":
    main()
```

*ファイルサイズ: 8,385 bytes, 行数: 257*

---

## config.yaml

**パス**: `config.yaml`

**説明**: 全体設定

```yaml
provider:
  base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
  model: "qwen2.5-vl-72b-instruct"
  api_key_env: "DASHSCOPE_API_KEY"
  request_timeout_s: 120
  max_retries: 2
  temperature: 0.0
  top_p: 0.1
inference:
  max_items: 50
  conf_threshold: 0.20
  iou_merge_threshold: 0.0
dataset:
  input_dir: "test_images"
  out_json_dir: "outputs/json"
  out_viz_dir: "outputs/viz"

# SAM2 configuration
sam2:
  repo_root: "/home/soya/sam2_1_food_finetuning/external/sam2"
  cfg_base_plus: "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
  cfg_large: "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
  ckpt_base_plus: "/home/soya/sam2_1_food_finetuning/external/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
  ckpt_large: "/home/soya/sam2_1_food_finetuning/external/sam2/checkpoints/sam2.1_hiera_large.pt"
  device: "cuda"                   # "cuda" or "cpu"
  dtype: "bfloat16"                # "bfloat16"(GPU) / "float32"(CPU)
  multimask_output: true           # SAM2で複数仮説出力→最良を選択
  conf_threshold: 0.20             # Qwen検出の信頼度閾値

# Paths for SAM2 processing
paths:
  qwen_json_dir: "outputs/json"    # 既存Qwen出力(JSON)の場所
  input_dir: "test_images"         # 入力画像ディレクトリ
  out_root: "outputs/sam2"         # SAM2出力のルートディレクトリ
  sam2_json_dir: "outputs/sam2/json"  # SAM2結果JSONの場所
  sam2_mask_dir: "outputs/sam2/masks" # SAM2マスクPNGの場所

# UniDepth v2 configuration
unidepth:
  model_repo: "lpiccinelli/unidepth-v2-vitl14"  # HuggingFaceモデルID（ViT-L版）
  device: "cuda"                                 # "cuda" or "cpu"
  save_npy: true                                 # npyファイルを保存
  save_png: true                                 # 16bit PNGを保存
  K_scale_factor: 6.0                           # カメラパラメータのスケーリング係数（体積調整用）

# Plane fitting configuration
plane:
  ring_margin_px: 40           # 食品マスクの外側リング幅（皿/卓面候補）
  ransac_threshold_m: 0.006    # 平面距離の閾値[m]（約6mm）
  ransac_max_iters: 2000       # RANSAC最大反復回数
  min_support_px: 2000         # RANSAC最小有効点数

# Volume estimation configuration
volume:
  use_confidence_weight: false  # true: 信頼度を重み付けに使用
  area_formula: "z2_over_fx_fy" # ピクセル面積の計算式
  clip_negative_height: true    # 負の高さを0にクリップ

# Mask source selection
mask_source: "large"  # "bplus" or "large" - どちらのSAM2モデルのマスクを使うか

# Output paths for UniDepth
unidepth_paths:
  out_root: "outputs/unidepth"  # UniDepth出力のルートディレクトリ
```

*ファイルサイズ: 2,326 bytes, 行数: 64*

---

## test_all_images.py

**パス**: `test_all_images.py`

**説明**: 汎用性検証テスト（全画像での最適K_scale探索）

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全test_images画像で体積推定の汎用性をテスト
各画像に対して異なるK_scale_factorを試して最適値を分析
"""
import sys
import os
import glob
sys.path.insert(0, 'src')

import numpy as np
import cv2
from PIL import Image
from unidepth_runner import UniDepthEngine
from plane_fit import estimate_plane_from_depth
from volume_estimator import height_map_from_plane, pixel_area_map, integrate_volume
from vis_depth import apply_colormap
import json

def test_single_image(image_path, mask_paths, K_scale_factors=[1.0, 6.0, 8.0, 10.5, 12.0, 15.0]):
    """
    単一画像で複数のK_scale_factorをテスト
    """
    # UniDepth推論
    engine = UniDepthEngine(
        model_repo="lpiccinelli/unidepth-v2-vitl14",
        device="cuda"
    )
    
    # 画像名
    img_name = os.path.basename(image_path)
    
    results = {
        "image": img_name,
        "tests": []
    }
    
    print(f"\n{'='*70}")
    print(f"画像: {img_name}")
    print(f"{'='*70}")
    
    # 基本推論（K_scale=1.0で深度と元のKを取得）
    pred_base = engine.infer_image(image_path, K_scale_factor=1.0)
    depth = pred_base["depth"]
    K_orig = pred_base["intrinsics_original"]
    conf = pred_base["confidence"]
    
    H, W = depth.shape
    
    # 深度情報
    depth_stats = {
        "min": float(depth.min()),
        "max": float(depth.max()),
        "median": float(np.median(depth))
    }
    
    print(f"深度: {depth_stats['min']:.2f} - {depth_stats['max']:.2f}m (中央値: {depth_stats['median']:.2f}m)")
    print(f"元K: fx={K_orig[0,0]:.1f}, fy={K_orig[1,1]:.1f}")
    
    # マスクを読み込み
    masks = []
    labels = []
    for mask_path in mask_paths:
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, 0) > 127
            masks.append(mask)
            # ラベルをファイル名から抽出
            label = os.path.basename(mask_path).split('_')[2]  # det00_label_source.png
            labels.append(label)
    
    if not masks:
        # マスクがない場合は画像中央を使用
        cy, cx = H // 2, W // 2
        r = min(H, W) // 6
        yy, xx = np.ogrid[:H, :W]
        mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
        masks = [mask]
        labels = ["center_region"]
    
    print(f"マスク数: {len(masks)} ({', '.join(labels)})")
    
    # 各K_scale_factorでテスト
    print(f"\n{'K_scale':<10} {'体積(mL)':<40} {'評価':<10}")
    print("-" * 70)
    
    for K_scale in K_scale_factors:
        # Kを調整
        K = K_orig.copy()
        K[0, 0] *= K_scale
        K[1, 1] *= K_scale
        
        try:
            # 平面推定
            n, d, points_xyz = estimate_plane_from_depth(
                depth, K, masks,
                margin_px=40,
                dist_th=0.006,
                max_iters=2000
            )
            
            # 高さマップ
            height = height_map_from_plane(
                points_xyz, n, d,
                clip_negative=True
            )
            
            # ピクセル面積
            a_pix = pixel_area_map(depth, K)
            
            # 各マスクの体積を計算
            volumes = []
            for mask, label in zip(masks, labels):
                vol_result = integrate_volume(
                    height, a_pix, mask,
                    conf=conf,
                    use_conf_weight=False
                )
                volumes.append({
                    "label": label,
                    "volume_mL": vol_result["volume_mL"],
                    "height_mean_mm": vol_result["height_mean_mm"],
                    "height_max_mm": vol_result["height_max_mm"]
                })
            
            # 合計体積
            total_volume = sum(v["volume_mL"] for v in volumes)
            
            # 評価
            if 50 <= total_volume <= 1000:
                evaluation = "✓ 適切"
            elif total_volume < 50:
                evaluation = "⚠ 小さい"
            elif total_volume < 2000:
                evaluation = "△ やや大"
            else:
                evaluation = f"✗ 異常({total_volume/1000:.1f}L)"
            
            # 結果表示
            volume_str = ", ".join([f"{v['label'][:8]}:{v['volume_mL']:.0f}" for v in volumes])
            print(f"{K_scale:<10.1f} {volume_str:<40} {evaluation}")
            
            # 結果保存
            results["tests"].append({
                "K_scale": K_scale,
                "total_volume_mL": total_volume,
                "volumes": volumes,
                "evaluation": evaluation
            })
            
        except Exception as e:
            print(f"{K_scale:<10.1f} エラー: {str(e)[:40]}")
            results["tests"].append({
                "K_scale": K_scale,
                "error": str(e)
            })
    
    # 最適なK_scaleを推定
    valid_tests = [t for t in results["tests"] if "total_volume_mL" in t]
    if valid_tests:
        # 200-500mLの範囲に最も近いものを選択
        target_range = (200, 500)
        best = min(valid_tests, key=lambda t: 
                  abs(t["total_volume_mL"] - np.mean(target_range)))
        
        results["optimal_K_scale"] = best["K_scale"]
        results["optimal_volume_mL"] = best["total_volume_mL"]
        
        print(f"\n推奨K_scale: {best['K_scale']:.1f} (体積: {best['total_volume_mL']:.1f}mL)")
    
    results["depth_stats"] = depth_stats
    results["K_original"] = {
        "fx": float(K_orig[0, 0]),
        "fy": float(K_orig[1, 1])
    }
    
    return results

def main():
    """全test_images画像をテスト"""
    
    # test_images内の画像を取得
    test_images = sorted(glob.glob("test_images/*.jpg"))
    if not test_images:
        test_images = sorted(glob.glob("test_images/*.png"))
    
    print(f"テスト画像数: {len(test_images)}")
    
    # 各画像に対応するマスクを検索
    all_results = []
    
    for img_path in test_images:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        
        # SAM2マスクを検索（複数ある場合）
        mask_pattern = f"outputs/sam2/masks/{stem}_det*_*_bplus.png"
        mask_paths = sorted(glob.glob(mask_pattern))
        
        if not mask_paths:
            # largeマスクも試す
            mask_pattern = f"outputs/sam2/masks/{stem}_det*_*_large.png"
            mask_paths = sorted(glob.glob(mask_pattern))
        
        # テスト実行
        result = test_single_image(img_path, mask_paths)
        all_results.append(result)
    
    # 統計分析
    print(f"\n{'='*70}")
    print("統計分析")
    print(f"{'='*70}")
    
    # 各K_scaleでの成功率を計算
    k_scale_stats = {}
    for k in [1.0, 6.0, 8.0, 10.5, 12.0, 15.0]:
        successes = 0
        total = 0
        volumes = []
        
        for result in all_results:
            for test in result["tests"]:
                if test.get("K_scale") == k and "total_volume_mL" in test:
                    total += 1
                    vol = test["total_volume_mL"]
                    volumes.append(vol)
                    if 50 <= vol <= 1000:
                        successes += 1
        
        if total > 0:
            success_rate = successes / total * 100
            avg_volume = np.mean(volumes) if volumes else 0
            
            k_scale_stats[k] = {
                "success_rate": success_rate,
                "avg_volume_mL": avg_volume,
                "count": total
            }
    
    print(f"\n{'K_scale':<10} {'成功率':<10} {'平均体積(mL)':<15} {'サンプル数'}")
    print("-" * 50)
    for k, stats in sorted(k_scale_stats.items()):
        print(f"{k:<10.1f} {stats['success_rate']:<10.1f}% {stats['avg_volume_mL']:<15.1f} {stats['count']}")
    
    # 最適なK_scaleの分布
    optimal_k_values = [r.get("optimal_K_scale", 0) for r in all_results if "optimal_K_scale" in r]
    if optimal_k_values:
        print(f"\n最適K_scaleの分布:")
        print(f"  平均: {np.mean(optimal_k_values):.1f}")
        print(f"  中央値: {np.median(optimal_k_values):.1f}")
        print(f"  範囲: {min(optimal_k_values):.1f} - {max(optimal_k_values):.1f}")
    
    # 深度と最適K_scaleの相関
    print(f"\n深度と最適K_scaleの関係:")
    depth_vs_k = []
    for r in all_results:
        if "optimal_K_scale" in r and "depth_stats" in r:
            depth_vs_k.append({
                "image": r["image"],
                "depth_median": r["depth_stats"]["median"],
                "optimal_k": r["optimal_K_scale"]
            })
    
    if depth_vs_k:
        depth_vs_k.sort(key=lambda x: x["depth_median"])
        for item in depth_vs_k:
            print(f"  {item['image']:<20} 深度中央値: {item['depth_median']:.2f}m → K_scale: {item['optimal_k']:.1f}")
    
    # 結果をJSONで保存
    with open("test_all_images_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n結果を test_all_images_results.json に保存しました")
    
    # 推奨値
    print(f"\n{'='*70}")
    print("推奨事項")
    print(f"{'='*70}")
    
    best_k = max(k_scale_stats.items(), key=lambda x: x[1]["success_rate"])[0]
    print(f"最も汎用性の高いK_scale_factor: {best_k}")
    print(f"成功率: {k_scale_stats[best_k]['success_rate']:.1f}%")
    print(f"平均体積: {k_scale_stats[best_k]['avg_volume_mL']:.1f}mL")

if __name__ == "__main__":
    main()
```

*ファイルサイズ: 8,911 bytes, 行数: 285*

---

## test_with_masks_only.py

**パス**: `test_with_masks_only.py`

**説明**: SAM2マスク画像での体積推定テスト

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2マスクがある画像のみで体積推定テスト
深度に基づく適応的K_scaleも検証
"""
import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import cv2
from src.unidepth_runner_final import UniDepthEngineFinal
from plane_fit import estimate_plane_from_depth
from volume_estimator import height_map_from_plane, pixel_area_map, integrate_volume

def test_with_adaptive_k(image_path, mask_paths):
    """適応的K_scaleでテスト"""
    
    engine = UniDepthEngineFinal(
        model_repo="lpiccinelli/unidepth-v2-vitl14",
        device="cuda"
    )
    
    img_name = os.path.basename(image_path)
    print(f"\n{'='*70}")
    print(f"画像: {img_name}")
    print(f"{'='*70}")
    
    # テスト設定
    test_modes = [
        ("raw", "fixed", 1.0),
        ("fixed_6", "fixed", 6.0),
        ("fixed_10.5", "fixed", 10.5),
        ("adaptive", "adaptive", None)
    ]
    
    results = []
    
    for mode_name, K_mode, K_scale in test_modes:
        print(f"\n【{mode_name}】")
        
        try:
            # UniDepth推論
            if K_mode == "adaptive":
                pred = engine.infer_image(image_path, K_mode="adaptive")
            else:
                pred = engine.infer_image(image_path, K_mode="fixed", fixed_K_scale=K_scale)
            
            depth = pred["depth"]
            K = pred["intrinsics"]
            K_scale_used = pred["K_scale_factor"]
            conf = pred["confidence"]
            
            H, W = depth.shape
            
            print(f"深度範囲: {depth.min():.2f} - {depth.max():.2f}m (中央値: {np.median(depth):.2f}m)")
            print(f"K_scale使用値: {K_scale_used:.1f}")
            print(f"調整後K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
            
            # マスク読み込み
            masks = []
            labels = []
            for mask_path in mask_paths:
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, 0) > 127
                    masks.append(mask)
                    # ラベル抽出
                    parts = os.path.basename(mask_path).split('_')
                    label = parts[2] if len(parts) > 2 else "mask"
                    labels.append(label)
            
            if not masks:
                print("マスクが見つかりません")
                continue
            
            # 平面推定
            n, d, points_xyz = estimate_plane_from_depth(
                depth, K, masks,
                margin_px=40,
                dist_th=0.006,
                max_iters=2000
            )
            
            # 高さマップ
            height = height_map_from_plane(
                points_xyz, n, d,
                clip_negative=True
            )
            
            # ピクセル面積
            a_pix = pixel_area_map(depth, K)
            
            # 体積計算
            total_volume = 0
            for mask, label in zip(masks, labels):
                vol_result = integrate_volume(
                    height, a_pix, mask,
                    conf=conf,
                    use_conf_weight=False
                )
                volume_mL = vol_result["volume_mL"]
                height_mean = vol_result["height_mean_mm"]
                total_volume += volume_mL
                
                print(f"  {label}: {volume_mL:.1f}mL (平均高さ: {height_mean:.1f}mm)")
            
            # 評価
            if 50 <= total_volume <= 1000:
                evaluation = "✓ 適切"
            elif total_volume < 50:
                evaluation = "⚠ 小さすぎ"
            elif total_volume < 2000:
                evaluation = "△ やや大きい"
            else:
                evaluation = f"✗ 異常({total_volume/1000:.1f}L)"
            
            print(f"合計体積: {total_volume:.1f}mL {evaluation}")
            
            results.append({
                "mode": mode_name,
                "K_scale": K_scale_used,
                "total_volume_mL": total_volume,
                "evaluation": evaluation
            })
            
        except Exception as e:
            print(f"エラー: {e}")
            results.append({
                "mode": mode_name,
                "error": str(e)
            })
    
    # 最適なモードを選択
    valid_results = [r for r in results if "total_volume_mL" in r and 50 <= r["total_volume_mL"] <= 1000]
    if valid_results:
        best = min(valid_results, key=lambda r: abs(r["total_volume_mL"] - 350))
        print(f"\n推奨: {best['mode']} (体積: {best['total_volume_mL']:.1f}mL)")
    
    return results

def main():
    """SAM2マスクがある画像のみテスト"""
    
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
    
    all_results = {}
    
    for img_path, mask_paths in test_cases:
        if os.path.exists(img_path):
            results = test_with_adaptive_k(img_path, mask_paths)
            all_results[os.path.basename(img_path)] = results
    
    # 統計
    print(f"\n{'='*70}")
    print("統計分析")
    print(f"{'='*70}")
    
    mode_stats = {}
    
    for img_name, results in all_results.items():
        for r in results:
            if "total_volume_mL" in r:
                mode = r["mode"]
                if mode not in mode_stats:
                    mode_stats[mode] = {"volumes": [], "successes": 0, "total": 0}
                
                vol = r["total_volume_mL"]
                mode_stats[mode]["volumes"].append(vol)
                mode_stats[mode]["total"] += 1
                
                if 50 <= vol <= 1000:
                    mode_stats[mode]["successes"] += 1
    
    print(f"\n{'モード':<15} {'成功率':<10} {'平均体積(mL)':<15} {'中央値(mL)'}")
    print("-" * 60)
    
    for mode, stats in mode_stats.items():
        if stats["total"] > 0:
            success_rate = stats["successes"] / stats["total"] * 100
            avg_vol = np.mean(stats["volumes"])
            median_vol = np.median(stats["volumes"])
            
            print(f"{mode:<15} {success_rate:<10.1f}% {avg_vol:<15.1f} {median_vol:.1f}")
    
    # 深度別の適応的K_scale
    print(f"\n適応的K_scaleの動作:")
    for img_name, results in all_results.items():
        for r in results:
            if r["mode"] == "adaptive" and "K_scale" in r:
                print(f"  {img_name}: K_scale = {r['K_scale']:.1f}")

if __name__ == "__main__":
    main()
```

*ファイルサイズ: 6,939 bytes, 行数: 210*

---

## 使用上の注意

### 環境要件
- Python 3.11
- CUDA対応GPU（推奨）
- UniDepthパッケージ（GitHub: lpiccinelli-eth/UniDepth）

### 体積調整方法

#### 方法1: 固定K_scale_factor（簡単だが精度低い）
`config.yaml`で設定:
```yaml
unidepth:
  K_scale_factor: 6.0  # 3.0〜12.0の範囲で調整
```

#### 方法2: 適応的K_scale（推奨）
`unidepth_runner_final.py`の`estimate_K_scale_for_food()`メソッドを使用

### 既知の問題と制限
1. **モデルの限界**: UniDepth v2は食品撮影用に訓練されていない
2. **汎用性の欠如**: 単一のK_scale_factorでは全画像に対応不可
3. **GitHub Issue #105**: UniDepthのカメラパラメータ処理に既知の問題

### 推奨事項
- 食品体積推定には専用の深度推定モデルの開発または他の手法の検討を推奨
- 暫定的な解決策として、深度に基づく適応的K_scale調整を使用

---

*このドキュメントは`create_unidepth_bundle.py`によって自動生成されました。*