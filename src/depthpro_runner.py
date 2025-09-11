"""
Depth Pro推論エンジン
Apple Depth Proモデルを使用した深度推定の実装
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from PIL import Image, ExifTags
import os
import sys
from src.camera_intrinsics import fx_fy_from_f35, validate_focal_length

# Depth Proパッケージのインポート
try:
    # ml-depth-pro/srcディレクトリをパスに追加
    depth_pro_path = os.path.join(os.path.dirname(__file__), "..", "ml-depth-pro", "src")
    if os.path.exists(depth_pro_path):
        sys.path.insert(0, depth_pro_path)
    
    import depth_pro
    DEPTH_PRO_AVAILABLE = True
    print("Depth Pro loaded successfully from:", depth_pro_path)
except ImportError as e:
    DEPTH_PRO_AVAILABLE = False
    print(f"Depth Pro import failed: {e}")
    # シミュレーション版で動作を継続
    pass


def read_f35_from_exif(img_pil: Image.Image) -> Optional[float]:
    """
    画像のEXIFデータから35mm換算焦点距離を取得
    
    Args:
        img_pil: PIL Image オブジェクト
    
    Returns:
        35mm換算焦点距離 [mm] または None（取得できない場合）
    """
    try:
        exif = img_pil.getexif()
        if not exif:
            return None
        
        # ExifIFDを取得
        exif_ifd = exif.get_ifd(0x8769)
        if not exif_ifd:
            return None
        
        # タグ名→値のマップを作成
        tag_map = {}
        for k, v in exif_ifd.items():
            tag_name = ExifTags.TAGS.get(k, str(k))
            tag_map[tag_name] = v
        
        # 35mm換算焦点距離のタグを探す（複数の可能性があるタグ名に対応）
        for key in ("FocalLengthIn35mmFilm", "FocalLenIn35mmFilm", "FocalLengthIn35mmFormat"):
            if key in tag_map:
                value = tag_map[key]
                if value:
                    try:
                        f35 = float(value)
                        if f35 > 0:
                            print(f"EXIFから35mm換算焦点距離を取得: {f35:.1f}mm")
                            return f35
                    except (TypeError, ValueError):
                        continue
        
        # 35mm換算が直接取得できない場合、通常の焦点距離から計算する可能性もあるが、
        # センサーサイズが不明な場合は推定が困難なので、ここではNoneを返す
        return None
        
    except Exception as e:
        print(f"EXIF読み取りエラー: {e}")
        return None


if not DEPTH_PRO_AVAILABLE:
    # Depth Proが利用できない場合はダミークラス
    class DepthProEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("Depth Proがインストールされていません。depthpro_runner_sim.pyを使用してください。")
else:
    class DepthProEngine:
        """Depth Proモデルを使用した深度推定エンジン"""
        
        def __init__(self, device: str = "cuda", checkpoint_path: Optional[str] = None):
            """
            Depth Proエンジンの初期化
            
            Args:
                device: 実行デバイス ("cuda" or "cpu")
                checkpoint_path: モデルチェックポイントのパス（オプション）
            """
            # デバイス設定
            if device == "cuda" and not torch.cuda.is_available():
                print("CUDAが利用できないため、CPUを使用します")
                device = "cpu"
            self.device = torch.device(device)
            print(f"Depth Proデバイス: {self.device}")
            
            # デフォルトのチェックポイントパスを設定
            if checkpoint_path is None:
                checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "depth_pro.pt")
            
            # モデルとtransformの作成
            print("Depth Proモデルをロード中...")
            try:
                # configを作成してチェックポイントを指定
                from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT, DepthProConfig
                
                if os.path.exists(checkpoint_path):
                    print(f"チェックポイントを使用: {checkpoint_path}")
                    config = DepthProConfig(
                        patch_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.patch_encoder_preset,
                        image_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.image_encoder_preset,
                        decoder_features=DEFAULT_MONODEPTH_CONFIG_DICT.decoder_features,
                        checkpoint_uri=checkpoint_path,
                        use_fov_head=DEFAULT_MONODEPTH_CONFIG_DICT.use_fov_head,
                        fov_encoder_preset=DEFAULT_MONODEPTH_CONFIG_DICT.fov_encoder_preset
                    )
                else:
                    print("デフォルトのチェックポイントを使用")
                    config = DEFAULT_MONODEPTH_CONFIG_DICT
                
                self.model, self.transform = depth_pro.create_model_and_transforms(
                    config=config,
                    device=self.device
                )
                
                self.model = self.model.to(self.device).eval()
                print("Depth Proモデルのロードが完了しました")
            except Exception as e:
                print(f"Depth Proモデルのロードに失敗しました: {e}")
                print("チェックポイントが必要な場合は以下を実行してください:")
                print("huggingface-cli download --local-dir checkpoints apple/DepthPro")
                raise
        
        def infer_image(self, image_path: str, force_fpx: Optional[float] = None) -> Dict[str, Any]:
            """
            画像から深度推定を実行
            
            Args:
                image_path: 入力画像のパス
                force_fpx: 強制的に使用する焦点距離 [pixels]（再推論用）
                
            Returns:
                以下のキーを含む辞書:
                - depth: 深度マップ (H,W) numpy配列 [m]
                - intrinsics: カメラ内部パラメータ行列 (3,3)
                - intrinsics_raw: 元のK（Depth Proでは同じ値）
                - points: 3D点群 (H,W,3) numpy配列 [m]
                - confidence: None (Depth Proは信頼度マップ未提供)
                - f35mm: EXIF 35mm換算焦点距離 [mm]（取得できた場合）
                - fpx_pred: モデル予測の焦点距離 [pixels]（ログ用）
                - fx: 横方向焦点距離 [pixels]
                - fy: 縦方向焦点距離 [pixels]
                - size: 画像サイズ (W, H)
            """
            print(f"画像を読み込み中: {image_path}")
            
            # 1. PIL画像として読み込み、EXIFデータから35mm換算焦点距離を取得
            img_pil = Image.open(image_path).convert("RGB")
            W_orig, H_orig = img_pil.size
            print(f"画像サイズ: {W_orig}x{H_orig}")
            
            # EXIFから35mm換算焦点距離を取得
            f35 = read_f35_from_exif(img_pil)
            fx, fy = None, None
            f_px_for_model = None
            
            # 強制焦点距離が指定されている場合（再推論用）
            if force_fpx is not None:
                f_px_for_model = force_fpx
                fx = force_fpx
                fy = fx * (H_orig / W_orig)
                print(f"強制焦点距離を使用: fx={fx:.1f}, fy={fy:.1f} pixels")
            elif f35 is not None and f35 > 0:
                # 35mm換算から正確なfx, fyを計算
                fx, fy = fx_fy_from_f35(W_orig, H_orig, f35)
                f_px_for_model = fx  # モデルには横方向の焦点距離を渡す
                print(f"35mm換算焦点距離: {f35:.1f}mm")
                print(f"計算された焦点距離: fx={fx:.1f}, fy={fy:.1f} pixels")
                
                # 焦点距離の妥当性チェック
                if not validate_focal_length(fx, fy, W_orig, H_orig):
                    print("焦点距離が異常な範囲です。モデル推定にフォールバックします。")
                    fx, fy = None, None
                    f_px_for_model = None
            else:
                print("EXIFから35mm換算焦点距離を取得できませんでした。モデル推定を使用します。")
            
            # 2. 画像をnumpy配列に変換してtransformを適用
            image_np = np.array(img_pil)
            
            # transformを適用（Depth Proの前処理）
            image_tensor = self.transform(image_np)
            
            # バッチ次元を追加してデバイスに送る
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor.to(self.device)
            
            # transformedイメージのサイズを取得
            _, _, H, W = image_tensor.shape
            print(f"推論用テンソルサイズ: {W}x{H}")
            
            # 3. Depth Pro推論を実行
            print("深度推定を実行中...")
            with torch.inference_mode():
                try:
                    # f_px_for_modelが計算できている場合は渡す
                    if f_px_for_model is not None:
                        # もしtransformでリサイズされている場合は、fx/fyを再計算
                        if W != W_orig or H != H_orig:
                            print(f"画像がリサイズされました: {W_orig}x{H_orig} → {W}x{H}")
                            # リサイズ後のサイズで再計算
                            if f35 is not None and f35 > 0:
                                fx, fy = fx_fy_from_f35(W, H, f35)
                            else:
                                # force_fpxの場合はスケール調整
                                scale = W / W_orig
                                fx = f_px_for_model * scale
                                fy = fx * (H / W)
                            f_px_for_model = fx
                            print(f"リサイズ後の焦点距離: fx={fx:.1f}, fy={fy:.1f} pixels")
                        
                        prediction = self.model.infer(image_tensor, f_px=f_px_for_model)
                        print(f"指定されたf_px={f_px_for_model:.1f}を使用して推論")
                    else:
                        # EXIFがない場合はモデルに推定させる
                        prediction = self.model.infer(image_tensor)
                        print("モデルによる焦点距離推定を使用")
                except TypeError as e:
                    # f_px引数がサポートされていない場合
                    print(f"f_px引数なしで推論を実行: {e}")
                    prediction = self.model.infer(image_tensor)
            
            # 4. 深度マップの取得
            depth_torch = prediction["depth"]
            if len(depth_torch.shape) == 3:  # (B,H,W)の場合
                depth_torch = depth_torch.squeeze(0)
            depth = depth_torch.cpu().numpy()
            print(f"深度マップサイズ: {depth.shape}")
            print(f"深度範囲: {depth.min():.3f}m - {depth.max():.3f}m")
            print(f"深度中央値: {np.median(depth):.3f}m")
            
            # 5. モデル予測の焦点距離を取得（ログ用途のみ）
            fpx_pred = None
            if "focallength_px" in prediction:
                fpx_pred = float(prediction["focallength_px"])
                print(f"モデル予測の焦点距離: {fpx_pred:.1f} pixels")
            
            # 6. カメラ内部パラメータ行列Kの構築
            H_final, W_final = depth.shape
            
            # fx, fyが計算できていない場合の処理
            if fx is None or fy is None:
                if fpx_pred is not None:
                    # モデル予測値を使用
                    fx = fpx_pred
                    # アスペクト比を保持してfyを計算
                    fy = fx * (H_final / W_final)
                    print(f"モデル予測値からの焦点距離: fx={fx:.1f}, fy={fy:.1f} pixels")
                else:
                    # デフォルト値（60度FOVと仮定）
                    fx = W_final / (2.0 * np.tan(np.deg2rad(60) / 2.0))
                    fy = fx * (H_final / W_final)
                    print(f"デフォルト焦点距離を使用: fx={fx:.1f}, fy={fy:.1f} pixels")
            
            # 光学中心
            cx, cy = W_final / 2.0, H_final / 2.0
            
            # K行列の構築（fx≠fyを正しく反映）
            K = np.array([[fx, 0,  cx],
                          [0,  fy, cy],
                          [0,  0,  1]], dtype=np.float64)
            
            print(f"カメラ内部パラメータ: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
            
            # 7. 3D点群の算出
            xyz = self._unproject_depth_to_xyz(depth, K)
            
            # 8. サニティチェック
            self._sanity_check(depth, K, xyz)
            
            # 9. 結果を返す
            result = {
                "depth": depth,
                "intrinsics": K,
                "intrinsics_raw": K.copy(),  # Depth Proでは同じ値
                "points": xyz,
                "confidence": None,  # Depth Proは信頼度マップ未提供
                "fx": fx,
                "fy": fy,
                "size": (W_final, H_final)
            }
            
            # 追加情報
            if f35 is not None:
                result["f35mm"] = f35
            if fpx_pred is not None:
                result["fpx_pred"] = fpx_pred
            
            return result
        
        def _unproject_depth_to_xyz(self, depth: np.ndarray, K: np.ndarray) -> np.ndarray:
            """
            深度マップを3D点群に変換
            
            Args:
                depth: 深度マップ (H,W) [m]
                K: カメラ内部パラメータ行列 (3,3)
                
            Returns:
                3D点群 (H,W,3) [m]
            """
            H, W = depth.shape
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # ピクセル座標のメッシュグリッド作成
            xx, yy = np.meshgrid(np.arange(W), np.arange(H))
            
            # カメラ座標系での3D座標を計算
            Z = depth
            X = (xx - cx) * Z / fx
            Y = (yy - cy) * Z / fy
            
            # 3D点群として結合
            xyz = np.stack([X, Y, Z], axis=-1)
            
            return xyz
        
        def _sanity_check(self, depth: np.ndarray, K: np.ndarray, xyz: np.ndarray):
            """
            推定結果のサニティチェック
            
            Args:
                depth: 深度マップ
                K: カメラ内部パラメータ
                xyz: 3D点群
            """
            # 深度の統計情報
            depth_median = np.median(depth)
            depth_mean = np.mean(depth)
            
            # 焦点距離の確認
            fx, fy = K[0, 0], K[1, 1]
            
            # 典型的な画素面積の計算（深度の中央値での）
            a_pix_typical = (depth_median ** 2) / (fx * fy)
            
            print("\n=== サニティチェック ===")
            print(f"深度の中央値: {depth_median:.3f}m")
            print(f"深度の平均値: {depth_mean:.3f}m")
            print(f"焦点距離: fx={fx:.2f}, fy={fy:.2f} pixels")
            print(f"典型的な画素面積 (深度中央値): {a_pix_typical:.9f} m^2")
            print(f"  = {a_pix_typical * 1e6:.3f} mm^2")
            
            # Depth Proでは正確なスケールなので、画素面積は小さくなることが期待される
            if a_pix_typical > 1e-5:
                print("注意: 画素面積が大きめです（ただしDepth Proでは問題ない可能性）")
            elif a_pix_typical < 1e-9:
                print("注意: 画素面積が非常に小さいです（解像度が高い可能性）")
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