"""
焦点距離の最適化
EXIFがない場合に平面の水平性を利用して最適な焦点距離を探索
"""
import numpy as np
from typing import Tuple, List, Optional

def refine_fpx_by_flatness(depth: np.ndarray, K_base: np.ndarray, mask: np.ndarray,
                           fpx0: float, size_hw: Tuple[int, int],
                           try_scales: List[float] = None,
                           verbose: bool = True) -> float:
    """
    EXIFが無い場合、モデル予測のfpx0を基準に、
    Kだけ一時的にスケールして"リングの高さの中央値"が最小になるスケールを探す。
    
    注意: この段階では深度を再計算しない（近似）。
    ベスト比率が得られたら、最終的に infer(..., f_px=best_fx) で再推論して正式採用。
    
    Args:
        depth: 深度マップ (H,W) [m]
        K_base: 基準カメラ内部パラメータ (3,3)
        mask: 食品マスク (H,W) bool
        fpx0: 初期焦点距離 [pixels]
        size_hw: 画像サイズ (H, W)
        try_scales: 試すスケール係数のリスト
        verbose: 詳細出力
    
    Returns:
        best_fx: 最適化された焦点距離 [pixels]
    """
    from src.plane_fit_depthpro import estimate_table_plane, build_support_ring
    from src.volume_depthpro import height_from_plane
    
    if try_scales is None:
        try_scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5]
    
    H, W = size_hw
    cx, cy = W/2, H/2
    
    best = (None, np.inf)
    results = []
    
    # リング領域を構築
    ring = build_support_ring(mask, min_margin=0.04, max_margin=0.12, step=0.02)
    ring_pixels = np.sum(ring)
    
    if verbose:
        print(f"\n焦点距離の最適化:")
        print(f"  初期値: fpx={fpx0:.1f}")
        print(f"  リング領域: {ring_pixels}ピクセル")
    
    z_med = np.median(depth)
    
    for s in try_scales:
        fx = fpx0 * s
        fy = fx * (H / W)
        
        # 一時的なK行列
        K = np.array([[fx, 0,  cx],
                      [0,  fy, cy],
                      [0,  0,  1]], dtype=np.float64)
        
        try:
            # 平面推定
            n, d = estimate_table_plane(depth, K, mask, z_med=z_med, verbose=False)
            
            # 高さマップ計算
            h = height_from_plane(depth, K, n, d)
            
            # リング領域の高さ統計
            h_ring = h[ring]
            h_ring_positive = h_ring[h_ring > 0]
            
            if len(h_ring_positive) > 0:
                med_height = float(np.median(h_ring_positive))
                mean_height = float(np.mean(h_ring_positive))
                std_height = float(np.std(h_ring_positive))
            else:
                med_height = 0
                mean_height = 0
                std_height = 0
            
            # 評価指標：中央値を主に、標準偏差を副次的に考慮
            score = med_height + 0.1 * std_height
            
            results.append({
                'scale': s,
                'fx': fx,
                'med_height': med_height,
                'mean_height': mean_height,
                'std_height': std_height,
                'score': score,
                'n_z': n[2]
            })
            
            if score < best[1]:
                best = (fx, score)
            
            if verbose:
                print(f"  scale={s:.1f}: fx={fx:.1f}, "
                      f"高さ中央値={med_height*1000:.1f}mm, "
                      f"n_z={n[2]:.3f}, score={score*1000:.2f}")
        
        except Exception as e:
            if verbose:
                print(f"  scale={s:.1f}: 平面推定失敗 - {e}")
            continue
    
    if best[0] is None:
        if verbose:
            print(f"  最適化失敗、初期値を使用: {fpx0:.1f}")
        return fpx0
    
    # 最良のスケールを選択
    best_fx = best[0]
    
    # 追加の検証：極端な値でないかチェック
    if best_fx < fpx0 * 0.5 or best_fx > fpx0 * 2.0:
        if verbose:
            print(f"  警告: 最適値が極端（{best_fx:.1f}）、制限範囲に収めます")
        best_fx = np.clip(best_fx, fpx0 * 0.6, fpx0 * 1.8)
    
    if verbose:
        print(f"  最適化結果: fpx={best_fx:.1f} (初期値の{best_fx/fpx0:.2f}倍)")
    
    return best_fx


def estimate_fpx_from_scene(image_path: str, size_hw: Tuple[int, int],
                           scene_type: str = "food") -> Optional[float]:
    """
    シーンタイプに基づいて適切な焦点距離を推定
    
    Args:
        image_path: 画像パス
        size_hw: 画像サイズ (H, W)
        scene_type: シーンタイプ（"food", "landscape", "portrait"など）
    
    Returns:
        推定焦点距離 [pixels] またはNone
    """
    H, W = size_hw
    
    # シーンタイプ別の35mm換算焦点距離の典型値
    f35_typical = {
        "food": 35,        # 料理撮影は標準〜やや広角
        "portrait": 50,    # ポートレートは中望遠
        "landscape": 24,   # 風景は広角
        "macro": 60,       # マクロは望遠気味
        "indoor": 28,      # 室内は広角
    }
    
    f35 = f35_typical.get(scene_type, 35)  # デフォルトは35mm
    
    # 35mm換算から焦点距離を計算
    # fx = W * (f35 / 36)
    fx = W * (f35 / 36.0)
    
    return fx


def validate_fpx(fx: float, fy: float, size_hw: Tuple[int, int],
                verbose: bool = True) -> Tuple[bool, str]:
    """
    焦点距離の妥当性を検証
    
    Args:
        fx: 横方向焦点距離 [pixels]
        fy: 縦方向焦点距離 [pixels]
        size_hw: 画像サイズ (H, W)
        verbose: 詳細出力
    
    Returns:
        (is_valid, message): 妥当性とメッセージ
    """
    H, W = size_hw
    
    # FOVの計算
    fov_x = 2 * np.rad2deg(np.arctan(W / (2 * fx)))
    fov_y = 2 * np.rad2deg(np.arctan(H / (2 * fy)))
    
    # 35mm換算焦点距離の逆算
    f35_from_fx = fx * 36.0 / W
    
    messages = []
    is_valid = True
    
    # FOVチェック
    if fov_x > 90:
        messages.append(f"横FOVが広すぎる: {fov_x:.1f}度 > 90度")
        is_valid = False
    elif fov_x < 20:
        messages.append(f"横FOVが狭すぎる: {fov_x:.1f}度 < 20度")
        is_valid = False
    
    # 35mm換算チェック
    if f35_from_fx < 20:
        messages.append(f"35mm換算が超広角: {f35_from_fx:.1f}mm < 20mm")
        is_valid = False
    elif f35_from_fx > 100:
        messages.append(f"35mm換算が望遠: {f35_from_fx:.1f}mm > 100mm")
        is_valid = False
    
    # アスペクト比チェック
    aspect_ratio = (fx / W) / (fy / H)
    if aspect_ratio < 0.8 or aspect_ratio > 1.2:
        messages.append(f"アスペクト比が異常: {aspect_ratio:.2f}")
        is_valid = False
    
    if verbose:
        print(f"\n焦点距離の検証:")
        print(f"  fx={fx:.1f}, fy={fy:.1f}")
        print(f"  FOV: 横={fov_x:.1f}度, 縦={fov_y:.1f}度")
        print(f"  35mm換算: {f35_from_fx:.1f}mm")
        print(f"  アスペクト比: {aspect_ratio:.2f}")
        if messages:
            for msg in messages:
                print(f"  警告: {msg}")
    
    return is_valid, "; ".join(messages) if messages else "OK"