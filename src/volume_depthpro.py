"""
Depth Pro用の体積計算関数
高さマップと画素面積の積分により体積を算出
"""
import numpy as np
from typing import Dict, Any, Optional

def pixel_area_map(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    各ピクセルが表す実世界の面積を計算
    
    単眼+ピンホール幾何の基本式:
    a_pix = Z^2 / (fx * fy)
    
    Args:
        depth: 深度マップ (H,W) [m]
        K: カメラ内部パラメータ (3,3)
    
    Returns:
        area: ピクセル面積マップ (H,W) [m^2/pixel]
    """
    fx, fy = K[0, 0], K[1, 1]
    
    # 面積計算（ゼロ除算回避）
    area = (depth ** 2) / (fx * fy + 1e-12)
    
    return area


def height_from_plane(depth: np.ndarray, K: np.ndarray, 
                      n: np.ndarray, d: float,
                      clip_negative: bool = True) -> np.ndarray:
    """
    平面からの高さマップを計算
    
    Args:
        depth: 深度マップ (H,W) [m]
        K: カメラ内部パラメータ (3,3)
        n: 平面法線（単位ベクトル）
        d: 平面パラメータ（n·p = d）
        clip_negative: 負の高さを0にクリップするか
    
    Returns:
        height: 高さマップ (H,W) [m]
    """
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # ピクセル座標グリッド
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # 3D座標を計算
    Z = depth
    X = (xx - cx) * Z / fx
    Y = (yy - cy) * Z / fy
    
    # 平面からの符号付き距離（n·p - d）
    height = (n[0] * X + n[1] * Y + n[2] * Z) - d
    
    # 負の高さをクリップ（テーブル下の点を0に）
    if clip_negative:
        height[height < 0] = 0.0
    
    return height


def integrate_volume(height: np.ndarray, a_pix: np.ndarray, 
                    mask: np.ndarray,
                    conf: Optional[np.ndarray] = None,
                    use_conf_weight: bool = False) -> Dict[str, Any]:
    """
    高さマップと画素面積から体積を積分
    
    Args:
        height: 高さマップ (H,W) [m]
        a_pix: 画素面積マップ (H,W) [m^2]
        mask: 積分対象マスク (H,W) bool
        conf: 信頼度マップ (H,W) [0,1]（オプション）
        use_conf_weight: 信頼度重み付けを使用するか
    
    Returns:
        結果辞書:
        - volume_m3: 体積 [m^3]
        - volume_mL: 体積 [mL]
        - height_mean_mm: 平均高さ [mm]
        - height_max_mm: 最大高さ [mm]
        - height_median_mm: 中央値高さ [mm]
        - area_m2: 投影面積 [m^2]
        - pixel_count: マスク内ピクセル数
    """
    # マスク内の値を取得
    h_masked = height[mask]
    a_masked = a_pix[mask]
    
    if len(h_masked) == 0:
        # マスクが空の場合
        return {
            "volume_m3": 0.0,
            "volume_mL": 0.0,
            "height_mean_mm": 0.0,
            "height_max_mm": 0.0,
            "height_median_mm": 0.0,
            "area_m2": 0.0,
            "pixel_count": 0
        }
    
    # 信頼度重み付け（オプション）
    if use_conf_weight and conf is not None:
        w_masked = conf[mask]
        # 重み付き体積計算
        volume_m3 = float(np.sum(h_masked * a_masked * w_masked))
        # 重み付き統計
        if np.sum(w_masked) > 0:
            height_mean = float(np.average(h_masked, weights=w_masked))
        else:
            height_mean = float(np.mean(h_masked))
    else:
        # 通常の体積計算
        volume_m3 = float(np.sum(h_masked * a_masked))
        height_mean = float(np.mean(h_masked))
    
    # 統計量の計算
    height_max = float(np.max(h_masked))
    height_median = float(np.median(h_masked))
    area_m2 = float(np.sum(a_masked))
    pixel_count = int(mask.sum())
    
    # 単位変換
    volume_mL = volume_m3 * 1e6  # m^3 → mL
    height_mean_mm = height_mean * 1000  # m → mm
    height_max_mm = height_max * 1000
    height_median_mm = height_median * 1000
    
    return {
        "volume_m3": volume_m3,
        "volume_mL": volume_mL,
        "height_mean_mm": height_mean_mm,
        "height_max_mm": height_max_mm,
        "height_median_mm": height_median_mm,
        "area_m2": area_m2,
        "pixel_count": pixel_count
    }


def sanity_check_volume(result: Dict[str, Any], 
                        food_type: Optional[str] = None,
                        verbose: bool = True) -> bool:
    """
    体積計算結果の妥当性をチェック
    
    Args:
        result: integrate_volumeの結果辞書
        food_type: 食品タイプ（"rice", "soup"など）
        verbose: 詳細出力
    
    Returns:
        妥当性フラグ
    """
    volume_mL = result["volume_mL"]
    height_mean_mm = result["height_mean_mm"]
    height_max_mm = result["height_max_mm"]
    area_m2 = result["area_m2"]
    
    # 食品タイプ別の期待範囲
    expected_ranges = {
        "rice": (50, 500, 10, 60),  # (min_vol_mL, max_vol_mL, min_height_mm, max_height_mm)
        "soup": (100, 800, 20, 100),
        "salad": (50, 400, 20, 80),
        "meat": (30, 300, 5, 40),
        "default": (10, 1000, 5, 150)
    }
    
    if food_type in expected_ranges:
        min_vol, max_vol, min_h, max_h = expected_ranges[food_type]
    else:
        min_vol, max_vol, min_h, max_h = expected_ranges["default"]
    
    issues = []
    
    # 体積チェック
    if volume_mL < min_vol:
        issues.append(f"体積が小さすぎる: {volume_mL:.1f}mL < {min_vol}mL")
    elif volume_mL > max_vol:
        issues.append(f"体積が大きすぎる: {volume_mL:.1f}mL > {max_vol}mL")
    
    # 高さチェック
    if height_mean_mm < min_h:
        issues.append(f"平均高さが低すぎる: {height_mean_mm:.1f}mm < {min_h}mm")
    elif height_mean_mm > max_h:
        issues.append(f"平均高さが高すぎる: {height_mean_mm:.1f}mm > {max_h}mm")
    
    # 最大高さチェック
    if height_max_mm > 200:
        issues.append(f"最大高さが異常: {height_max_mm:.1f}mm > 200mm")
    
    # 面積チェック（食器サイズの常識的範囲）
    area_cm2 = area_m2 * 1e4
    if area_cm2 < 10:
        issues.append(f"投影面積が小さすぎる: {area_cm2:.1f}cm² < 10cm²")
    elif area_cm2 > 500:
        issues.append(f"投影面積が大きすぎる: {area_cm2:.1f}cm² > 500cm²")
    
    is_valid = len(issues) == 0
    
    if verbose:
        print("\n体積計算の妥当性チェック:")
        print(f"  体積: {volume_mL:.1f} mL")
        print(f"  平均高さ: {height_mean_mm:.1f} mm")
        print(f"  最大高さ: {height_max_mm:.1f} mm")
        print(f"  投影面積: {area_cm2:.1f} cm²")
        
        if is_valid:
            print("  → ✓ 妥当な範囲内")
        else:
            print("  → ⚠ 問題あり:")
            for issue in issues:
                print(f"    - {issue}")
    
    return is_valid