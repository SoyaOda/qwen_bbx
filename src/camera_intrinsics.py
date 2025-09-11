"""
カメラ内部パラメータの計算ユーティリティ
35mm換算焦点距離から正確な横/縦方向の焦点距離を算出
"""
import math
from typing import Optional, Tuple

def fx_fy_from_f35(width_px: int, height_px: int, f35_mm: float) -> Tuple[float, float]:
    """
    35mm換算焦点距離 f35_mm から、画像サイズ (W,H) に対する
    横方向/縦方向の焦点距離 [pixels] を厳密に算出します。
    
    35mmフルフレームセンサーのサイズ:
    - 横幅: 36mm
    - 縦幅: 24mm
    
    Args:
        width_px: 画像の横幅 [pixels]
        height_px: 画像の縦幅 [pixels]
        f35_mm: 35mm換算焦点距離 [mm]
    
    Returns:
        (fx, fy): 横方向と縦方向の焦点距離 [pixels]
    
    Note:
        fx = W * (f35 / 36)
        fy = H * (f35 / 24)
        
        これにより、横方向と縦方向で異なるピクセル密度を正確に反映できます。
        公式のutils.pyは対角ベースの換算を使用していますが、
        体積計算では横/縦を別々に扱う必要があるため、この方法が正確です。
    """
    # 横方向の焦点距離 [pixels]
    fx = width_px * (f35_mm / 36.0)
    
    # 縦方向の焦点距離 [pixels]
    fy = height_px * (f35_mm / 24.0)
    
    return float(fx), float(fy)


def fx_fy_from_fov(width_px: int, height_px: int, 
                   fov_x_deg: Optional[float] = None, 
                   fov_y_deg: Optional[float] = None) -> Tuple[float, float]:
    """
    視野角（FOV）から焦点距離を計算する代替方法
    
    Args:
        width_px: 画像の横幅 [pixels]
        height_px: 画像の縦幅 [pixels]
        fov_x_deg: 横方向の視野角 [degrees]（指定されない場合はfov_yから計算）
        fov_y_deg: 縦方向の視野角 [degrees]（指定されない場合はfov_xから計算）
    
    Returns:
        (fx, fy): 横方向と縦方向の焦点距離 [pixels]
    """
    if fov_x_deg is not None:
        # FOVから焦点距離を計算: f = (W/2) / tan(FOV_x/2)
        fx = width_px / (2.0 * math.tan(math.radians(fov_x_deg) / 2.0))
        
        if fov_y_deg is not None:
            fy = height_px / (2.0 * math.tan(math.radians(fov_y_deg) / 2.0))
        else:
            # アスペクト比を保持してfyを計算
            fy = fx * (height_px / width_px)
    
    elif fov_y_deg is not None:
        # 縦方向のFOVから計算
        fy = height_px / (2.0 * math.tan(math.radians(fov_y_deg) / 2.0))
        # アスペクト比を保持してfxを計算
        fx = fy * (width_px / height_px)
    
    else:
        raise ValueError("少なくともfov_x_degまたはfov_y_degのいずれかを指定してください")
    
    return float(fx), float(fy)


def validate_focal_length(fx: float, fy: float, width_px: int, height_px: int) -> bool:
    """
    焦点距離の妥当性をチェック
    
    Args:
        fx: 横方向の焦点距離 [pixels]
        fy: 縦方向の焦点距離 [pixels]
        width_px: 画像の横幅 [pixels]
        height_px: 画像の縦幅 [pixels]
    
    Returns:
        bool: 妥当な範囲内の場合True
    """
    # 一般的なカメラの視野角は20度〜120度程度
    # fx < W/2 の場合、FOV > 90度（超広角）
    # fx > 2*W の場合、FOV < 28度（望遠）
    
    if fx < width_px / 2:
        print(f"警告: fx={fx:.1f}は超広角相当です（FOV > 90度）")
        return False
    
    if fx > 2 * width_px:
        print(f"警告: fx={fx:.1f}は望遠相当です（FOV < 28度）")
        return False
    
    # fyも同様にチェック
    if fy < height_px / 2:
        print(f"警告: fy={fy:.1f}は超広角相当です（縦FOV > 90度）")
        return False
    
    if fy > 2 * height_px:
        print(f"警告: fy={fy:.1f}は望遠相当です（縦FOV < 28度）")
        return False
    
    # アスペクト比のチェック（通常は0.5〜2.0の範囲）
    aspect_ratio = (fx / width_px) / (fy / height_px)
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        print(f"警告: アスペクト比が異常です: {aspect_ratio:.2f}")
        return False
    
    return True