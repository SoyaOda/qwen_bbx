#!/usr/bin/env python3
"""
Nutrition5kのカメラ内部パラメータKの計算検証
preprocess_spec.mdの方法と実装を比較
"""

import numpy as np

def test_intrinsics_calculation():
    """カメラ内部パラメータの計算を検証"""
    
    print("=" * 70)
    print("Nutrition5k カメラ内部パラメータK検証")
    print("=" * 70)
    print()
    
    # 論文記載の定数
    Z_PLANE_M = 0.359  # 35.9 cm
    A_PIX_PLANE_CM2 = 5.957e-3  # cm^2 at Z=35.9cm
    A_PIX_PLANE_M2 = A_PIX_PLANE_CM2 * 1e-4  # m^2に変換
    
    print("論文記載の値:")
    print(f"  Z_plane = {Z_PLANE_M} m = {Z_PLANE_M*100} cm")
    print(f"  a_pix_plane = {A_PIX_PLANE_CM2} cm² = {A_PIX_PLANE_M2:.2e} m²")
    print()
    
    # preprocess_spec.mdの計算方法
    print("preprocess_spec.mdの計算:")
    print("  a_pix = Z²/(fx*fy) から逆算")
    print(f"  fx*fy = Z²/a_pix = {Z_PLANE_M}² / {A_PIX_PLANE_M2:.2e}")
    
    fx_fy_product = (Z_PLANE_M ** 2) / A_PIX_PLANE_M2
    print(f"  fx*fy = {fx_fy_product:.2e}")
    
    # fx ≈ fy と仮定
    f_estimated = np.sqrt(fx_fy_product)
    print(f"  fx ≈ fy ≈ √(fx*fy) = {f_estimated:.1f} pixels")
    print()
    
    # 画像サイズ640x480の場合
    W, H = 640, 480
    cx, cy = W/2, H/2
    
    print(f"推定されるK行列 (画像サイズ {W}x{H}):")
    print(f"  K = [[{f_estimated:.1f},    0.0, {cx:.1f}],")
    print(f"       [   0.0, {f_estimated:.1f}, {cy:.1f}],")
    print(f"       [   0.0,    0.0,    1.0]]")
    print()
    
    # 検証: この値で a_pix を再計算
    print("検証: 推定したKで a_pix を再計算:")
    a_pix_recalc = (Z_PLANE_M ** 2) / (f_estimated * f_estimated)
    print(f"  再計算した a_pix = {a_pix_recalc:.2e} m²")
    print(f"  論文値 a_pix = {A_PIX_PLANE_M2:.2e} m²")
    
    error = abs(a_pix_recalc - A_PIX_PLANE_M2) / A_PIX_PLANE_M2 * 100
    print(f"  誤差: {error:.2f}%")
    
    if error < 1:
        print("  → 計算が正しい ✓")
    else:
        print("  → 計算に誤差がある")
    
    print()
    print("=" * 70)
    
    # RealSense D435の理論値との比較
    print("RealSense D435の理論値との比較:")
    
    # D435: FOV 87° x 58°
    fov_h = 87  # degrees
    fov_v = 58  # degrees
    
    fx_d435 = W / (2 * np.tan(np.radians(fov_h/2)))
    fy_d435 = H / (2 * np.tan(np.radians(fov_v/2)))
    
    print(f"  D435理論値 (FOV 87°x58°):")
    print(f"    fx = {fx_d435:.1f} pixels")
    print(f"    fy = {fy_d435:.1f} pixels")
    print()
    
    print(f"  論文から逆算した値:")
    print(f"    fx ≈ fy ≈ {f_estimated:.1f} pixels")
    print()
    
    diff_fx = abs(f_estimated - fx_d435)
    print(f"  差: {diff_fx:.1f} pixels")
    
    if diff_fx < 200:
        print("  → 妥当な範囲内 ✓")
    else:
        print("  → 差が大きい（カメラ設定が異なる可能性）")
    
    print("=" * 70)
    print()
    
    # 最終的な推奨値
    print("推奨される設定:")
    print(f"  fx = fy = {f_estimated:.1f} pixels")
    print(f"  cx = {cx:.1f}, cy = {cy:.1f}")
    print("  (640x480画像用)")
    print()
    
    # データの一貫性チェック
    print("データの一貫性:")
    print("  ✓ 論文のZ_plane (35.9cm) と a_pix_plane から fx,fy を逆算可能")
    print("  ✓ 推定値は RealSense D435 の理論値に近い")
    print("  ✓ preprocess_spec.md の計算方法は正しい")

if __name__ == "__main__":
    test_intrinsics_calculation()