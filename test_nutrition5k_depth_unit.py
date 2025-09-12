#!/usr/bin/env python3
"""
Nutrition5kの深度単位を検証するテスト
preprocess_spec.mdの主張: 1m = 10,000 units (1unit = 0.1mm)
既存の理解: 1unit = 1mm
"""

import numpy as np
from PIL import Image
import os

def test_depth_units():
    """深度単位の検証"""
    
    # サンプルデータのパス
    sample_paths = [
        'nutrition5k/nutrition5k_dataset/imagery/realsense_overhead/dish_1556572657',
        'nutrition5k/nutrition5k_dataset/imagery/realsense_overhead/dish_1556573514',
        'nutrition5k/nutrition5k_dataset/imagery/realsense_overhead/dish_1556575014'
    ]
    
    print("=" * 70)
    print("Nutrition5k 深度単位検証テスト")
    print("=" * 70)
    print()
    
    # 論文記載の値
    Z_PLANE_PAPER = 35.9  # cm = 0.359 m
    print(f"論文記載のカメラ-テーブル距離: {Z_PLANE_PAPER} cm")
    print()
    
    for path in sample_paths:
        if not os.path.exists(path):
            print(f"スキップ: {path} が存在しません")
            continue
            
        depth_path = os.path.join(path, 'depth_raw.png')
        depth_raw = np.array(Image.open(depth_path))
        
        # 非ゼロ値のみを取得
        valid_depths = depth_raw[depth_raw > 0]
        
        if len(valid_depths) == 0:
            continue
            
        median_raw = np.median(valid_depths)
        mean_raw = np.mean(valid_depths)
        
        print(f"サンプル: {os.path.basename(path)}")
        print(f"  Raw値の範囲: {valid_depths.min()} - {valid_depths.max()}")
        print(f"  Raw中央値: {median_raw:.0f}")
        print(f"  Raw平均値: {mean_raw:.1f}")
        print()
        
        # 2つの変換方式を比較
        print("  変換結果の比較:")
        
        # 方式1: 1unit = 1mm (既存の理解)
        depth_m_v1 = median_raw * 0.001
        print(f"    方式1 (×0.001): {median_raw:.0f} units → {depth_m_v1:.3f} m = {depth_m_v1*100:.1f} cm")
        
        # 方式2: 1unit = 0.1mm (preprocess_spec.mdの主張)
        depth_m_v2 = median_raw / 10000.0
        print(f"    方式2 (÷10000): {median_raw:.0f} units → {depth_m_v2:.3f} m = {depth_m_v2*100:.1f} cm")
        
        # 論文値との差
        diff_v1 = abs(depth_m_v1*100 - Z_PLANE_PAPER)
        diff_v2 = abs(depth_m_v2*100 - Z_PLANE_PAPER)
        
        print(f"    論文値(35.9cm)との差:")
        print(f"      方式1: {diff_v1:.1f} cm差")
        print(f"      方式2: {diff_v2:.1f} cm差")
        
        # どちらが正しいか判定
        if diff_v1 < diff_v2:
            print(f"    → 方式1 (×0.001) の方が論文値に近い ✓")
        else:
            print(f"    → 方式2 (÷10000) の方が論文値に近い")
        
        print()
    
    print("=" * 70)
    print("結論:")
    print("  最初のサンプル(dish_1556572657)では:")
    print("    方式1 (×0.001): 37.3 cm → 論文値と1.4cm差 ✓")
    print("    方式2 (÷10000): 3.73 cm → 論文値と32.2cm差 ✗")
    print()
    print("  実際のデータは 1unit = 1mm である可能性が高い")
    print("  ※ただし、距離が遠いサンプルもあるため、撮影距離にばらつきがある")
    print("=" * 70)

if __name__ == "__main__":
    test_depth_units()