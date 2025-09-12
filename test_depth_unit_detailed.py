#!/usr/bin/env python3
"""
深度単位の詳細な検証
サンプル間の違いから正しい単位を特定
"""

import numpy as np
from PIL import Image
import os

def analyze_depth_consistency():
    """すべてのサンプルで一貫した単位を見つける"""
    
    print("=" * 70)
    print("深度単位の詳細分析")
    print("=" * 70)
    print()
    
    # 論文の基準値
    Z_PLANE_PAPER = 35.9  # cm
    
    # より多くのサンプルを調査
    sample_paths = [
        'dish_1556572657',  # 最初のサンプル
        'dish_1556573514',
        'dish_1556575014',
        'dish_1556575083',
        'dish_1556575124'
    ]
    
    results = []
    
    for sample in sample_paths:
        path = f'nutrition5k/nutrition5k_dataset/imagery/realsense_overhead/{sample}'
        if not os.path.exists(path):
            continue
            
        depth_path = os.path.join(path, 'depth_raw.png')
        depth_raw = np.array(Image.open(depth_path))
        
        # 非ゼロ値の中央値
        valid_depths = depth_raw[depth_raw > 0]
        if len(valid_depths) == 0:
            continue
            
        median_raw = np.median(valid_depths)
        
        # 両方の変換を試す
        depth_cm_v1 = median_raw * 0.1  # 1unit = 1mm = 0.1cm
        depth_cm_v2 = median_raw / 100  # 1unit = 0.1mm = 0.01cm
        
        results.append({
            'sample': sample,
            'raw': median_raw,
            'v1_cm': depth_cm_v1,
            'v2_cm': depth_cm_v2
        })
    
    # 結果を分析
    print("サンプル分析:")
    print("-" * 70)
    print("サンプル        | Raw値  | v1(×0.1→cm) | v2(÷100→cm) | 判定")
    print("-" * 70)
    
    for r in results:
        # どちらが35.9cmに近いかチェック
        if abs(r['v1_cm'] - Z_PLANE_PAPER) < 5:  # 5cm以内なら正常
            status_v1 = "✓"
        else:
            status_v1 = "✗"
            
        if abs(r['v2_cm'] - Z_PLANE_PAPER) < 5:
            status_v2 = "✓"
        else:
            status_v2 = "✗"
            
        print(f"{r['sample'][:15]:15} | {r['raw']:6.0f} | {r['v1_cm']:11.1f} | {r['v2_cm']:11.1f} | v1:{status_v1} v2:{status_v2}")
    
    print("-" * 70)
    print()
    
    # パターンを識別
    print("パターン分析:")
    print("  - Raw値 < 1000 のサンプル: 単位は mm (×0.001でm変換)")
    print("  - Raw値 > 3000 のサンプル: 単位は 0.1mm (÷10000でm変換)")
    print()
    
    print("結論:")
    print("  データセットに2種類の単位が混在している可能性:")
    print("  1. 古いデータ: 1unit = 1mm")
    print("  2. 新しいデータ: 1unit = 0.1mm (README記載通り)")
    print()
    print("  → Raw値の大きさで判定する必要がある")
    print("     if raw_value < 1000: depth_m = raw * 0.001")
    print("     else: depth_m = raw / 10000.0")

if __name__ == "__main__":
    analyze_depth_consistency()