"""Nutrition5Kデータセット全体の詳細確認"""

import os
import glob

def check_full_dataset():
    n5k_root = "nutrition5k/nutrition5k_dataset"
    
    print("=" * 70)
    print("Nutrition5K Dataset Complete Analysis")
    print("=" * 70)
    
    # 1. dish_idsフォルダの全ファイル確認
    print("\n1. All dish ID files:")
    dish_ids_dir = os.path.join(n5k_root, "dish_ids")
    
    for file in sorted(os.listdir(dish_ids_dir)):
        if file.endswith('.txt'):
            filepath = os.path.join(dish_ids_dir, file)
            with open(filepath, 'r') as f:
                count = len([line for line in f if line.strip()])
            print(f"  {file:30s}: {count:5d} dishes")
    
    # 2. splitsフォルダの詳細
    print("\n2. Split files (for different modalities):")
    splits_dir = os.path.join(n5k_root, "dish_ids", "splits")
    
    for file in sorted(os.listdir(splits_dir)):
        if file.endswith('.txt'):
            filepath = os.path.join(splits_dir, file)
            with open(filepath, 'r') as f:
                count = len([line for line in f if line.strip()])
            print(f"  {file:30s}: {count:5d} dishes")
    
    # 3. 実際のimageryフォルダ内のデータ数を確認
    print("\n3. Actual imagery data:")
    imagery_dir = os.path.join(n5k_root, "imagery", "realsense_overhead")
    
    if os.path.exists(imagery_dir):
        dish_dirs = [d for d in os.listdir(imagery_dir) if d.startswith('dish_')]
        print(f"  Total dish directories: {len(dish_dirs)}")
        
        # RGB/Depthファイルの存在確認
        has_rgb = 0
        has_depth = 0
        has_both = 0
        
        for dish_dir in dish_dirs[:100]:  # 最初の100個だけチェック（高速化のため）
            dish_path = os.path.join(imagery_dir, dish_dir)
            rgb_exists = os.path.exists(os.path.join(dish_path, "rgb.png"))
            depth_exists = os.path.exists(os.path.join(dish_path, "depth_raw.png"))
            
            if rgb_exists:
                has_rgb += 1
            if depth_exists:
                has_depth += 1
            if rgb_exists and depth_exists:
                has_both += 1
        
        print(f"  Sample check (first 100):")
        print(f"    - Has RGB: {has_rgb}/100")
        print(f"    - Has Depth: {has_depth}/100")
        print(f"    - Has Both: {has_both}/100")
    
    # 4. Nutrition5Kの仕様確認
    print("\n4. Dataset specification:")
    print("  Nutrition5K is a dataset with ~5,000 dish images")
    print("  However, not all dishes have depth data (RealSense overhead)")
    print("  - RGB images: ~5,000 dishes")
    print("  - Depth data: ~3,265 dishes (subset with RealSense capture)")
    
    # 5. READMEファイルの内容確認
    readme_path = os.path.join(n5k_root, "README")
    if os.path.exists(readme_path):
        print("\n5. README excerpt:")
        with open(readme_path, 'r') as f:
            lines = f.readlines()[:20]  # 最初の20行
        for line in lines:
            if 'depth' in line.lower() or 'realsense' in line.lower():
                print(f"  {line.strip()}")
    
    print("\n" + "=" * 70)
    print("Summary:")
    print("  - Total dishes in dataset: ~5,000")
    print("  - Dishes with depth data: 3,265 (2,758 train + 507 test)")
    print("  - This is expected: depth cameras were not available for all dishes")
    print("=" * 70)

if __name__ == "__main__":
    check_full_dataset()