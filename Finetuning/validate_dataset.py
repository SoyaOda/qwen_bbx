"""データセットの整合性チェックとクリーンアップ"""

import os
from tqdm import tqdm

try:
    from Finetuning.datasets.nutrition5k_depthonly import load_split_ids
except ImportError:
    from datasets.nutrition5k_depthonly import load_split_ids

def validate_and_clean_dataset():
    n5k_root = "nutrition5k/nutrition5k_dataset"
    
    print("=" * 70)
    print("Validating Nutrition5K Dataset Integrity")
    print("=" * 70)
    
    # Split IDsを読み込み
    train_ids, val_ids, test_ids = load_split_ids(n5k_root, val_ratio=0.1, seed=42)
    all_ids = train_ids + val_ids + test_ids
    
    print(f"\nTotal IDs to check: {len(all_ids)}")
    
    # 各IDのファイル存在確認
    missing_rgb = []
    missing_depth = []
    missing_both = []
    valid_ids = []
    
    print("\nChecking file existence...")
    for dish_id in tqdm(all_ids):
        base_path = os.path.join(n5k_root, "imagery", "realsense_overhead", dish_id)
        rgb_path = os.path.join(base_path, "rgb.png")
        depth_path = os.path.join(base_path, "depth_raw.png")
        
        rgb_exists = os.path.exists(rgb_path)
        depth_exists = os.path.exists(depth_path)
        
        if not rgb_exists and not depth_exists:
            missing_both.append(dish_id)
        elif not rgb_exists:
            missing_rgb.append(dish_id)
        elif not depth_exists:
            missing_depth.append(dish_id)
        else:
            valid_ids.append(dish_id)
    
    # 結果を表示
    print("\n" + "=" * 70)
    print("Validation Results:")
    print(f"  Valid dishes (both RGB and Depth): {len(valid_ids)}")
    print(f"  Missing RGB only: {len(missing_rgb)}")
    print(f"  Missing Depth only: {len(missing_depth)}")
    print(f"  Missing both files: {len(missing_both)}")
    
    if missing_both:
        print(f"\nFirst 10 missing both:")
        for dish_id in missing_both[:10]:
            print(f"    - {dish_id}")
    
    # 問題のあるdish_1558109511を特別にチェック
    problem_id = "dish_1558109511"
    print(f"\nSpecific check for {problem_id}:")
    problem_path = os.path.join(n5k_root, "imagery", "realsense_overhead", problem_id)
    print(f"  Directory exists: {os.path.exists(problem_path)}")
    if os.path.exists(problem_path):
        print(f"  Contents: {os.listdir(problem_path)}")
    
    # クリーンなIDリストを保存
    if len(valid_ids) > 0:
        # train/val/testに分割
        valid_train = [id for id in train_ids if id in valid_ids]
        valid_val = [id for id in val_ids if id in valid_ids]
        valid_test = [id for id in test_ids if id in valid_ids]
        
        print(f"\nCleaned splits:")
        print(f"  Train: {len(train_ids)} -> {len(valid_train)}")
        print(f"  Val: {len(val_ids)} -> {len(valid_val)}")
        print(f"  Test: {len(test_ids)} -> {len(valid_test)}")
        
        # クリーンなIDリストを保存
        output_dir = "Finetuning/cleaned_splits"
        os.makedirs(output_dir, exist_ok=True)
        
        for name, ids in [("train", valid_train), ("val", valid_val), ("test", valid_test)]:
            output_file = os.path.join(output_dir, f"clean_{name}_ids.txt")
            with open(output_file, 'w') as f:
                for id in ids:
                    f.write(f"{id}\n")
            print(f"  Saved: {output_file}")
    
    print("=" * 70)
    
    return valid_train, valid_val, valid_test

if __name__ == "__main__":
    validate_and_clean_dataset()