"""破損した画像ファイルを検出して、クリーンなIDリストを作成"""

import os
from PIL import Image
import numpy as np
from tqdm import tqdm

try:
    from Finetuning.datasets.nutrition5k_depthonly import load_split_ids
except ImportError:
    from datasets.nutrition5k_depthonly import load_split_ids

def verify_image_file(path, is_depth=False):
    """画像ファイルが正常に読み込めるか確認"""
    try:
        if not os.path.exists(path):
            return False, "File not found"
        
        # ファイルサイズチェック
        if os.path.getsize(path) == 0:
            return False, "Empty file"
        
        # 画像として開けるか確認
        img = Image.open(path)
        
        # 深度画像の場合は追加チェック
        if is_depth:
            arr = np.array(img)
            if arr.size == 0:
                return False, "Empty array"
            # 16bit深度画像の確認
            if img.mode not in ['I', 'I;16', 'L']:
                return False, f"Unexpected mode: {img.mode}"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)

def verify_and_clean_dataset():
    n5k_root = "nutrition5k/nutrition5k_dataset"
    
    print("=" * 70)
    print("Verifying and Cleaning Nutrition5K Dataset")
    print("=" * 70)
    
    # オリジナルのSplit IDsを読み込み（use_clean=False）
    train_ids, val_ids, test_ids = load_split_ids(n5k_root, val_ratio=0.1, seed=42, use_clean=False)
    
    all_splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    clean_splits = {}
    corrupted_files = []
    
    for split_name, ids in all_splits.items():
        print(f"\n[{split_name}] Verifying {len(ids)} samples...")
        valid_ids = []
        
        for dish_id in tqdm(ids, desc=f"Checking {split_name}"):
            base_path = os.path.join(n5k_root, "imagery", "realsense_overhead", dish_id)
            rgb_path = os.path.join(base_path, "rgb.png")
            depth_path = os.path.join(base_path, "depth_raw.png")
            
            # RGBファイルのチェック
            rgb_ok, rgb_msg = verify_image_file(rgb_path, is_depth=False)
            # 深度ファイルのチェック
            depth_ok, depth_msg = verify_image_file(depth_path, is_depth=True)
            
            if rgb_ok and depth_ok:
                valid_ids.append(dish_id)
            else:
                error_info = f"{dish_id}: "
                if not rgb_ok:
                    error_info += f"RGB({rgb_msg}) "
                if not depth_ok:
                    error_info += f"Depth({depth_msg})"
                corrupted_files.append(error_info)
        
        clean_splits[split_name] = valid_ids
        print(f"  Valid: {len(valid_ids)}/{len(ids)} samples")
    
    # 破損ファイルの詳細を表示
    if corrupted_files:
        print(f"\n{len(corrupted_files)} corrupted samples found:")
        for i, error in enumerate(corrupted_files[:10]):  # 最初の10個だけ表示
            print(f"  {i+1}. {error}")
        if len(corrupted_files) > 10:
            print(f"  ... and {len(corrupted_files) - 10} more")
    
    # クリーンなIDリストを保存
    output_dir = "Finetuning/cleaned_splits_v2"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving clean ID lists to {output_dir}/")
    for split_name, ids in clean_splits.items():
        output_file = os.path.join(output_dir, f"clean_{split_name}_ids.txt")
        with open(output_file, 'w') as f:
            for dish_id in ids:
                f.write(f"{dish_id}\n")
        print(f"  {split_name}: {len(ids)} samples saved")
    
    # 統計情報
    print("\n" + "=" * 70)
    print("Summary:")
    total_orig = sum(len(ids) for ids in all_splits.values())
    total_clean = sum(len(ids) for ids in clean_splits.values())
    print(f"  Original total: {total_orig}")
    print(f"  Clean total: {total_clean}")
    print(f"  Removed: {total_orig - total_clean} ({100*(total_orig - total_clean)/total_orig:.1f}%)")
    print("=" * 70)
    
    return clean_splits

if __name__ == "__main__":
    verify_and_clean_dataset()