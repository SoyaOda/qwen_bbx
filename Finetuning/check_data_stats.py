"""データセットの統計情報を確認"""

try:
    from Finetuning.datasets.nutrition5k_depthonly import load_split_ids
except ImportError:
    from datasets.nutrition5k_depthonly import load_split_ids

def check_data_stats():
    n5k_root = "nutrition5k/nutrition5k_dataset"
    
    print("=" * 60)
    print("Nutrition5K Dataset Statistics")
    print("=" * 60)
    
    # val_ratio=0.1でsplitを読み込み
    train_ids, val_ids, test_ids = load_split_ids(n5k_root, val_ratio=0.1, seed=42)
    
    print(f"\nData split with val_ratio=0.1:")
    print(f"  Train samples: {len(train_ids)}")
    print(f"  Val samples:   {len(val_ids)}")
    print(f"  Test samples:  {len(test_ids)}")
    print(f"  Total:         {len(train_ids) + len(val_ids) + len(test_ids)}")
    
    # バッチサイズ2での反復回数を計算
    batch_size = 2
    num_iterations = len(train_ids) // batch_size
    if len(train_ids) % batch_size != 0:
        num_iterations += 1
    
    print(f"\nWith batch_size={batch_size}:")
    print(f"  Iterations per epoch: {num_iterations}")
    
    # 元のsplitファイルの内容も確認
    import os
    split_dir = os.path.join(n5k_root, "dish_ids", "splits")
    
    def count_lines(filepath):
        with open(filepath, 'r') as f:
            return len([line for line in f if line.strip()])
    
    print(f"\nOriginal split files:")
    train_file = os.path.join(split_dir, "depth_train_ids.txt")
    test_file = os.path.join(split_dir, "depth_test_ids.txt")
    
    orig_train = count_lines(train_file)
    orig_test = count_lines(test_file)
    
    print(f"  depth_train_ids.txt: {orig_train} lines")
    print(f"  depth_test_ids.txt:  {orig_test} lines")
    print(f"  Total:               {orig_train + orig_test}")
    
    print("\nNote: val_ratio=0.1 means 10% of training data is used for validation")
    print(f"  Actual train: {orig_train} * 0.9 ≈ {int(orig_train * 0.9)}")
    print(f"  Actual val:   {orig_train} * 0.1 ≈ {int(orig_train * 0.1)}")
    
    print("=" * 60)

if __name__ == "__main__":
    check_data_stats()