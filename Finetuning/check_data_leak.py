#!/usr/bin/env python3
"""
データリークのチェックスクリプト
train/val/testセット間でIDの重複がないか確認
"""

import os

def read_ids(filepath):
    """IDファイルを読み込んでセットを返す"""
    with open(filepath, 'r') as f:
        return set(line.strip() for line in f if line.strip())

def check_overlaps():
    """train/val/test間の重複をチェック"""
    base_dir = "Finetuning/cleaned_splits_v2"
    
    # 各セットのIDを読み込み
    train_ids = read_ids(os.path.join(base_dir, "clean_train_ids.txt"))
    val_ids = read_ids(os.path.join(base_dir, "clean_val_ids.txt"))
    test_ids = read_ids(os.path.join(base_dir, "clean_test_ids.txt"))
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_ids)} samples")
    print(f"  Val:   {len(val_ids)} samples")
    print(f"  Test:  {len(test_ids)} samples")
    print()
    
    # 重複チェック
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    
    print("Overlap analysis:")
    print(f"  Train ∩ Val:  {len(train_val_overlap)} samples")
    print(f"  Train ∩ Test: {len(train_test_overlap)} samples")
    print(f"  Val ∩ Test:   {len(val_test_overlap)} samples")
    print()
    
    # 全体の重複チェック
    all_ids = list(train_ids) + list(val_ids) + list(test_ids)
    unique_ids = set(all_ids)
    print(f"Total samples:  {len(all_ids)}")
    print(f"Unique samples: {len(unique_ids)}")
    print(f"Duplicates:     {len(all_ids) - len(unique_ids)}")
    
    # 重複があれば詳細を表示
    if train_test_overlap:
        print("\n⚠️  DATA LEAK DETECTED: Train and Test sets have overlapping IDs!")
        print("First 10 overlapping IDs:")
        for i, dish_id in enumerate(sorted(train_test_overlap)[:10]):
            print(f"  {i+1}. {dish_id}")
    
    if train_val_overlap:
        print("\n⚠️  Train and Val sets have overlapping IDs!")
        print("First 10 overlapping IDs:")
        for i, dish_id in enumerate(sorted(train_val_overlap)[:10]):
            print(f"  {i+1}. {dish_id}")
    
    if val_test_overlap:
        print("\n⚠️  Val and Test sets have overlapping IDs!")
        print("First 10 overlapping IDs:")
        for i, dish_id in enumerate(sorted(val_test_overlap)[:10]):
            print(f"  {i+1}. {dish_id}")
    
    if not (train_test_overlap or train_val_overlap or val_test_overlap):
        print("✅ No data leak detected! All sets are disjoint.")
    
    return len(train_test_overlap) == 0

if __name__ == "__main__":
    check_overlaps()