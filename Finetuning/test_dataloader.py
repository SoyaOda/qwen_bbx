import sys
import os

# データローダーのテスト
try:
    from Finetuning.datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
except ImportError:
    from datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids

def test_dataloader():
    # Nutrition5kデータセットのパス
    n5k_root = "nutrition5k/nutrition5k_dataset"
    
    print("=" * 50)
    print("Testing Nutrition5K DataLoader")
    print("=" * 50)
    
    # 1. Split IDsの読み込みテスト
    print("\n1. Loading split IDs...")
    try:
        train_ids, val_ids, test_ids = load_split_ids(n5k_root, val_ratio=0.1, seed=42)
        print(f"✓ Train IDs: {len(train_ids)}")
        print(f"✓ Val IDs: {len(val_ids)}")
        print(f"✓ Test IDs: {len(test_ids)}")
        print(f"  Sample train IDs: {train_ids[:3]}")
    except Exception as e:
        print(f"✗ Failed to load split IDs: {e}")
        return False
    
    # 2. Datasetの作成テスト
    print("\n2. Creating dataset...")
    try:
        dataset = Nutrition5KDepthOnly(
            root_dir=n5k_root,
            ids=train_ids[:5],  # 最初の5サンプルだけテスト
            depth_scale=1e-4,
            max_depth_m=10.0
        )
        print(f"✓ Dataset created with {len(dataset)} samples")
    except Exception as e:
        print(f"✗ Failed to create dataset: {e}")
        return False
    
    # 3. データの読み込みテスト
    print("\n3. Loading sample data...")
    try:
        img, depth, valid = dataset[0]
        print(f"✓ Image type: {type(img)}")
        print(f"✓ Depth shape: {depth.shape}, dtype: {depth.dtype}")
        print(f"✓ Valid mask shape: {valid.shape}, dtype: {valid.dtype}")
        print(f"  Depth min: {depth.min():.4f}, max: {depth.max():.4f}")
        print(f"  Valid pixels: {valid.sum().item():.0f} / {valid.numel()}")
    except Exception as e:
        print(f"✗ Failed to load sample: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_dataloader()
    sys.exit(0 if success else 1)