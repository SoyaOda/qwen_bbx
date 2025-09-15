#!/usr/bin/env python3
"""
異常に良い結果の原因を詳細調査
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

try:
    from Finetuning.datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
    from Finetuning.train_dav2_metric import collate_rgbd
except ImportError:
    from datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
    from train_dav2_metric import collate_rgbd

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

@torch.no_grad()
def analyze_predictions(model, processor, loader, device, model_name, n_samples=5):
    """予測値と真値を詳細分析"""
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"Analyzing {model_name} predictions...")
    print(f"{'='*60}")
    
    sample_count = 0
    
    for imgs, depths, valids in loader:
        if sample_count >= n_samples:
            break
            
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        depths = depths.to(device)
        valids = valids.to(device)
        
        out = model(**inputs)
        pred = out.predicted_depth
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        
        # 解像度を合わせる
        gt = F.interpolate(depths, size=pred.shape[-2:], mode="nearest")
        vm = F.interpolate(valids, size=pred.shape[-2:], mode="nearest")
        
        for i in range(len(imgs)):
            if sample_count >= n_samples:
                break
                
            mask = (vm[i] > 0.5) & (gt[i] > 0) & (gt[i] <= 10.0)
            
            if mask.sum() > 0:
                p = pred[i][mask]
                g = gt[i][mask]
                
                print(f"\nSample {sample_count + 1}:")
                print(f"  GT depth - min: {g.min().item():.4f}, max: {g.max().item():.4f}, mean: {g.mean().item():.4f}")
                print(f"  Pred depth - min: {p.min().item():.4f}, max: {p.max().item():.4f}, mean: {p.mean().item():.4f}")
                print(f"  Absolute difference - mean: {(p - g).abs().mean().item():.4f}")
                print(f"  Relative error: {((p - g).abs() / g.clamp(min=1e-6)).mean().item():.4f}")
                
                # スケールの比率を確認
                scale_ratio = (p.mean() / g.mean()).item()
                print(f"  Scale ratio (pred/gt): {scale_ratio:.4f}")
                
                sample_count += 1

def check_training_data():
    """訓練データとテストデータの統計を比較"""
    n5k_root = "./nutrition5k/nutrition5k_dataset"
    train_ids, val_ids, test_ids = load_split_ids(n5k_root, val_ratio=0.1, seed=42)
    
    print("\n" + "="*60)
    print("Dataset Statistics Check")
    print("="*60)
    
    # 訓練データの最初の10サンプル
    train_ds = Nutrition5KDepthOnly(n5k_root, train_ids[:10], 1e-4, 10.0)
    test_ds = Nutrition5KDepthOnly(n5k_root, test_ids[:10], 1e-4, 10.0)
    
    def get_stats(dataset, name):
        depths = []
        for i in range(len(dataset)):
            try:
                _, depth, valid = dataset[i]
                mask = (valid > 0.5) & (depth > 0) & (depth <= 10.0)
                if mask.sum() > 0:
                    depths.append(depth[mask].numpy())
            except:
                continue
        
        if depths:
            all_depths = np.concatenate(depths)
            print(f"\n{name} Set Statistics:")
            print(f"  Mean depth: {all_depths.mean():.4f} m")
            print(f"  Std depth: {all_depths.std():.4f} m")
            print(f"  Min depth: {all_depths.min():.4f} m")
            print(f"  Max depth: {all_depths.max():.4f} m")
            return all_depths
        return None
    
    train_depths = get_stats(train_ds, "Train")
    test_depths = get_stats(test_ds, "Test")
    
    if train_depths is not None and test_depths is not None:
        print(f"\nDistribution similarity:")
        print(f"  Mean difference: {abs(train_depths.mean() - test_depths.mean()):.4f} m")
        print(f"  Std difference: {abs(train_depths.std() - test_depths.std()):.4f} m")

def check_model_weights():
    """モデルの重みを確認"""
    print("\n" + "="*60)
    print("Model Weight Analysis")
    print("="*60)
    
    # Pretrainedモデル
    model_pre = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    
    # Finetunedモデル
    try:
        model_ft = AutoModelForDepthEstimation.from_pretrained("checkpoints/dav2_metric_n5k")
        
        # 最初と最後の層の重みを比較
        for name, param_pre in list(model_pre.named_parameters())[:5]:
            if name in dict(model_ft.named_parameters()):
                param_ft = dict(model_ft.named_parameters())[name]
                diff = (param_ft - param_pre).abs().mean().item()
                print(f"\n{name}:")
                print(f"  Weight difference: {diff:.6f}")
                print(f"  Pretrained mean: {param_pre.mean().item():.6f}")
                print(f"  Finetuned mean: {param_ft.mean().item():.6f}")
        
    except Exception as e:
        print(f"Error loading finetuned model: {e}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. データセット統計の確認
    check_training_data()
    
    # 2. モデル重みの確認
    check_model_weights()
    
    # 3. 予測値の詳細分析
    n5k_root = "./nutrition5k/nutrition5k_dataset"
    train_ids, val_ids, test_ids = load_split_ids(n5k_root, val_ratio=0.1, seed=42)
    
    # テストセットのローダー
    test_ds = Nutrition5KDepthOnly(n5k_root, test_ids[:10], 1e-4, 10.0)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=collate_rgbd)
    
    # Pretrainedモデルの分析
    processor_pre = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    model_pre = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf").to(device)
    
    analyze_predictions(model_pre, processor_pre, test_loader, device, "PRETRAINED")
    
    del model_pre
    torch.cuda.empty_cache()
    
    # Finetunedモデルの分析
    try:
        processor_ft = AutoImageProcessor.from_pretrained("checkpoints/dav2_metric_n5k")
        model_ft = AutoModelForDepthEstimation.from_pretrained("checkpoints/dav2_metric_n5k").to(device)
        
        analyze_predictions(model_ft, processor_ft, test_loader, device, "FINETUNED")
    except Exception as e:
        print(f"Error with finetuned model: {e}")
    
    print("\n" + "="*60)
    print("CONCLUSIONS:")
    print("="*60)
    print("If finetuned model predictions are very close to GT values,")
    print("it suggests the model may have learned dataset-specific patterns")
    print("rather than generalizable depth estimation.")

if __name__ == "__main__":
    main()