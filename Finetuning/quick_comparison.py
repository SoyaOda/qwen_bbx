#!/usr/bin/env python3
"""
Pretrained vs Finetuned モデルの簡易比較
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

try:
    from Finetuning.datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
    from Finetuning.train_dav2_metric import collate_rgbd
except ImportError:
    from datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
    from train_dav2_metric import collate_rgbd

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

@torch.no_grad()
def evaluate_quick(model, processor, loader, device, model_name, max_samples=20):
    """簡易評価（最初のN個のサンプルのみ）"""
    model.eval()
    absrels = []
    rmses = []
    count = 0
    
    for imgs, depths, valids in loader:
        if count >= max_samples:
            break
            
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        depths = depths.to(device)
        valids = valids.to(device)
        
        out = model(**inputs)
        pred = out.predicted_depth
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        
        gt = F.interpolate(depths, size=pred.shape[-2:], mode="nearest")
        vm = F.interpolate(valids, size=pred.shape[-2:], mode="nearest")
        
        mask = (vm > 0.5) & (gt > 0) & (gt <= 10.0)
        if mask.sum() == 0:
            continue
        
        p, g = pred[mask], gt[mask]
        absrels.append(torch.mean(torch.abs(p - g) / torch.clamp(g, min=1e-6)).item())
        rmses.append(torch.sqrt(torch.mean((p - g) ** 2)).item())
        count += len(imgs)
    
    return np.mean(absrels), np.mean(rmses)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # データ準備
    n5k_root = "./nutrition5k/nutrition5k_dataset"
    train_ids, val_ids, test_ids = load_split_ids(n5k_root, val_ratio=0.1, seed=42)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_ids)} samples")
    print(f"  Val:   {len(val_ids)} samples")
    print(f"  Test:  {len(test_ids)} samples")
    
    # Test setのローダー
    test_ds = Nutrition5KDepthOnly(n5k_root, test_ids[:50], 1e-4, 10.0)  # 最初の50個
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=collate_rgbd)
    
    # Pretrainedモデル
    print("\n" + "="*60)
    print("Evaluating PRETRAINED model...")
    processor_pre = AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    model_pre = AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf").to(device)
    
    absrel_pre, rmse_pre = evaluate_quick(model_pre, processor_pre, test_loader, device, "Pretrained")
    print(f"Pretrained - AbsRel: {absrel_pre:.4f}, RMSE: {rmse_pre:.4f}")
    
    del model_pre
    torch.cuda.empty_cache()
    
    # Finetunedモデル
    print("\n" + "="*60)
    print("Evaluating FINETUNED model...")
    try:
        processor_ft = AutoImageProcessor.from_pretrained("checkpoints/dav2_metric_n5k")
        model_ft = AutoModelForDepthEstimation.from_pretrained("checkpoints/dav2_metric_n5k").to(device)
        
        absrel_ft, rmse_ft = evaluate_quick(model_ft, processor_ft, test_loader, device, "Finetuned")
        print(f"Finetuned - AbsRel: {absrel_ft:.4f}, RMSE: {rmse_ft:.4f}")
        
        # 比較
        print("\n" + "="*60)
        print("COMPARISON:")
        print("="*60)
        
        absrel_imp = ((absrel_pre - absrel_ft) / absrel_pre * 100)
        rmse_imp = ((rmse_pre - rmse_ft) / rmse_pre * 100)
        
        print(f"AbsRel improvement: {absrel_imp:.1f}%")
        print(f"RMSE improvement:   {rmse_imp:.1f}%")
        
        if absrel_imp > 50 or rmse_imp > 50:
            print("\n⚠️  WARNING: Improvements >50% are unusual!")
            print("Possible causes:")
            print("1. Overfitting to Nutrition5k specific patterns")
            print("2. The model learned dataset-specific biases")
            print("3. Indoor metric model is already well-suited for this data")
            
    except Exception as e:
        print(f"Error loading finetuned model: {e}")
        print("Make sure the model has been trained first.")

if __name__ == "__main__":
    main()