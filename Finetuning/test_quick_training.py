#!/usr/bin/env python
"""
短時間トレーニングテスト - 1エポック、少数サンプルで動作確認
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 相対インポート
try:
    from Finetuning.datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
    from Finetuning.losses.silog import SiLogLoss
except ImportError:
    from datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
    from losses.silog import SiLogLoss

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def collate_rgbd(batch):
    imgs = [b[0] for b in batch]
    depths = torch.stack([b[1] for b in batch], 0)
    valids = torch.stack([b[2] for b in batch], 0)
    return imgs, depths, valids

def test_quick_training():
    print("=" * 50)
    print("Quick Training Test (1 epoch, 10 samples)")
    print("=" * 50)
    
    # 設定
    n5k_root = "nutrition5k/nutrition5k_dataset"
    model_id = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_samples = 10  # 少数サンプルでテスト
    
    # データ準備
    print("\n1. Preparing data...")
    train_ids, val_ids, _ = load_split_ids(n5k_root, val_ratio=0.1, seed=42)
    train_ids = train_ids[:num_samples]  # 10サンプルのみ
    val_ids = val_ids[:5]  # 5サンプルのみ
    
    train_ds = Nutrition5KDepthOnly(n5k_root, train_ids, depth_scale=1e-4, max_depth_m=10.0)
    val_ds = Nutrition5KDepthOnly(n5k_root, val_ids, depth_scale=1e-4, max_depth_m=10.0)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                            num_workers=2, pin_memory=True, collate_fn=collate_rgbd)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate_rgbd)
    
    print(f"✓ Train samples: {len(train_ds)}")
    print(f"✓ Val samples: {len(val_ds)}")
    
    # モデル準備
    print("\n2. Loading model...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)
    print(f"✓ Model loaded on {device}")
    
    # 最適化設定
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = SiLogLoss(lam=0.85)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    
    # トレーニング（1エポック）
    print("\n3. Training for 1 epoch...")
    model.train()
    losses = []
    
    for imgs, depths, valids in tqdm(train_loader, desc="Training"):
        # 前処理
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        depths = depths.to(device)
        valids = valids.to(device)
        
        # Forward
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = model(**inputs)
            pred = outputs.predicted_depth
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            
            # GT深度を予測サイズにリサイズ
            gt = F.interpolate(depths, size=pred.shape[-2:], mode="nearest")
            vm = F.interpolate(valids, size=pred.shape[-2:], mode="nearest")
            
            loss = criterion(pred, gt, vm)
        
        # Backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        losses.append(loss.item())
    
    avg_train_loss = sum(losses) / len(losses)
    print(f"✓ Average training loss: {avg_train_loss:.4f}")
    
    # 検証
    print("\n4. Validation...")
    model.eval()
    absrels = []
    
    with torch.no_grad():
        for imgs, depths, valids in tqdm(val_loader, desc="Validation"):
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            depths = depths.to(device)
            valids = valids.to(device)
            
            outputs = model(**inputs)
            pred = outputs.predicted_depth
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            
            gt = F.interpolate(depths, size=pred.shape[-2:], mode="nearest")
            vm = F.interpolate(valids, size=pred.shape[-2:], mode="nearest")
            
            mask = (vm > 0.5) & (gt > 0)
            if mask.sum() > 0:
                p, g = pred[mask], gt[mask]
                absrel = torch.mean(torch.abs(p - g) / torch.clamp(g, min=1e-6))
                absrels.append(absrel.item())
    
    if absrels:
        avg_absrel = sum(absrels) / len(absrels)
        print(f"✓ Average AbsRel: {avg_absrel:.4f}")
    
    print("\n" + "=" * 50)
    print("Quick training test completed successfully! ✓")
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = test_quick_training()
    sys.exit(0 if success else 1)