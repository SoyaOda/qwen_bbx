#!/usr/bin/env python3
"""
修正版: 時系列考慮型データ分割を使用したDAV2 Metric Finetuning
データリークを排除した適切な学習
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 相対 import（-m 実行推奨）
try:
    from Finetuning.datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
    from Finetuning.losses.silog import SiLogLoss
except ImportError:
    # 単体実行時のフォールバック
    from datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
    from losses.silog import SiLogLoss

from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from transformers import get_cosine_schedule_with_warmup

def collate_rgbd(batch):
    imgs = [b[0] for b in batch]
    depths = torch.stack([b[1] for b in batch], 0)  # [B,1,H,W]
    valids = torch.stack([b[2] for b in batch], 0)  # [B,1,H,W]
    return imgs, depths, valids

@torch.no_grad()
def evaluate(model, processor, loader, device, max_depth_m: float = 10.0):
    model.eval()
    absrels, rmses = [], []
    
    for imgs, depths, valids in loader:
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        depths = depths.to(device)
        valids = valids.to(device)

        out = model(**inputs)
        pred = out.predicted_depth
        if pred.dim() == 3: 
            pred = pred.unsqueeze(1)  # [B,1,H',W']

        gt = F.interpolate(depths, size=pred.shape[-2:], mode="nearest")
        vm = F.interpolate(valids, size=pred.shape[-2:], mode="nearest")

        mask = (vm > 0.5) & (gt > 0) & (gt <= max_depth_m)
        if mask.sum() == 0: 
            continue
            
        p, g = pred[mask], gt[mask]
        absrels.append(torch.mean(torch.abs(p - g) / torch.clamp(g, min=1e-6)).item())
        rmses.append(torch.sqrt(torch.mean((p - g)**2)).item())
    
    return (float(np.mean(absrels)) if absrels else float("nan"),
            float(np.mean(rmses)) if rmses else float("nan"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n5k_root", required=True, type=str,
                    help="Path to Nutrition5k dataset root")
    ap.add_argument("--hf_model", type=str,
        default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    ap.add_argument("--depth_scale", type=float, default=1e-4,
                    help="Nutrition5K depth scale (1e-4 for meter)")
    ap.add_argument("--max_depth_m", type=float, default=10.0)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3,  # 減らす（過学習防止）
                    help="Number of epochs (reduced to prevent overfitting)")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6,  # 学習率を下げる
                    help="Learning rate (reduced for stability)")
    ap.add_argument("--weight_decay", type=float, default=0.1,  # 正則化を強化
                    help="Weight decay (increased for regularization)")
    ap.add_argument("--warmup_steps", type=int, default=100)  # ウォームアップを減らす
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="checkpoints/dav2_metric_n5k_fixed")
    ap.add_argument("--use_temporal_split", type=bool, default=True,
                    help="Use temporal-aware data split to avoid data leak")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Split 読み込み（時系列考慮型を使用）
    print(f"Loading data splits from {args.n5k_root}...")
    print(f"Using temporal-aware split: {args.use_temporal_split}")
    
    train_ids, val_ids, test_ids = load_split_ids(
        args.n5k_root, args.val_ratio, seed=42, 
        use_clean=True, use_temporal=args.use_temporal_split
    )
    
    print(f"Train: {len(train_ids)} samples, Val: {len(val_ids)} samples, Test: {len(test_ids)} samples")
    
    # データリークチェック
    if args.use_temporal_split:
        print("\n✅ Using temporal-aware splits - No data leak between train/val/test")
    else:
        print("\n⚠️  Warning: Using standard splits - May contain data leak!")

    # Dataset / Loader
    tr_ds = Nutrition5KDepthOnly(args.n5k_root, train_ids, args.depth_scale, args.max_depth_m)
    va_ds = Nutrition5KDepthOnly(args.n5k_root, val_ids, args.depth_scale, args.max_depth_m)
    tr_ldr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_rgbd)
    va_ldr = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_rgbd)

    # HF Processor & Model（DAV2 Metric Indoor Large）
    print(f"Loading model: {args.hf_model}...")
    processor = AutoImageProcessor.from_pretrained(args.hf_model)
    model = AutoModelForDepthEstimation.from_pretrained(args.hf_model).to(device)
    model.train()

    # Optim / Scheduler / Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_total = args.epochs * len(tr_ldr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, steps_total)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    criterion = SiLogLoss(lam=0.85)

    # 初期性能の評価
    print("\nEvaluating initial performance...")
    init_absrel, init_rmse = evaluate(model, processor, va_ldr, device, args.max_depth_m)
    print(f"[Initial] AbsRel={init_absrel:.4f} RMSE(m)={init_rmse:.4f}")

    best_absrel = init_absrel
    patience = 2  # Early stoppingのための忍耐パラメータ
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("With regularization to prevent overfitting...")
    
    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(tr_ldr, desc=f"[Epoch {ep}/{args.epochs}]")
        losses = []
        
        for imgs, depths, valids in pbar:
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            depths = depths.to(device)
            valids = valids.to(device)
            
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(**inputs)
                pred = out.predicted_depth
                if pred.dim() == 3: 
                    pred = pred.unsqueeze(1)
                    
                gt = F.interpolate(depths, size=pred.shape[-2:], mode="nearest")
                vm = F.interpolate(valids, size=pred.shape[-2:], mode="nearest")
                loss = criterion(pred, gt, vm)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            losses.append(loss.detach().cpu().item())
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        # エポック終了後の統計
        avg_loss = np.mean(losses)
        print(f"[Epoch {ep}] Avg Loss: {avg_loss:.4f}")

        # 検証
        print("Evaluating on validation set...")
        absrel, rmse = evaluate(model, processor, va_ldr, device, args.max_depth_m)
        print(f"[Val] AbsRel={absrel:.4f} RMSE(m)={rmse:.4f}")
        
        # 改善率のチェック
        improvement = ((best_absrel - absrel) / best_absrel * 100) if best_absrel > 0 else 0
        print(f"  Improvement from best: {improvement:.1f}%")
        
        if absrel < best_absrel:
            best_absrel = absrel
            model.save_pretrained(args.out_dir)       # HF 形式で保存
            processor.save_pretrained(args.out_dir)
            print(f"  -> Saved best model to {args.out_dir}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {ep} epochs")
                break
    
    # 最終的な改善率
    final_improvement = ((init_absrel - best_absrel) / init_absrel * 100) if init_absrel > 0 else 0
    
    print(f"\nTraining complete.")
    print(f"Initial AbsRel: {init_absrel:.4f}")
    print(f"Best AbsRel: {best_absrel:.4f}")
    print(f"Total improvement: {final_improvement:.1f}%")
    
    if final_improvement > 50:
        print("\n⚠️  Warning: Improvement >50% may indicate remaining issues.")
        print("Consider evaluating on a completely different dataset.")
    else:
        print("\n✅ Improvement looks reasonable with temporal-aware splits.")

if __name__ == "__main__":
    main()