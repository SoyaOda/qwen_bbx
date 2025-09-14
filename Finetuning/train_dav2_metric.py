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
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default="checkpoints/dav2_metric_n5k")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    # Split 読み込み
    print(f"Loading data splits from {args.n5k_root}...")
    train_ids, val_ids, _ = load_split_ids(args.n5k_root, args.val_ratio, seed=42)
    print(f"Train: {len(train_ids)} samples, Val: {len(val_ids)} samples")

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

    best_absrel = float("inf")
    print(f"Starting training for {args.epochs} epochs...")
    
    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(tr_ldr, desc=f"[Epoch {ep}/{args.epochs}]")
        
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
            
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        # 検証
        print("Evaluating on validation set...")
        absrel, rmse = evaluate(model, processor, va_ldr, device, args.max_depth_m)
        print(f"[Val] AbsRel={absrel:.4f} RMSE(m)={rmse:.4f}")
        
        if absrel < best_absrel:
            best_absrel = absrel
            model.save_pretrained(args.out_dir)       # HF 形式で保存
            processor.save_pretrained(args.out_dir)
            print(f"  -> Saved best model to {args.out_dir}")
    
    print(f"Training complete. Best AbsRel: {best_absrel:.4f}")

if __name__ == "__main__":
    main()