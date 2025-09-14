import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 相対 import
try:
    from Finetuning.datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
    from Finetuning.train_dav2_metric import collate_rgbd
except ImportError:
    # 単体実行時のフォールバック
    from datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
    from train_dav2_metric import collate_rgbd

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n5k_root", required=True, type=str,
                    help="Path to Nutrition5k dataset root")
    ap.add_argument("--ckpt_dir", required=True, type=str, 
                    help="Path to checkpoint directory")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--depth_scale", type=float, default=1e-4)
    ap.add_argument("--max_depth_m", type=float, default=10.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # データセット準備
    print(f"Loading {args.split} split...")
    tr_ids, va_ids, te_ids = load_split_ids(args.n5k_root, val_ratio=0.1, seed=42)
    ids = va_ids if args.split == "val" else te_ids
    print(f"Evaluating on {len(ids)} samples")
    
    ds = Nutrition5KDepthOnly(args.n5k_root, ids, args.depth_scale, args.max_depth_m)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True,
                    collate_fn=collate_rgbd)

    # モデル読み込み
    print(f"Loading model from {args.ckpt_dir}...")
    processor = AutoImageProcessor.from_pretrained(args.ckpt_dir)
    model = AutoModelForDepthEstimation.from_pretrained(args.ckpt_dir).to(device).eval()

    absrels, rmses = [], []
    
    for imgs, depths, valids in tqdm(ld, desc=f"[Eval:{args.split}]"):
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
        
        mask = (vm > 0.5) & (gt > 0) & (gt <= args.max_depth_m)
        if mask.sum() == 0: 
            continue
            
        p, g = pred[mask], gt[mask]
        absrels.append(torch.mean(torch.abs(p - g) / torch.clamp(g, min=1e-6)).item())
        rmses.append(torch.sqrt(torch.mean((p - g)**2)).item())

    print(f"\n{'='*50}")
    print(f"Results on {args.split} split:")
    print(f"AbsRel (mean): {np.mean(absrels):.4f}")
    print(f"RMSE(m) (mean): {np.mean(rmses):.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()