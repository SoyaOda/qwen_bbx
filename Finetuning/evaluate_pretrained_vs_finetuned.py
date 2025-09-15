#!/usr/bin/env python3
"""
ファインチューニング前後のモデル性能を比較
Pretrainedモデル vs Finetunedモデルの詳細比較
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

try:
    from Finetuning.datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
    from Finetuning.train_dav2_metric import collate_rgbd
except ImportError:
    from datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
    from train_dav2_metric import collate_rgbd

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

@torch.no_grad()
def evaluate_model(model, processor, loader, device, max_depth_m=10.0, model_name="Model"):
    """単一モデルの評価"""
    model.eval()
    
    results = {
        'absrels': [],
        'rmses': [],
        'sqrels': [],  # 二乗相対誤差
        'rmlogs': [],  # RMS対数誤差
        'delta1': [],  # δ < 1.25の割合
        'delta2': [],  # δ < 1.25^2の割合
        'delta3': [],  # δ < 1.25^3の割合
        'sample_count': 0,
        'failed_samples': 0
    }
    
    for imgs, depths, valids in tqdm(loader, desc=f"Evaluating {model_name}"):
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        depths = depths.to(device)
        valids = valids.to(device)
        
        try:
            out = model(**inputs)
            pred = out.predicted_depth
            if pred.dim() == 3:
                pred = pred.unsqueeze(1)
            
            # 解像度を合わせる
            gt = F.interpolate(depths, size=pred.shape[-2:], mode="nearest")
            vm = F.interpolate(valids, size=pred.shape[-2:], mode="nearest")
            
            # 有効ピクセルのマスク
            mask = (vm > 0.5) & (gt > 0) & (gt <= max_depth_m)
            
            if mask.sum() == 0:
                results['failed_samples'] += 1
                continue
            
            p, g = pred[mask], gt[mask]
            
            # 各種メトリクスの計算
            abs_rel = torch.mean(torch.abs(p - g) / torch.clamp(g, min=1e-6))
            sq_rel = torch.mean((p - g) ** 2 / torch.clamp(g, min=1e-6))
            rmse = torch.sqrt(torch.mean((p - g) ** 2))
            rmse_log = torch.sqrt(torch.mean((torch.log(p + 1e-6) - torch.log(g + 1e-6)) ** 2))
            
            # δメトリクス
            ratio = torch.maximum(p / torch.clamp(g, min=1e-6), 
                                g / torch.clamp(p, min=1e-6))
            delta1 = torch.mean((ratio < 1.25).float())
            delta2 = torch.mean((ratio < 1.25 ** 2).float())
            delta3 = torch.mean((ratio < 1.25 ** 3).float())
            
            results['absrels'].append(abs_rel.item())
            results['rmses'].append(rmse.item())
            results['sqrels'].append(sq_rel.item())
            results['rmlogs'].append(rmse_log.item())
            results['delta1'].append(delta1.item())
            results['delta2'].append(delta2.item())
            results['delta3'].append(delta3.item())
            results['sample_count'] += 1
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            results['failed_samples'] += 1
            continue
    
    # 平均値の計算
    for key in ['absrels', 'rmses', 'sqrels', 'rmlogs', 'delta1', 'delta2', 'delta3']:
        if results[key]:
            results[f'{key}_mean'] = float(np.mean(results[key]))
            results[f'{key}_std'] = float(np.std(results[key]))
            results[f'{key}_median'] = float(np.median(results[key]))
        else:
            results[f'{key}_mean'] = float('nan')
            results[f'{key}_std'] = float('nan')
            results[f'{key}_median'] = float('nan')
    
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n5k_root", required=True, type=str)
    ap.add_argument("--pretrained_model", type=str,
                    default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    ap.add_argument("--finetuned_dir", type=str,
                    default="checkpoints/dav2_metric_n5k")
    ap.add_argument("--split", choices=["val", "test", "train"], default="test")
    ap.add_argument("--depth_scale", type=float, default=1e-4)
    ap.add_argument("--max_depth_m", type=float, default=10.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--output_json", type=str, default="finetuning_comparison_results.json")
    args = ap.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # データセット準備
    print(f"Loading {args.split} split...")
    train_ids, val_ids, test_ids = load_split_ids(args.n5k_root, val_ratio=0.1, seed=42)
    
    if args.split == "train":
        ids = train_ids[:100]  # 訓練データは最初の100サンプルのみ（時間短縮）
        print(f"Using first 100 training samples for comparison")
    elif args.split == "val":
        ids = val_ids
    else:
        ids = test_ids
    
    print(f"Evaluating on {len(ids)} samples")
    
    ds = Nutrition5KDepthOnly(args.n5k_root, ids, args.depth_scale, args.max_depth_m)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True,
                       collate_fn=collate_rgbd)
    
    # Pretrainedモデルの評価
    print("\n" + "="*60)
    print("Loading PRETRAINED model...")
    processor_pre = AutoImageProcessor.from_pretrained(args.pretrained_model)
    model_pre = AutoModelForDepthEstimation.from_pretrained(args.pretrained_model).to(device)
    
    results_pre = evaluate_model(model_pre, processor_pre, loader, device, 
                                args.max_depth_m, "Pretrained")
    
    # メモリ解放
    del model_pre
    torch.cuda.empty_cache()
    
    # Finetunedモデルの評価
    print("\n" + "="*60)
    print("Loading FINETUNED model...")
    processor_ft = AutoImageProcessor.from_pretrained(args.finetuned_dir)
    model_ft = AutoModelForDepthEstimation.from_pretrained(args.finetuned_dir).to(device)
    
    results_ft = evaluate_model(model_ft, processor_ft, loader, device,
                               args.max_depth_m, "Finetuned")
    
    # 結果の比較と出力
    print("\n" + "="*60)
    print("COMPARISON RESULTS:")
    print("="*60)
    
    metrics = ['absrels', 'rmses', 'sqrels', 'rmlogs', 'delta1', 'delta2', 'delta3']
    
    for metric in metrics:
        pre_mean = results_pre[f'{metric}_mean']
        ft_mean = results_ft[f'{metric}_mean']
        
        if not np.isnan(pre_mean) and not np.isnan(ft_mean):
            improvement = ((pre_mean - ft_mean) / pre_mean * 100) if metric in ['absrels', 'rmses', 'sqrels', 'rmlogs'] else ((ft_mean - pre_mean) / pre_mean * 100)
            
            print(f"\n{metric.upper()}:")
            print(f"  Pretrained: {pre_mean:.4f} (±{results_pre[f'{metric}_std']:.4f})")
            print(f"  Finetuned:  {ft_mean:.4f} (±{results_ft[f'{metric}_std']:.4f})")
            
            if metric in ['absrels', 'rmses', 'sqrels', 'rmlogs']:
                # 小さいほど良い指標
                if ft_mean < pre_mean:
                    print(f"  Improvement: {abs(improvement):.1f}% ↓ (better)")
                else:
                    print(f"  Degradation: {abs(improvement):.1f}% ↑ (worse)")
            else:
                # 大きいほど良い指標（delta）
                if ft_mean > pre_mean:
                    print(f"  Improvement: {abs(improvement):.1f}% ↑ (better)")
                else:
                    print(f"  Degradation: {abs(improvement):.1f}% ↓ (worse)")
    
    # 結果をJSONに保存
    output = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'pretrained_results': results_pre,
        'finetuned_results': results_ft,
        'comparison': {
            metric: {
                'pretrained': results_pre[f'{metric}_mean'],
                'finetuned': results_ft[f'{metric}_mean'],
                'improvement_percent': ((results_pre[f'{metric}_mean'] - results_ft[f'{metric}_mean']) / results_pre[f'{metric}_mean'] * 100)
                    if metric in ['absrels', 'rmses', 'sqrels', 'rmlogs']
                    else ((results_ft[f'{metric}_mean'] - results_pre[f'{metric}_mean']) / results_pre[f'{metric}_mean'] * 100)
            }
            for metric in metrics
            if not np.isnan(results_pre[f'{metric}_mean']) and not np.isnan(results_ft[f'{metric}_mean'])
        }
    }
    
    with open(args.output_json, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n Results saved to {args.output_json}")
    
    # 異常検知
    print("\n" + "="*60)
    print("ANOMALY DETECTION:")
    print("="*60)
    
    suspicious = False
    
    # 1. 改善率が異常に高い場合
    for metric in ['absrels', 'rmses']:
        if metric in output['comparison']:
            imp = abs(output['comparison'][metric]['improvement_percent'])
            if imp > 50:  # 50%以上の改善は疑わしい
                print(f"⚠️  WARNING: {metric} improved by {imp:.1f}% - This is unusually high!")
                suspicious = True
    
    # 2. すべての指標が改善している場合
    all_improved = all(
        output['comparison'].get(m, {}).get('improvement_percent', 0) > 0
        for m in ['delta1', 'delta2', 'delta3']
    ) and all(
        output['comparison'].get(m, {}).get('improvement_percent', 0) > 0
        for m in ['absrels', 'rmses'] if m in output['comparison']
    )
    
    if all_improved:
        print("⚠️  WARNING: ALL metrics improved - This is unusual and may indicate overfitting or data issues!")
        suspicious = True
    
    # 3. 訓練/検証/テストでの性能差を確認
    if args.split == "test" and not suspicious:
        print("✅ Results appear reasonable for test set evaluation")
    
    if suspicious:
        print("\nRecommendations:")
        print("1. Check if the model is truly learning generalizable features")
        print("2. Verify the train/val/test split is correct")
        print("3. Consider evaluating on completely unseen data")
        print("4. Check for any preprocessing differences between training and evaluation")

if __name__ == "__main__":
    main()