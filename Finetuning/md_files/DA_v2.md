了解です。**「Finetuning 専用コードは Finetuning/ フォルダに完結配置」**し、Nutrition5K の RGB–Depth のみを教師にして **Depth‑Anything V2 Metric‑Indoor‑Large（DAV2‑Metric）をシンプル（SiLog=scale‑invariant log RMSE のみ）**に微調整する実装一式を提示します。
モデルは Hugging Face の depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf を Transformers から読み込み、出力は DepthEstimatorOutput の predicted_depth を用います（HF公式ドキュメント・モデルカードに準拠）
Hugging Face
+1
。
Nutrition5K の overhead RGB‑D は rgb.png / depth_raw.png(16bit) 構成で、深度単位は “1 m = 10,000 units（= PNG 値 × 1e‑4 [m]）” です（データ配布 README / 論文の明記）
GitHub
+1
。
SiLog（SigLoss）は DAV2 の metric 深度学習系で一般的に用いられる損失で、DA‑V2 公式リポジトリにも metric depth 学習コードが含まれています（metric_depth ディレクトリ）
GitHub
。

ライセンス注意：DA‑V2 の Base/Large/Giant は CC‑BY‑NC‑4.0（研究非商用）です。商用の可能性がある場合はモデルカード・リポジトリのライセンスを必ず確認してください
GitHub
。

ディレクトリ構成（新規追加）
qwen_bbx/
├─ Finetuning/                      # ← ここにFT専用コードを完結配置
│  ├─ __init__.py
│  ├─ datasets/
│  │   └─ nutrition5k_depthonly.py  # Nutrition5K RGB-D の読み込み（深度のみ教師）
│  ├─ losses/
│  │   └─ silog.py                  # SiLog 損失
│  ├─ train_dav2_metric.py          # 学習スクリプト（SiLogのみ／マスクや体積は不使用）
│  ├─ eval_depth_n5k.py             # 検証（AbsRel / RMSE）
│  └─ README.md                     # 使い方ドキュメント
└─ data/
   └─ Nutrition5k/                  # ← データは既存配置（パスは可変）


実行は python -m Finetuning.train_dav2_metric ...（推奨）。単体実行 (python Finetuning/train_dav2_metric.py) でも動くよう相対 import を記述します。

依存関係（最小）
pip install -U torch torchvision transformers>=4.45.0 accelerate tqdm pillow opencv-python


HFの AutoModelForDepthEstimation / AutoImageProcessor で DAV2 を直接ロード可能。出力は DepthEstimatorOutput に predicted_depth フィールド（2Dか [B,1,H,W]）。入力前処理は ImageProcessor が担当します
Hugging Face
。

DAV2 Metric‑Indoor‑Large の HF 版モデル ID：depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf（カードに Transformers 使用例あり）
GitHub
。

実装コード
1) Dataset（Nutrition5K：深度のみ教師）

Finetuning/datasets/nutrition5k_depthonly.py

import os
import random
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class Nutrition5KDepthOnly(Dataset):
    """
    Nutrition5K overhead RGB-D を読み込み、RGB(PIL)・depth[m]・valid(>0) を返す。
    - 教師は深度のみ（マスク・体積は一切扱わない）
    - depth_raw.png は 16bit で 1 unit = 1e-4 [m]（引数で可変）
    """
    def __init__(self, root_dir: str, ids: List[str],
                 depth_scale: float = 1e-4, max_depth_m: float = 10.0):
        self.root = root_dir
        self.ids = ids
        self.depth_scale = float(depth_scale)
        self.max_depth = float(max_depth_m)

    def __len__(self): return len(self.ids)

    def _paths(self, dish_id: str) -> Tuple[str, str]:
        base = os.path.join(
            self.root, "imagery", "realsense_overhead", f"dish_{dish_id}")
        return os.path.join(base, "rgb.png"), os.path.join(base, "depth_raw.png")

    @staticmethod
    def _read_rgb(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _read_depth_m(self, path: str) -> np.ndarray:
        arr = np.array(Image.open(path))  # uint16 想定
        if arr.dtype != np.uint16: arr = arr.astype(np.uint16)
        depth_m = arr.astype(np.float32) * self.depth_scale
        # 無効値処理
        depth_m[(depth_m <= 0) | (depth_m > self.max_depth)] = 0.0
        return depth_m  # [H,W] in meters

    def __getitem__(self, idx):
        did = self.ids[idx]
        rgb_path, dep_path = self._paths(did)
        img = self._read_rgb(rgb_path)
        depth_m = self._read_depth_m(dep_path)
        depth_t = torch.from_numpy(depth_m).unsqueeze(0)  # [1,H,W]
        valid = (depth_t > 0).float()
        return img, depth_t, valid

def load_split_ids(n5k_root: str, val_ratio: float = 0.1, seed: int = 42):
    """ dish_ids/splits/depth_train_ids.txt, depth_test_ids.txt を読む """
    split_dir = os.path.join(n5k_root, "dish_ids", "splits")
    def _read(p):
        with open(p, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]
    train_all = _read(os.path.join(split_dir, "depth_train_ids.txt"))
    test_ids  = _read(os.path.join(split_dir, "depth_test_ids.txt"))
    rnd = random.Random(seed); rnd.shuffle(train_all)
    n_val = max(1, int(len(train_all) * val_ratio))
    val_ids = train_all[:n_val]; train_ids = train_all[n_val:]
    return train_ids, val_ids, test_ids


Nutrition5K の ファイル仕様と深度単位（1e‑4m） は配布 README と論文に記載（rgb.png/depth_raw.png、overhead撮影、units=10000/m）
GitHub
+1
。

2) SiLog 損失

Finetuning/losses/silog.py

import torch
import torch.nn as nn

class SiLogLoss(nn.Module):
    """ Scale-invariant log RMSE（SigLoss）。 """
    def __init__(self, lam: float = 0.85, eps: float = 1e-6):
        super().__init__(); self.lam = lam; self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
        # pred/target: [B,1,H,W] (m), valid: [B,1,H,W] (1=valid)
        mask = (valid > 0.5)
        if mask.sum() == 0:  # 安全策
            return pred.new_tensor(0.0, requires_grad=True)
        p = torch.log(torch.clamp(pred[mask], min=self.eps))
        t = torch.log(torch.clamp(target[mask], min=self.eps))
        d = p - t
        return torch.sqrt((d**2).mean() - self.lam * (d.mean()**2)) * 10.0


DA‑V2 公式リポジトリには metric depth 学習コード（metric_depth/）があり、SiLog 系の損失が一般的に使われます（本実装は最小構成として SiLog 単独）
GitHub
。

3) 学習スクリプト（SiLogのみ／マスク・体積は不使用）

Finetuning/train_dav2_metric.py

import os, argparse
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# 相対 import（-m 実行推奨）
from Finetuning.datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
from Finetuning.losses.silog import SiLogLoss

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
        depths = depths.to(device); valids = valids.to(device)

        out = model(**inputs)
        pred = out.predicted_depth
        if pred.dim() == 3: pred = pred.unsqueeze(1)  # [B,1,H',W']

        gt = F.interpolate(depths, size=pred.shape[-2:], mode="nearest")
        vm = F.interpolate(valids, size=pred.shape[-2:], mode="nearest")

        mask = (vm > 0.5) & (gt > 0) & (gt <= max_depth_m)
        if mask.sum() == 0: continue
        p, g = pred[mask], gt[mask]
        absrels.append(torch.mean(torch.abs(p - g) / torch.clamp(g, min=1e-6)).item())
        rmses.append(torch.sqrt(torch.mean((p - g)**2)).item())
    import numpy as np
    return (float(np.mean(absrels)) if absrels else float("nan"),
            float(np.mean(rmses)) if rmses else float("nan"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n5k_root", required=True, type=str)
    ap.add_argument("--hf_model", type=str,
        default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
    ap.add_argument("--depth_scale", type=float, default=1e-4)  # Nutrition5K 既定
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
    os.makedirs(args.out_dir, exist_ok=True)

    # Split 読み込み
    train_ids, val_ids, _ = load_split_ids(args.n5k_root, args.val_ratio, seed=42)

    # Dataset / Loader
    tr_ds = Nutrition5KDepthOnly(args.n5k_root, train_ids, args.depth_scale, args.max_depth_m)
    va_ds = Nutrition5KDepthOnly(args.n5k_root, val_ids,   args.depth_scale, args.max_depth_m)
    tr_ldr = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_rgbd)
    va_ldr = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate_rgbd)

    # HF Processor & Model（DAV2 Metric Indoor Large）
    processor = AutoImageProcessor.from_pretrained(args.hf_model)
    model = AutoModelForDepthEstimation.from_pretrained(args.hf_model).to(device)  # 出力: predicted_depth
    model.train()

    # Optim / Scheduler / Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_total = args.epochs * len(tr_ldr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, steps_total)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    criterion = SiLogLoss(lam=0.85)

    best_absrel = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(tr_ldr, desc=f"[Epoch {ep}/{args.epochs}]")
        for imgs, depths, valids in pbar:
            inputs = processor(images=imgs, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            depths = depths.to(device); valids = valids.to(device)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                out = model(**inputs)
                pred = out.predicted_depth
                if pred.dim() == 3: pred = pred.unsqueeze(1)
                gt = F.interpolate(depths, size=pred.shape[-2:], mode="nearest")
                vm = F.interpolate(valids, size=pred.shape[-2:], mode="nearest")
                loss = criterion(pred, gt, vm)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            optimizer.zero_grad(set_to_none=True); scheduler.step()
            pbar.set_postfix(loss=float(loss.detach().cpu().item()))

        # 検証
        absrel, rmse = evaluate(model, processor, va_ldr, device, args.max_depth_m)
        print(f"[Val] AbsRel={absrel:.4f} RMSE(m)={rmse:.4f}")
        if absrel < best_absrel:
            best_absrel = absrel
            model.save_pretrained(args.out_dir)       # HF 形式で保存
            processor.save_pretrained(args.out_dir)
            print(f"  -> saved best to {args.out_dir}")
    print("done. best AbsRel:", best_absrel)

if __name__ == "__main__":
    main()

4) 評価スクリプト（AbsRel / RMSE）

Finetuning/eval_depth_n5k.py

import argparse, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from Finetuning.datasets.nutrition5k_depthonly import Nutrition5KDepthOnly, load_split_ids
from Finetuning.train_dav2_metric import collate_rgbd
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import numpy as np

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n5k_root", required=True, type=str)
    ap.add_argument("--ckpt_dir", required=True, type=str)  # train の保存先
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--depth_scale", type=float, default=1e-4)
    ap.add_argument("--max_depth_m", type=float, default=10.0)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_ids, va_ids, te_ids = load_split_ids(args.n5k_root, val_ratio=0.1, seed=42)
    ids = va_ids if args.split == "val" else te_ids
    ds = Nutrition5KDepthOnly(args.n5k_root, ids, args.depth_scale, args.max_depth_m)
    ld = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True,
                    collate_fn=collate_rgbd)

    processor = AutoImageProcessor.from_pretrained(args.ckpt_dir)
    model = AutoModelForDepthEstimation.from_pretrained(args.ckpt_dir).to(device).eval()

    absrels, rmses = [], []
    for imgs, depths, valids in tqdm(ld, desc=f"[Eval:{args.split}]"):
        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        depths = depths.to(device); valids = valids.to(device)
        out = model(**inputs)
        pred = out.predicted_depth
        if pred.dim() == 3: pred = pred.unsqueeze(1)
        gt = F.interpolate(depths, size=pred.shape[-2:], mode="nearest")
        vm = F.interpolate(valids, size=pred.shape[-2:], mode="nearest")
        mask = (vm > 0.5) & (gt > 0) & (gt <= args.max_depth_m)
        if mask.sum() == 0: continue
        p, g = pred[mask], gt[mask]
        absrels.append(torch.mean(torch.abs(p - g) / torch.clamp(g, min=1e-6)).item())
        rmses.append(torch.sqrt(torch.mean((p - g)**2)).item())

    print(f"AbsRel(mean)={np.mean(absrels):.4f}, RMSE(m)(mean)={np.mean(rmses):.4f}")

if __name__ == "__main__":
    main()

5) README（抜粋）

Finetuning/README.md（抜粋）

# DAV2 Metric-Indoor-Large Fine-tuning (Nutrition5K, depth-only)

## Install
```bash
pip install -U torch torchvision transformers>=4.45.0 accelerate tqdm pillow opencv-python

Train
python -m Finetuning.train_dav2_metric \
  --n5k_root ./data/Nutrition5k \
  --hf_model depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf \
  --depth_scale 1e-4 \
  --batch_size 2 --epochs 5 \
  --out_dir checkpoints/dav2_metric_n5k

Eval
python -m Finetuning.eval_depth_n5k \
  --n5k_root ./data/Nutrition5k \
  --ckpt_dir checkpoints/dav2_metric_n5k \
  --split test --depth_scale 1e-4

Notes

Nutrition5K depth units: 1 m = 10,000 (i.e., PNG * 1e-4 m).

Model output: DepthEstimatorOutput.predicted_depth (Transformers).


---

## サニティチェック & 運用ヒント

1) **深度単位**：Nutrition5K は **1e‑4 m** が既定（`--depth_scale 1e-4`）。もし実配布が mm（×1e‑3）である派生版を使う場合は引数で変更してください。根拠は公式 README / CVPR 論文です:contentReference[oaicite:8]{index=8}。  
2) **前処理**：HF ImageProcessor が入力サイズ/正規化を内部で処理。**推論後に GT 深度を予測解像度へ補間**して損失・評価を行います（HF ドキュメントの出力仕様に従う）:contentReference[oaicite:9]{index=9}。  
3) **損失**：まずは **SiLog 単独**。必要なら L1 併用や正則化を追加。DA‑V2 の **metric depth 学習コードがリポジトリに含まれている**ので詳細実装の参考にできます:contentReference[oaicite:10]{index=10}。  
4) **VRAM 対策**：Large は重め。`--batch_size 1`、AMP は既に有効、必要なら `accelerate` 導入や勾配チェックポイントを追加。  
5) **ライセンス**：DA‑V2 Large は **CC‑BY‑NC‑4.0**。商用検討時は Small（Apache‑2.0）や別モデルも選択肢に:contentReference[oaicite:11]{index=11}。  

---

## 参考（一次情報）

- **HF Transformers: DepthAnythingV2**（出力 `DepthEstimatorOutput.predicted_depth`、post‑process API 等）:contentReference[oaicite:12]{index=12}  
- **HF モデルカード: DAV2 Metric‑Indoor‑Large (hf 版)**（Transformers 例・要件）:contentReference[oaicite:13]{index=13}  
- **Depth‑Anything V2 公式リポジトリ**（`metric_depth/`、ライセンス表記）:contentReference[oaicite:14]{index=14}  
- **Nutrition5K**（overhead RGB‑D、`depth_raw` の単位=1e‑4 [m]）:contentReference[oaicite:15]{index=15}  

---

### まとめ

- **Finetuning は `Finetuning/` のスクリプトのみで完結**（モデルは HF から読み込み、データは `data/Nutrition5k/` を参照）。  
- **マスク・体積は一切使わず**、**RGB–Depth の SiLog 監督のみ**で DAV2‑Metric を Nutrition5K に適応。  
- 学習後は、既存の `test_*` 系体積推定パイプラインへ **深度マップ置換**するだけで比較可能です（体積アルゴリズムは後段で調整）。  

必要であれば、このままコピペで `Finetuning/` を作成すれば動作します。実行時の細かなログ（AbsRel/RMSE 推移）や保存先はお好みで拡張してください。
::contentReference[oaicite:16]{index=16}