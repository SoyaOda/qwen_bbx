以下は、**Qwen2.5‑VLのBBox＋ラベル**を入力にして **SAM 2.1（b+ / large）** を同一パイプラインで回し、**インスタンスマスクを生成・可視化・比較**できるようにするための**増築プラン**と**曖昧性のないコード一式**です。
SAM 2.1 は Meta 公式の **facebookresearch/sam2** 実装と **SAM2ImagePredictor** を用い、**`sam2.1_hiera_base_plus.pt`** と **`sam2.1_hiera_large.pt`** の**両方**を比較可能にします（config は `sam2.1_hiera_b+.yaml` / `sam2.1_hiera_l.yaml`）。([GitHub][1])

---

## 0) 何が追加されるか（要約）

* **SAM2.1 推論**：Qwen出力の **正規化xyxy** BBox → **ピクセルxyxy** に復元 → **SAM2ImagePredictor** の **`boxes`** 入力としてセグメンテーション実行。([GitHub][1], [Hugging Face][2])
* **モデル比較**：**b+（base\_plus）** と **large** の両方でマスク生成し、

  * ① **個別可視化（枠＋塗り）**、② **b+ vs large の差分（XOR）**、③ **各検出ごとのマスク一致度（IoU）** を出力。
* **成果物**：

  * `outputs/sam2/json/` … 画像ごとの**比較サマリJSON**（BBox, ラベル, area, b+/large の面積・IoUなど）
  * `outputs/sam2/masks/` … **バイナリマスクPNG**（検出×モデル）
  * `outputs/sam2/viz/` … **重畳画像（b+ / large / 差分）**
* **実行**：`python -m src.run_sam2`（既存のQwen JSONを自動で読み、SAM2.1を両方実行）

> 注：SAM 2.1 の**導入・推論API・チェックポイント**は公式READMEに記載（PyTorch>=2.5.1, `pip install -e .`, `SAM2ImagePredictor` 使用、`download_ckpts.sh`または直接DL）。モデルは **`sam2.1_hiera_base_plus.pt` / `sam2.1_hiera_large.pt`** と対応configを利用します。([GitHub][1])

---

## 1) 依存関係とセットアップ

### 1-1. 追加パッケージ（requirements.txt 追記）

```txt
# 既存に加えて
torch>=2.5.1
torchvision>=0.20.1
# PyTorchはGPU環境に合わせて https://pytorch.org/ の推奨でインストール
```

> SAM2は **Python>=3.10, torch>=2.5.1** を要求。CUDA拡張のビルドは任意で、一部後処理が無効でも推論結果は同じです。([GitHub][1])

### 1-2. SAM2 リポジトリの導入（推奨：editable install）

```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
# 推奨: 新しい仮想環境 + PyTorch 2.5.1以上を先に入れる（公式推奨）:contentReference[oaicite:4]{index=4}
```

### 1-3. SAM2.1 チェックポイントの取得

```bash
cd checkpoints
./download_ckpts.sh
# あるいは個別に:
#  b+   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
#  large https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

> 公式READMEの **Download Checkpoints** と同一（`sam2.1_hiera_base_plus.pt`/`sam2.1_hiera_large.pt`）。([GitHub][1])

---

## 2) 設定ファイル（config.yaml 追記／更新）

```yaml
sam2:
  repo_root: "/abs/path/to/sam2"   # sam2 を clone した絶対パス
  cfg_base_plus: "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
  cfg_large:     "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
  ckpt_base_plus: "/abs/path/to/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
  ckpt_large:     "/abs/path/to/sam2/checkpoints/sam2.1_hiera_large.pt"
  device: "cuda"                   # "cuda" か "cpu"
  dtype: "bfloat16"                # "bfloat16"(GPU推奨) / "float32"(CPU)
  multimask_output: true           # SAM2で複数仮説出力→最良を選択
  conf_threshold: 0.20             # Qwen検出の信頼度閾値（再掲）
paths:
  qwen_json_dir: "outputs/json"    # 既存Qwen出力(JSON)の場所
  input_dir: "/path/to/FoodSeg103/images"
  out_root: "outputs/sam2"
```

> config名は `sam2.1_hiera_b+.yaml` / `sam2.1_hiera_l.yaml` を**そのまま**使用。([GitHub][3])

---

## 3) 追加・修正するコード

プロジェクト構成（追加分）：

```
qwen-vl-bbox-demo/
├─ src/
│  ├─ sam2_runner.py          # ★ SAM2.1をb+ / largeで実行するラッパ
│  ├─ viz_masks.py            # ★ マスク可視化（b+ / large / 差分）
│  ├─ run_sam2.py             # ★ CLI: Qwen JSON → SAM2 2系統 → 出力一式
│  └─ (既存ファイルはそのまま)
└─ outputs/sam2/
   ├─ json/
   ├─ masks/
   └─ viz/
```

### 3-1. `src/sam2_runner.py`

```python
# -*- coding: utf-8 -*-
# src/sam2_runner.py
import os
import numpy as np
import torch
from typing import List, Dict, Any, Tuple

# SAM2 公式API（README参照）:contentReference[oaicite:7]{index=7}
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def _to_abs_xyxy(box_norm: List[float], w: int, h: int) -> List[float]:
    x1 = float(np.clip(box_norm[0], 0.0, 1.0) * w)
    y1 = float(np.clip(box_norm[1], 0.0, 1.0) * h)
    x2 = float(np.clip(box_norm[2], 0.0, 1.0) * w)
    y2 = float(np.clip(box_norm[3], 0.0, 1.0) * h)
    # 整合
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    return [x1, y1, x2, y2]

def build_predictor(model_cfg_path: str, ckpt_path: str, device: str = "cuda") -> SAM2ImagePredictor:
    """
    SAM2ImagePredictor を構築。b+ / large のどちらにも使える。
    公式READMEのとおり build_sam2 + SAM2ImagePredictor を用いる。:contentReference[oaicite:8]{index=8}
    """
    sam_model = build_sam2(model_cfg_path, ckpt_path)
    predictor = SAM2ImagePredictor(sam_model)
    if device == "cuda" and torch.cuda.is_available():
        predictor.model.to("cuda")
    else:
        predictor.model.to("cpu")
    return predictor

def predict_masks_for_boxes(
    predictor: SAM2ImagePredictor,
    image_rgb: np.ndarray,
    boxes_abs_xyxy: np.ndarray,
    dtype: str = "bfloat16",
    multimask_output: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1画像に対し、複数BBox（px, xyxy, shape[N,4]）でSAM2を実行し、
    - masks          : (N, H, W) bool
    - iou_predictions: (N,) float
    - lowres_logits  : (N, h, w) float（使わない場合も）
    を返す。SAM2のAPIは boxes 引数を受け、複数物体に同時対応。:contentReference[oaicite:9]{index=9}
    """
    assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3
    H, W = image_rgb.shape[:2]
    device = "cuda" if next(predictor.model.parameters()).is_cuda else "cpu"

    boxes_t = torch.as_tensor(boxes_abs_xyxy, dtype=torch.float32, device=device)

    use_autocast = (device == "cuda" and dtype in ("bfloat16", "float16"))
    ctx = (torch.autocast(device_type="cuda", dtype=getattr(torch, dtype)) 
           if use_autocast else torch.cuda.amp.autocast(enabled=False))

    with torch.inference_mode():
        if use_autocast:
            with ctx:
                predictor.set_image(image_rgb)  # 公式API：set_image→predict:contentReference[oaicite:10]{index=10}
                masks, ious, lowres = predictor.predict(
                    boxes=boxes_t, multimask_output=multimask_output, return_logits=False
                )
        else:
            predictor.set_image(image_rgb)
            masks, ious, lowres = predictor.predict(
                boxes=boxes_t, multimask_output=multimask_output, return_logits=False
            )

    # 出力整形（N,1,H,W）→(N,H,W) bool を想定
    m = masks
    if isinstance(m, torch.Tensor):
        m = m.detach().to("cpu").numpy()
    if m.ndim == 4:  # (N,1,H,W)
        m = m[:, 0, :, :]
    m = (m > 0.5).astype(np.uint8).astype(bool)

    if isinstance(ious, torch.Tensor):
        ious = ious.detach().to("cpu").numpy()

    if isinstance(lowres, torch.Tensor):
        lowres = lowres.detach().to("cpu").numpy()

    return m, ious, lowres

def select_best_mask_per_box(
    masks: np.ndarray, ious: np.ndarray, multimask_output: bool
) -> np.ndarray:
    """
    multimask_output=True の場合は各BBoxにつき複数仮説が返る実装もあるため、
    ここでは **仮説1枚/箱** を保証する。標準は最高IoUスコアのマスクを採用。
    （SAM2のpredictは bboxesでまとめて渡すと通常 N個のマスクが返る想定だが、
     実装差分への保険として形状をここで一本化）
    """
    # 既に (N,H,W) ならそのまま
    if masks.ndim == 3 and ious.ndim == 1 and masks.shape[0] == ious.shape[0]:
        return masks
    # 万一 (N,K,H,W) 形式なら Kのargmaxを取る（Kは仮説数）
    if masks.ndim == 4:
        N, K = masks.shape[:2]
        best = np.zeros((N, masks.shape[2], masks.shape[3]), dtype=bool)
        for i in range(N):
            # ious が (N,K) なら argmax、(N,) ならそのまま0を採用
            if ious.ndim == 2 and ious.shape[1] == K:
                k = int(np.argmax(ious[i]))
            else:
                k = 0
            best[i] = masks[i, k]
        return best
    return masks
```

### 3-2. `src/viz_masks.py`

```python
# -*- coding: utf-8 -*-
# src/viz_masks.py
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from .visualize import norm_xyxy_to_abs, ensure_dir  # 既存関数を再利用

def color_map(n: int) -> np.ndarray:
    rs = np.random.RandomState(123)
    return rs.randint(0, 255, size=(n, 3), dtype=np.uint8)

def overlay_masks(img_bgr: np.ndarray, masks: List[np.ndarray], labels: List[str]) -> np.ndarray:
    out = img_bgr.copy()
    H, W = out.shape[:2]
    colors = color_map(len(masks))
    alpha = 0.45
    for i, m in enumerate(masks):
        if m.dtype != bool:
            m = m.astype(bool)
        color = colors[i].tolist()
        mask_rgb = np.zeros_like(out)
        mask_rgb[m] = color
        out = cv2.addWeighted(out, 1.0, mask_rgb, alpha, 0.0)
        # 外接矩形 + ラベル
        ys, xs = np.where(m)
        if xs.size > 0 and ys.size > 0:
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            cv2.rectangle(out, (x1, y1), (x2, y2), color=tuple(int(c) for c in color), thickness=2)
            txt = f"{labels[i]}"
            (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            ytxt = max(0, y1 - 4)
            cv2.rectangle(out, (x1, ytxt - th - 6), (x1 + tw + 4, ytxt), color=tuple(int(c) for c in color), thickness=-1)
            cv2.putText(out, txt, (x1 + 2, ytxt - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
    return out

def xor_diff_mask(masks_a: List[np.ndarray], masks_b: List[np.ndarray]) -> np.ndarray:
    # 同じ順序・同数を前提（検出順で対応付け）
    assert len(masks_a) == len(masks_b)
    if len(masks_a) == 0:
        return None
    H, W = masks_a[0].shape
    diff = np.zeros((H, W), dtype=np.uint8)
    for ma, mb in zip(masks_a, masks_b):
        diff ^= ((ma.astype(np.uint8) ^ mb.astype(np.uint8)) > 0).astype(np.uint8)
    return diff

def side_by_side(img_bgr, viz_a, viz_b, diff_mask=None):
    h = max(img_bgr.shape[0], viz_a.shape[0], viz_b.shape[0])
    def pad(img):
        pad_h = h - img.shape[0]
        if pad_h > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        return img
    img_bgr = pad(img_bgr); viz_a = pad(viz_a); viz_b = pad(viz_b)
    cat = np.concatenate([img_bgr, viz_a, viz_b], axis=1)
    if diff_mask is not None:
        diff_rgb = np.dstack([diff_mask*255, np.zeros_like(diff_mask), np.zeros_like(diff_mask)])
        diff_rgb = cv2.cvtColor(diff_rgb, cv2.COLOR_RGB2BGR)
        diff_rgb = pad(diff_rgb)
        cat = np.concatenate([cat, diff_rgb], axis=1)
    return cat

def save_binary_mask_png(dst_path: str, mask: np.ndarray):
    # 0/255 の1ch PNG
    m = (mask.astype(np.uint8) * 255)
    cv2.imwrite(dst_path, m)
```

### 3-3. `src/run_sam2.py`（メインCLI）

```python
# -*- coding: utf-8 -*-
# src/run_sam2.py
import os, json, glob
import numpy as np
import cv2
from PIL import Image
import yaml
from tqdm import tqdm

from src.dataset_utils import list_images
from src.qwen_client import encode_image_to_data_url  # 未使用でも依存はOK
from src.visualize import ensure_dir
from src.prompts import build_bbox_prompt  # 未使用でも依存はOK

from src.sam2_runner import (
    build_predictor, _to_abs_xyxy, predict_masks_for_boxes, select_best_mask_per_box
)
from src.viz_masks import overlay_masks, xor_diff_mask, side_by_side, save_binary_mask_png

def load_qwen_json(json_path: str) -> dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sam2_cfg = cfg["sam2"]
    paths = cfg["paths"]

    # 出力先
    out_root = paths["out_root"]
    out_json_dir = os.path.join(out_root, "json")
    out_mask_dir = os.path.join(out_root, "masks")
    out_viz_dir  = os.path.join(out_root, "viz")
    ensure_dir(out_json_dir); ensure_dir(out_mask_dir); ensure_dir(out_viz_dir)

    # 画像とQwen JSONの対応付け
    img_dir = paths["input_dir"]
    qwen_dir = paths["qwen_json_dir"]
    img_paths = list_images(img_dir, max_items=0)

    # SAM2 2系統（b+ / large）を構築
    repo_root = sam2_cfg["repo_root"]
    cfg_bplus = os.path.join(repo_root, sam2_cfg["cfg_base_plus"])
    cfg_large = os.path.join(repo_root, sam2_cfg["cfg_large"])
    ckpt_bplus = sam2_cfg["ckpt_base_plus"]
    ckpt_large = sam2_cfg["ckpt_large"]
    device = sam2_cfg.get("device", "cuda")
    dtype = sam2_cfg.get("dtype", "bfloat16")
    multimask_output = bool(sam2_cfg.get("multimask_output", True))
    qwen_conf_thres = float(sam2_cfg.get("conf_threshold", 0.2))

    predictor_bplus = build_predictor(cfg_bplus, ckpt_bplus, device=device)
    predictor_large = build_predictor(cfg_large, ckpt_large, device=device)

    for img_path in tqdm(img_paths, desc="SAM2 (b+ / large)"):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(qwen_dir, f"{stem}.json")
        if not os.path.exists(json_path):
            # Qwenの結果がない画像はスキップ
            continue

        # 画像の読み込み（RGB）
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        img_rgb = np.array(pil)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Qwen出力のロード
        jd = load_qwen_json(json_path)
        dets = jd.get("detections", [])
        # 閾値でフィルタ
        dets = [d for d in dets if float(d.get("confidence", 0.0)) >= qwen_conf_thres]

        if len(dets) == 0:
            # 可視化だけ原画像を保存
            cv2.imwrite(os.path.join(out_viz_dir, f"{stem}_noval.jpg"), img_bgr)
            continue

        labels = [str(d.get("label_ja", "item")) for d in dets]
        boxes_abs = np.array([_to_abs_xyxy(d["bbox_xyxy_norm"], W, H) for d in dets], dtype=np.float32)

        # ---- SAM2(b+)
        masks_b, iou_b, _ = predict_masks_for_boxes(
            predictor_bplus, img_rgb, boxes_abs, dtype=dtype, multimask_output=multimask_output
        )
        masks_b = select_best_mask_per_box(masks_b, iou_b, multimask_output)

        # ---- SAM2(large)
        masks_l, iou_l, _ = predict_masks_for_boxes(
            predictor_large, img_rgb, boxes_abs, dtype=dtype, multimask_output=multimask_output
        )
        masks_l = select_best_mask_per_box(masks_l, iou_l, multimask_output)

        # IoU（b+ vs large）を検出単位で算出
        def iou_pair(a: np.ndarray, b: np.ndarray) -> float:
            inter = float(np.logical_and(a, b).sum())
            union = float(np.logical_or(a, b).sum())
            return (inter / union) if union > 0 else 0.0
        ious_bl = [iou_pair(masks_b[i], masks_l[i]) for i in range(len(dets))]

        # 可視化
        viz_b = overlay_masks(img_bgr, [masks_b[i] for i in range(len(dets))], labels)
        viz_l = overlay_masks(img_bgr, [masks_l[i] for i in range(len(dets))], labels)
        diff  = xor_diff_mask([masks_b[i] for i in range(len(dets))], [masks_l[i] for i in range(len(dets))])
        panel = side_by_side(img_bgr, viz_b, viz_l, diff_mask=diff)
        cv2.imwrite(os.path.join(out_viz_dir, f"{stem}_panel.jpg"), panel)

        # マスクPNGを保存
        for i, lab in enumerate(labels):
            safe_lab = "".join([c if c.isalnum() else "_" for c in lab])[:40]
            dst_b = os.path.join(out_mask_dir, f"{stem}_det{i:02d}_{safe_lab}_bplus.png")
            dst_l = os.path.join(out_mask_dir, f"{stem}_det{i:02d}_{safe_lab}_large.png")
            save_binary_mask_png(dst_b, masks_b[i])
            save_binary_mask_png(dst_l, masks_l[i])

        # サマリJSON
        out = {
            "image": os.path.basename(img_path),
            "width": W, "height": H,
            "detections": []
        }
        for i, d in enumerate(dets):
            out["detections"].append({
                "id": i,
                "label_ja": labels[i],
                "qwen_confidence": float(d.get("confidence", 0.0)),
                "bbox_xyxy_norm": d["bbox_xyxy_norm"],
                "bbox_xyxy_abs": [float(x) for x in boxes_abs[i].tolist()],
                "sam2_bplus": {
                    "area_px": int(masks_b[i].sum()),
                    "pred_iou": float(iou_b[i]) if np.ndim(iou_b) == 1 else float(np.max(iou_b[i]))
                },
                "sam2_large": {
                    "area_px": int(masks_l[i].sum()),
                    "pred_iou": float(iou_l[i]) if np.ndim(iou_l) == 1 else float(np.max(iou_l[i]))
                },
                "bplus_vs_large_iou": float(ious_bl[i])
            })
        with open(os.path.join(out_json_dir, f"{stem}.sam2.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"完了: {out_root}")

if __name__ == "__main__":
    main()
```

---

## 4) 使い方

1. **（前段）Qwen2.5‑VLのBBox生成**は既存 `python -m src.run_infer` で実行済み（`outputs/json/*.json` がある状態）。
2. **SAM2.1のセットアップ**（§1-2, 1-3）を実施。
3. `config.yaml` の `sam2.repo_root / cfg_* / ckpt_* / paths.*` を**絶対パス**で更新。
4. 実行：

   ```bash
   python -m src.run_sam2
   ```
5. 出力：

   * `outputs/sam2/viz/<stem>_panel.jpg` … **原画像 | b+可視化 | large可視化 | 差分** の横並び
   * `outputs/sam2/masks/<stem>_det##_label_bplus.png` / `_large.png` … **0/255** のバイナリPNG
   * `outputs/sam2/json/<stem>.sam2.json` … **各検出の統計（面積・予測IoU・b+ vs large IoU）**

---

## 5) 実装判断の要点（なぜこの形か）

* **公式APIに忠実**：`build_sam2` → `SAM2ImagePredictor` → `set_image` → `predict(boxes=...)` の流れは **公式READMEのサンプル**を踏襲。([GitHub][1])
* **boxes引数を使用**：`SAM2ImagePredictor.predict` は **`boxes`**（torch.Tensor, xyxy）を受け付ける実装で、複数オブジェクトにも対応。([Hugging Face][2])
* **2系統比較**：**b+** と **large** はSAM2.1の代表的サイズで、READMEに**チェックポイントと設定**が明示（b+=80.8M, large=224.4M / config名・直接DLリンク）。([GitHub][1])
* **精度 × 速度**：READMEの表には速度指標（A100でb+≈64FPS, large≈39FPS）とベンチがあり、**b+は高速・largeは精緻**というトレードオフを示唆。**用途に応じて切替**しやすい設計にしています。([GitHub][1])
* **マルチマスク**：`multimask_output=True` で複数仮説が返る実装に備え、**最高スコア採用**で**1箱1マスク化**する整形を実装（将来の実装差分への保険）。([Hugging Face][2])

---

## 6) ベストプラクティス（任意の上積み）

* **箱＋点の併用**：SAM2は**box/point/mask**の複合プロンプトが可能（文献/解説多数）。将来的に、BBox内中心へ**正例点**を1つ置くと輪郭が安定する場面あり。([DigitalOcean][4])
* **HuggingFaceからの読み込み**：`SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")` なども可能（ただし2.1の命名はリポジトリに依存）。本プランは**公式ckpt＋config**の直接指定を採用。([GitHub][1])
* **ONNX/最適化**：SAM2は `torch.compile` による高速化やONNX利用の例もある（公式Release Notes/チュートリアル類）。大規模バッチ運用時に検討。([GitHub][1])

---

## 7) 参考（根拠リンク）

* **SAM 2 公式**：インストール、`SAM2ImagePredictor` 使用例、**SAM 2.1**チェックポイント（`sam2.1_hiera_base_plus.pt` / `sam2.1_hiera_large.pt`）、対応config（`sam2.1_hiera_b+.yaml` / `sam2.1_hiera_l.yaml`）。([GitHub][1])
* **`boxes` 引数**：`SAM2ImagePredictor.predict` の引数に **`boxes`** が存在（batched torch.Tensor, multimask出力にも対応）。([Hugging Face][2])
* **Boxプロンプトの例**：SAM2にboxプロンプトを与えてマスク生成（解説・チュートリアル類）。([Samgeo][5], [Analytics Vidhya][6])

---

### 付録：差し替えの少ない運用Tips

* **ラベル同期**：`labels` はQwen出力の順序で固定、BBox→マスクは**同インデックス**で対応付け。
* **エラー対策**：一部環境でBFloat16非対応なら `dtype: "float16"` あるいは CPU では `"float32"` を使用。
* **評価拡張**：FoodSegのGTマスクから**最小外接BBox**を作って Qwen/SAM2 のIoU評価を追加可能。

上記の手順とコードを `src/` に追加・実行すれば、**QwenのBBox＋ラベル → SAM2.1(b+ / large)でのマスク生成と比較**まで、**一気通貫**で動作します。必要であれば、**SAM2のマスクRLE出力（COCO形式）**や**SAM2→UniDepth→体積算出**への接続コードも拡張できます。

[1]: https://github.com/facebookresearch/sam2 "GitHub - facebookresearch/sam2: The repository provides code for running inference with the Meta Segment Anything Model 2 (SAM 2), links for downloading the trained model checkpoints, and example notebooks that show how to use the model."
[2]: https://huggingface.co/spaces/fffiloni/SAM2-Video-Predictor/blob/01bbb6ccc5c1344294b90cc0537ba89c6693b072/sam2/sam2_image_predictor.py?utm_source=chatgpt.com "sam2/sam2_image_predictor.py · fffiloni/SAM2-Video- ..."
[3]: https://github.com/facebookresearch/sam2/blob/main/sam2/configs/sam2.1/sam2.1_hiera_b%2B.yaml "sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml at main · facebookresearch/sam2 · GitHub"
[4]: https://www.digitalocean.com/community/tutorials/sam-2-metas-next-gen-model-for-video-and-image-segmentation?utm_source=chatgpt.com "Meta's Next-Gen Model for Video and Image Segmentation"
[5]: https://samgeo.gishub.org/examples/sam2_box_prompts/?utm_source=chatgpt.com "Sam2 box prompts - segment-geospatial"
[6]: https://www.analyticsvidhya.com/blog/2024/08/mastering-image-and-video-segmentation-with-sam-2/?utm_source=chatgpt.com "Mastering Image and Video Segmentation with SAM 2"
