#!/usr/bin/env python
# -*- coding: utf-8 -*-
# src/run_sam2.py
"""
SAM2.1（base_plus / large）を用いて、Qwen2.5-VLの検出結果からマスクを生成し、
比較・可視化するメインスクリプト
"""

import os
import sys
import json
import glob
import numpy as np
import cv2
from PIL import Image
import yaml
from tqdm import tqdm

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_utils import list_images
from src.visualize import ensure_dir
from src.sam2_runner import (
    build_predictor, _to_abs_xyxy, predict_masks_for_boxes, select_best_mask_per_box
)
from src.viz_masks import (
    overlay_masks, xor_diff_mask, side_by_side, save_binary_mask_png
)

def load_qwen_json(json_path: str) -> dict:
    """Qwen出力のJSONをロード"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    # 設定ファイルの読み込み
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    sam2_cfg = cfg["sam2"]
    paths = cfg["paths"]

    # 出力先ディレクトリの作成
    out_root = paths["out_root"]
    out_json_dir = os.path.join(out_root, "json")
    out_mask_dir = os.path.join(out_root, "masks")
    out_viz_dir = os.path.join(out_root, "viz")
    ensure_dir(out_json_dir)
    ensure_dir(out_mask_dir)
    ensure_dir(out_viz_dir)

    # 画像とQwen JSONの対応付け
    img_dir = paths["input_dir"]
    qwen_dir = paths["qwen_json_dir"]
    img_paths = list_images(img_dir, max_items=0)

    if not img_paths:
        print(f"警告: {img_dir} に画像が見つかりません")
        return

    # SAM2 2系統（b+ / large）を構築
    repo_root = sam2_cfg["repo_root"]
    # build_sam2にはconfig名だけを渡す（フルパスではなく）
    cfg_bplus = sam2_cfg["cfg_base_plus"]
    cfg_large = sam2_cfg["cfg_large"]
    full_cfg_bplus = os.path.join(repo_root, cfg_bplus)
    full_cfg_large = os.path.join(repo_root, cfg_large)
    ckpt_bplus = sam2_cfg["ckpt_base_plus"]
    ckpt_large = sam2_cfg["ckpt_large"]
    device = sam2_cfg.get("device", "cuda")
    dtype = sam2_cfg.get("dtype", "bfloat16")
    multimask_output = bool(sam2_cfg.get("multimask_output", True))
    qwen_conf_thres = float(sam2_cfg.get("conf_threshold", 0.2))

    print(f"SAM2モデルを初期化中...")
    print(f"  - Base+ config: {full_cfg_bplus}")
    print(f"  - Large config: {full_cfg_large}")
    print(f"  - Device: {device}")
    print(f"  - Dtype: {dtype}")
    
    # Predictorの構築
    predictor_bplus = build_predictor(cfg_bplus, ckpt_bplus, device=device)
    predictor_large = build_predictor(cfg_large, ckpt_large, device=device)
    print("SAM2モデルの初期化完了")

    # 処理のメインループ
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
            # 検出がない場合、原画像だけ保存
            cv2.imwrite(os.path.join(out_viz_dir, f"{stem}_noval.jpg"), img_bgr)
            continue

        # ラベルの抽出（英語版を優先、なければ日本語版）
        labels = []
        for d in dets:
            if "label_en" in d:
                labels.append(str(d.get("label_en", "item")))
            else:
                labels.append(str(d.get("label_ja", "item")))
        
        boxes_abs = np.array([_to_abs_xyxy(d["bbox_xyxy_norm"], W, H) for d in dets], dtype=np.float32)

        # ---- SAM2(b+) 実行
        masks_b, iou_b, _ = predict_masks_for_boxes(
            predictor_bplus, img_rgb, boxes_abs, dtype=dtype, multimask_output=multimask_output
        )
        masks_b = select_best_mask_per_box(masks_b, iou_b, multimask_output)

        # ---- SAM2(large) 実行
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
        diff = xor_diff_mask([masks_b[i] for i in range(len(dets))], 
                            [masks_l[i] for i in range(len(dets))])
        panel = side_by_side(img_bgr, viz_b, viz_l, diff_mask=diff)
        cv2.imwrite(os.path.join(out_viz_dir, f"{stem}_panel.jpg"), panel)

        # 個別の可視化画像も保存
        cv2.imwrite(os.path.join(out_viz_dir, f"{stem}_bplus.jpg"), viz_b)
        cv2.imwrite(os.path.join(out_viz_dir, f"{stem}_large.jpg"), viz_l)

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
            "width": W,
            "height": H,
            "detections": []
        }
        
        for i, d in enumerate(dets):
            detection_data = {
                "id": i,
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
            }
            
            # ラベルの追加（英語版を優先）
            if "label_en" in d:
                detection_data["label_en"] = d["label_en"]
            if "label_ja" in d:
                detection_data["label_ja"] = d["label_ja"]
            
            out["detections"].append(detection_data)
        
        with open(os.path.join(out_json_dir, f"{stem}.sam2.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"完了: {out_root}")
    print(f"  - JSON: {out_json_dir}")
    print(f"  - マスクPNG: {out_mask_dir}")
    print(f"  - 可視化: {out_viz_dir}")

if __name__ == "__main__":
    main()