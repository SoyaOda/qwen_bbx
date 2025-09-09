#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM2.1（base_plus / large）を用いて、Qwen2.5-VLの検出結果からマスクを生成
改良版：作業ディレクトリを変更してSAM2を実行
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image
import yaml
from tqdm import tqdm
import torch

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_utils import list_images
from src.visualize import ensure_dir

def load_qwen_json(json_path: str) -> dict:
    """Qwen出力のJSONをロード"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _to_abs_xyxy(box_norm, w: int, h: int):
    """正規化座標をピクセル座標に変換"""
    x1 = float(np.clip(box_norm[0], 0.0, 1.0) * w)
    y1 = float(np.clip(box_norm[1], 0.0, 1.0) * h)
    x2 = float(np.clip(box_norm[2], 0.0, 1.0) * w)
    y2 = float(np.clip(box_norm[3], 0.0, 1.0) * h)
    # 座標整合
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    return [x1, y1, x2, y2]

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

    # SAM2の設定
    device = sam2_cfg.get("device", "cuda")
    dtype = sam2_cfg.get("dtype", "bfloat16")
    multimask_output = bool(sam2_cfg.get("multimask_output", True))
    qwen_conf_thres = float(sam2_cfg.get("conf_threshold", 0.2))

    print(f"SAM2モデルを初期化中...")
    print(f"  - Device: {device}")
    print(f"  - Dtype: {dtype}")
    
    # 元のディレクトリを保存
    original_cwd = os.getcwd()
    
    # SAM2リポジトリに移動
    SAM2_REPO = sam2_cfg["repo_root"]
    os.chdir(SAM2_REPO)
    sys.path.append(SAM2_REPO)
    
    try:
        # SAM2のインポート
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # Predictorの構築（相対パスで指定）
        print("Loading base_plus model...")
        predictor_bplus = build_sam2(
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "./checkpoints/sam2.1_hiera_base_plus.pt"
        )
        predictor_bplus = SAM2ImagePredictor(predictor_bplus)
        if device == "cuda" and torch.cuda.is_available():
            predictor_bplus.model.to("cuda")
        
        print("Loading large model...")
        predictor_large = build_sam2(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            "./checkpoints/sam2.1_hiera_large.pt"
        )
        predictor_large = SAM2ImagePredictor(predictor_large)
        if device == "cuda" and torch.cuda.is_available():
            predictor_large.model.to("cuda")
        
        print("SAM2モデルの初期化完了")
        
        # 元のディレクトリに戻る（画像読み込みのため）
        os.chdir(original_cwd)
        
        # 処理のメインループ
        for img_path in tqdm(img_paths[:3], desc="SAM2 (b+ / large)"):  # まず3枚でテスト
            stem = os.path.splitext(os.path.basename(img_path))[0]
            json_path = os.path.join(qwen_dir, f"{stem}.json")
            
            if not os.path.exists(json_path):
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
                cv2.imwrite(os.path.join(out_viz_dir, f"{stem}_noval.jpg"), img_bgr)
                continue

            # ラベルの抽出
            labels = []
            for d in dets:
                if "label_en" in d:
                    labels.append(str(d.get("label_en", "item")))
                else:
                    labels.append(str(d.get("label_ja", "item")))
            
            boxes_abs = np.array([_to_abs_xyxy(d["bbox_xyxy_norm"], W, H) for d in dets], dtype=np.float32)

            # SAM2ディレクトリに戻る（推論のため）
            os.chdir(SAM2_REPO)
            
            # ---- SAM2(b+) 実行
            print(f"  Processing {stem} with base_plus...")
            masks_b = []
            iou_b = []
            
            with torch.inference_mode():
                if device == "cuda" and dtype == "bfloat16":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predictor_bplus.set_image(img_rgb)
                        for box in boxes_abs:
                            mask, score, _ = predictor_bplus.predict(
                                box=box,
                                multimask_output=multimask_output
                            )
                            if multimask_output and mask.shape[0] > 1:
                                best_idx = np.argmax(score)
                                mask = mask[best_idx:best_idx+1]
                                score = score[best_idx:best_idx+1]
                            masks_b.append(mask[0])
                            iou_b.append(score[0])
                else:
                    predictor_bplus.set_image(img_rgb)
                    for box in boxes_abs:
                        mask, score, _ = predictor_bplus.predict(
                            box=box,
                            multimask_output=multimask_output
                        )
                        if multimask_output and mask.shape[0] > 1:
                            best_idx = np.argmax(score)
                            mask = mask[best_idx:best_idx+1]
                            score = score[best_idx:best_idx+1]
                        masks_b.append(mask[0])
                        iou_b.append(score[0])

            # ---- SAM2(large) 実行
            print(f"  Processing {stem} with large...")
            masks_l = []
            iou_l = []
            
            with torch.inference_mode():
                if device == "cuda" and dtype == "bfloat16":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        predictor_large.set_image(img_rgb)
                        for box in boxes_abs:
                            mask, score, _ = predictor_large.predict(
                                box=box,
                                multimask_output=multimask_output
                            )
                            if multimask_output and mask.shape[0] > 1:
                                best_idx = np.argmax(score)
                                mask = mask[best_idx:best_idx+1]
                                score = score[best_idx:best_idx+1]
                            masks_l.append(mask[0])
                            iou_l.append(score[0])
                else:
                    predictor_large.set_image(img_rgb)
                    for box in boxes_abs:
                        mask, score, _ = predictor_large.predict(
                            box=box,
                            multimask_output=multimask_output
                        )
                        if multimask_output and mask.shape[0] > 1:
                            best_idx = np.argmax(score)
                            mask = mask[best_idx:best_idx+1]
                            score = score[best_idx:best_idx+1]
                        masks_l.append(mask[0])
                        iou_l.append(score[0])
            
            # 元のディレクトリに戻る
            os.chdir(original_cwd)
            
            # マスクを配列に変換
            masks_b = np.array(masks_b)
            masks_l = np.array(masks_l)
            iou_b = np.array(iou_b)
            iou_l = np.array(iou_l)

            # IoU（b+ vs large）を計算
            def iou_pair(a: np.ndarray, b: np.ndarray) -> float:
                inter = float(np.logical_and(a, b).sum())
                union = float(np.logical_or(a, b).sum())
                return (inter / union) if union > 0 else 0.0
            
            ious_bl = [iou_pair(masks_b[i], masks_l[i]) for i in range(len(dets))]

            # 可視化（シンプル版）
            viz_b = img_bgr.copy()
            viz_l = img_bgr.copy()
            
            for i, (mask_b, mask_l, label) in enumerate(zip(masks_b, masks_l, labels)):
                # Base+の可視化
                color_b = (0, 255, 0)  # 緑
                mask_rgb_b = np.zeros_like(viz_b)
                mask_b_bool = mask_b.astype(bool)
                mask_rgb_b[mask_b_bool] = color_b
                viz_b = cv2.addWeighted(viz_b, 0.7, mask_rgb_b, 0.3, 0)
                
                # Largeの可視化
                color_l = (0, 0, 255)  # 赤
                mask_rgb_l = np.zeros_like(viz_l)
                mask_l_bool = mask_l.astype(bool)
                mask_rgb_l[mask_l_bool] = color_l
                viz_l = cv2.addWeighted(viz_l, 0.7, mask_rgb_l, 0.3, 0)
                
                # ラベルを追加
                ys, xs = np.where(mask_b_bool)
                if xs.size > 0 and ys.size > 0:
                    x1, y1 = int(xs.min()), int(ys.min())
                    cv2.putText(viz_b, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_b, 2)
                
                ys, xs = np.where(mask_l_bool)
                if xs.size > 0 and ys.size > 0:
                    x1, y1 = int(xs.min()), int(ys.min())
                    cv2.putText(viz_l, label, (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_l, 2)
            
            # 画像を保存
            cv2.imwrite(os.path.join(out_viz_dir, f"{stem}_bplus.jpg"), viz_b)
            cv2.imwrite(os.path.join(out_viz_dir, f"{stem}_large.jpg"), viz_l)
            
            # XOR差分マスクの計算と可視化
            diff_mask = np.zeros((H, W), dtype=np.uint8)
            for mask_b, mask_l in zip(masks_b, masks_l):
                diff_mask ^= ((mask_b.astype(np.uint8) ^ mask_l.astype(np.uint8)) > 0).astype(np.uint8)
            
            # 差分マスクの可視化（赤色で表示）
            diff_viz = img_bgr.copy()
            if diff_mask.sum() > 0:
                diff_rgb = np.zeros_like(diff_viz)
                diff_rgb[diff_mask > 0] = (0, 0, 255)  # 赤色
                diff_viz = cv2.addWeighted(diff_viz, 0.7, diff_rgb, 0.3, 0)
            
            # 横並びパネルも作成（原画像 | b+ | large | 差分）
            panel = np.concatenate([img_bgr, viz_b, viz_l, diff_viz], axis=1)
            cv2.imwrite(os.path.join(out_viz_dir, f"{stem}_panel.jpg"), panel)
            
            # 個別マスクPNGの保存
            for i, (mask_b, mask_l, label) in enumerate(zip(masks_b, masks_l, labels)):
                safe_label = "".join([c if c.isalnum() else "_" for c in label])[:40]
                
                # バイナリマスクを0/255のPNGとして保存
                mask_b_png = (mask_b.astype(np.uint8) * 255)
                mask_l_png = (mask_l.astype(np.uint8) * 255)
                
                cv2.imwrite(os.path.join(out_mask_dir, f"{stem}_det{i:02d}_{safe_label}_bplus.png"), mask_b_png)
                cv2.imwrite(os.path.join(out_mask_dir, f"{stem}_det{i:02d}_{safe_label}_large.png"), mask_l_png)

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
                        "pred_iou": float(iou_b[i])
                    },
                    "sam2_large": {
                        "area_px": int(masks_l[i].sum()),
                        "pred_iou": float(iou_l[i])
                    },
                    "bplus_vs_large_iou": float(ious_bl[i])
                }
                
                if "label_en" in d:
                    detection_data["label_en"] = d["label_en"]
                if "label_ja" in d:
                    detection_data["label_ja"] = d["label_ja"]
                
                out["detections"].append(detection_data)
            
            with open(os.path.join(out_json_dir, f"{stem}.sam2.json"), "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            
            print(f"  Saved results for {stem}")

    finally:
        # 必ず元のディレクトリに戻る
        os.chdir(original_cwd)
    
    print(f"\n完了: {out_root}")
    print(f"  - JSON: {out_json_dir}")
    print(f"  - 可視化: {out_viz_dir}")

if __name__ == "__main__":
    main()