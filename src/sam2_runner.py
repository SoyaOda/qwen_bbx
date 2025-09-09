# -*- coding: utf-8 -*-
# src/sam2_runner.py
import os
import sys
import numpy as np
import torch
from typing import List, Dict, Any, Tuple

# SAM2リポジトリをパスに追加
SAM2_REPO = "/home/soya/sam2_1_food_finetuning/external/sam2"
if SAM2_REPO not in sys.path:
    sys.path.append(SAM2_REPO)

def _to_abs_xyxy(box_norm: List[float], w: int, h: int) -> List[float]:
    """正規化座標をピクセル座標に変換"""
    x1 = float(np.clip(box_norm[0], 0.0, 1.0) * w)
    y1 = float(np.clip(box_norm[1], 0.0, 1.0) * h)
    x2 = float(np.clip(box_norm[2], 0.0, 1.0) * w)
    y2 = float(np.clip(box_norm[3], 0.0, 1.0) * h)
    # 座標整合
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    return [x1, y1, x2, y2]

def build_predictor(model_cfg_path: str, ckpt_path: str, device: str = "cuda"):
    """
    SAM2ImagePredictor を構築。b+ / large のどちらにも使える。
    """
    # 作業ディレクトリをSAM2リポジトリに変更（相対パス解決のため）
    original_cwd = os.getcwd()
    os.chdir(SAM2_REPO)
    
    try:
        # SAM2のインポート（cdした後に行う）
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        
        # config名を相対パスに変換
        if "b+" in model_cfg_path or "base_plus" in model_cfg_path:
            config_path = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif "large" in model_cfg_path:
            config_path = "configs/sam2.1/sam2.1_hiera_l.yaml"
        else:
            # フルパスから相対パスを抽出
            config_path = model_cfg_path.replace(SAM2_REPO + "/", "")
        
        # チェックポイントも相対パスに変換
        if os.path.isabs(ckpt_path):
            ckpt_path_rel = ckpt_path.replace(SAM2_REPO + "/", "")
            if not os.path.exists(ckpt_path_rel):
                ckpt_path_rel = ckpt_path
        else:
            ckpt_path_rel = ckpt_path
        
        # build_sam2を呼び出し
        sam_model = build_sam2(config_path, ckpt_path_rel)
        
        # Predictorを作成
        predictor = SAM2ImagePredictor(sam_model)
        
        # デバイスの設定
        if device == "cuda" and torch.cuda.is_available():
            predictor.model.to("cuda")
        else:
            predictor.model.to("cpu")
            
        return predictor
        
    finally:
        # 元のディレクトリに戻る
        os.chdir(original_cwd)

def predict_masks_for_boxes(
    predictor,
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
    を返す。
    """
    assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3
    H, W = image_rgb.shape[:2]
    device = "cuda" if next(predictor.model.parameters()).is_cuda else "cpu"

    # 作業ディレクトリを変更
    original_cwd = os.getcwd()
    os.chdir(SAM2_REPO)
    
    try:
        boxes_t = torch.as_tensor(boxes_abs_xyxy, dtype=torch.float32, device=device)
        
        use_autocast = (device == "cuda" and dtype in ("bfloat16", "float16"))
        
        if use_autocast:
            ctx = torch.autocast(device_type="cuda", dtype=getattr(torch, dtype))
        else:
            ctx = torch.no_grad()
        
        with torch.inference_mode():
            with ctx:
                predictor.set_image(image_rgb)
                
                # 複数ボックスを処理
                all_masks = []
                all_ious = []
                all_lowres = []
                
                for i in range(boxes_t.shape[0]):
                    box = boxes_t[i:i+1]  # (1, 4)
                    masks, ious, lowres = predictor.predict(
                        box=box, 
                        multimask_output=multimask_output, 
                        return_logits=False
                    )
                    
                    # multimask_outputの場合、最良のマスクを選択
                    if multimask_output and masks.shape[0] > 1:
                        best_idx = np.argmax(ious)
                        masks = masks[best_idx:best_idx+1]
                        ious = ious[best_idx:best_idx+1]
                        if lowres is not None:
                            lowres = lowres[best_idx:best_idx+1]
                    
                    all_masks.append(masks[0])
                    all_ious.append(ious[0] if isinstance(ious, np.ndarray) else ious.item())
                    all_lowres.append(lowres[0] if lowres is not None else None)
        
        # 結果を整形
        masks = np.stack(all_masks, axis=0)
        ious = np.array(all_ious)
        lowres = np.stack(all_lowres, axis=0) if all_lowres[0] is not None else None
        
        # bool型に変換
        masks = (masks > 0.5).astype(bool)
        
        return masks, ious, lowres
        
    finally:
        # 元のディレクトリに戻る
        os.chdir(original_cwd)

def select_best_mask_per_box(
    masks: np.ndarray, ious: np.ndarray, multimask_output: bool
) -> np.ndarray:
    """
    multimask_output=True の場合は各BBoxにつき複数仮説が返る実装もあるため、
    ここでは **仮説1枚/箱** を保証する。
    """
    # 既に (N,H,W) ならそのまま
    if masks.ndim == 3:
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