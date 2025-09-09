# -*- coding: utf-8 -*-
# src/viz_masks.py
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from .visualize import norm_xyxy_to_abs, ensure_dir  # 既存関数を再利用

def color_map(n: int) -> np.ndarray:
    """ランダムカラーマップ生成"""
    rs = np.random.RandomState(123)
    return rs.randint(0, 255, size=(n, 3), dtype=np.uint8)

def overlay_masks(img_bgr: np.ndarray, masks: List[np.ndarray], labels: List[str]) -> np.ndarray:
    """マスクを画像に重ねて可視化"""
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
            cv2.rectangle(out, (x1, ytxt - th - 6), (x1 + tw + 4, ytxt), 
                         color=tuple(int(c) for c in color), thickness=-1)
            cv2.putText(out, txt, (x1 + 2, ytxt - 4), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (0,0,0), 2, cv2.LINE_AA)
    return out

def xor_diff_mask(masks_a: List[np.ndarray], masks_b: List[np.ndarray]) -> np.ndarray:
    """マスクの差分（XOR）を計算"""
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
    """複数画像を横並びに結合"""
    h = max(img_bgr.shape[0], viz_a.shape[0], viz_b.shape[0])
    
    def pad(img):
        pad_h = h - img.shape[0]
        if pad_h > 0:
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
        return img
    
    img_bgr = pad(img_bgr)
    viz_a = pad(viz_a)
    viz_b = pad(viz_b)
    cat = np.concatenate([img_bgr, viz_a, viz_b], axis=1)
    
    if diff_mask is not None:
        diff_rgb = np.dstack([diff_mask*255, np.zeros_like(diff_mask), np.zeros_like(diff_mask)])
        diff_rgb = cv2.cvtColor(diff_rgb, cv2.COLOR_RGB2BGR)
        diff_rgb = pad(diff_rgb)
        cat = np.concatenate([cat, diff_rgb], axis=1)
    return cat

def save_binary_mask_png(dst_path: str, mask: np.ndarray):
    """バイナリマスクをPNGとして保存（0/255）"""
    m = (mask.astype(np.uint8) * 255)
    cv2.imwrite(dst_path, m)