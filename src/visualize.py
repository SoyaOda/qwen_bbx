# -*- coding: utf-8 -*-
# src/visualize.py
import os
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

def norm_xyxy_to_abs(b: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = int(round(max(0.0, min(1.0, b[0])) * w))
    y1 = int(round(max(0.0, min(1.0, b[1])) * h))
    x2 = int(round(max(0.0, min(1.0, b[2])) * w))
    y2 = int(round(max(0.0, min(1.0, b[3])) * h))
    # 座標整合
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return x1, y1, x2, y2

def draw_detections(
    img_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    conf_thres: float = 0.2
) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]
    for det in detections:
        try:
            label = det.get("label_en", det.get("label_ja", "item"))
            conf = float(det.get("confidence", 0.0))
            if conf < conf_thres:
                continue
            box = det["bbox_xyxy_norm"]
            x1, y1, x2, y2 = norm_xyxy_to_abs(box, w, h)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{label} {conf:.2f}"
            # テキスト背景
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(out, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        except Exception:
            continue
    return out

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)