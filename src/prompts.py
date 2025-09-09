# -*- coding: utf-8 -*-
# src/prompts.py
from textwrap import dedent

def build_bbox_prompt(image_w: int, image_h: int, language: str = "en") -> str:
    """
    Qwen2.5-VL に対し、料理・食材の検出＋BBox＋ラベルを厳格JSONで出させるプロンプト。
    - 出力は JSON のみ（前後の説明文・コードフェンス禁止）。
    - BBox は正規化 xyxy (x_min, y_min, x_max, y_max) in [0,1]。小数6桁以内。
    - confidence は [0,1]。
    """
    en_instr = f"""
    You are a food image object detection assistant.
    The input image size is width={image_w}px, height={image_h}px.
    Please follow these requirements:

    1) Output **strict JSON only** with English labels. No explanatory text or code fences.
    2) Detect only "food or ingredients", excluding dishes, cutlery, or shadows.
    3) BBox should be **normalized xyxy** coordinates (x_min, y_min, x_max, y_max) in [0,1] range.
       Clip values below 0 or above 1 by rounding. Maximum 6 decimal places.
    4) Each element should include {{"label_en": str, "bbox_xyxy_norm": [x1,y1,x2,y2], "confidence": float}}.
    5) Use accurate English names for foods and ingredients.
    6) The top level JSON should be {{ "detections": [ ... ] }}.

    Output example:
    {{
      "detections": [
        {{"label_en": "rice", "bbox_xyxy_norm": [0.12, 0.40, 0.58, 0.78], "confidence": 0.87}},
        {{"label_en": "curry sauce", "bbox_xyxy_norm": [0.20, 0.45, 0.70, 0.82], "confidence": 0.81}}
      ]
    }}
    """
    return dedent(en_instr).strip()