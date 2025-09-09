# -*- coding: utf-8 -*-
# src/run_infer.py
import os
import cv2
import json
import yaml
from tqdm import tqdm
from PIL import Image
from prompts import build_bbox_prompt
from qwen_client import call_qwen_bbox, encode_image_to_data_url
from dataset_utils import list_images
from visualize import draw_detections, ensure_dir

def main():
    # 設定読込
    # Get config path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(os.path.dirname(script_dir), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_url    = cfg["provider"]["base_url"]
    model       = cfg["provider"]["model"]
    api_key_env = cfg["provider"]["api_key_env"]
    timeout_s   = int(cfg["provider"]["request_timeout_s"])
    max_retries = int(cfg["provider"]["max_retries"])
    temperature = float(cfg["provider"]["temperature"])
    top_p       = float(cfg["provider"]["top_p"])

    input_dir   = cfg["dataset"]["input_dir"]
    out_json    = cfg["dataset"]["out_json_dir"]
    out_viz     = cfg["dataset"]["out_viz_dir"]
    max_items   = int(cfg["inference"]["max_items"])
    conf_thres  = float(cfg["inference"]["conf_threshold"])

    ensure_dir(out_json); ensure_dir(out_viz)

    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"環境変数 {api_key_env} が未設定です。")

    img_paths = list_images(input_dir, max_items=max_items)
    if not img_paths:
        raise RuntimeError(f"画像が見つかりません: {input_dir}")

    for path in tqdm(img_paths, desc="Qwen2.5-VL 推論"):
        # 画像読み込み
        pil = Image.open(path).convert("RGB")
        w, h = pil.size

        # プロンプト
        system_prompt = "出力は厳格なJSONのみ。説明文・コードフェンスを一切含めないこと。"
        user_prompt   = build_bbox_prompt(w, h)

        # 画像→Data URL
        data_url = encode_image_to_data_url(path)

        # Qwen呼び出し
        result = call_qwen_bbox(
            api_key=api_key,
            base_url=base_url,
            model=model,
            image_data_urls=[data_url],  # 1画像（複数枚でも可）
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            top_p=top_p,
            timeout_s=timeout_s,
            max_retries=max_retries
        )

        # JSON保存
        stem = os.path.splitext(os.path.basename(path))[0]
        json_path = os.path.join(out_json, f"{stem}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        # 可視化
        import numpy as np
        img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        detections = result.get("detections", [])
        viz = draw_detections(img_bgr, detections, conf_thres=conf_thres)
        viz_path = os.path.join(out_viz, f"{stem}.jpg")
        cv2.imwrite(viz_path, viz)

    print(f"完了: JSON→{out_json} / 可視化→{out_viz}")

if __name__ == "__main__":
    main()