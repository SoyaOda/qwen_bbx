# -*- coding: utf-8 -*-
# src/qwen_client.py
import os
import base64
import mimetypes
import json
from typing import Dict, Any, List
from openai import OpenAI, APIError, APITimeoutError

def encode_image_to_data_url(path: str) -> str:
    """ローカル画像を base64 Data URL (image/*) に変換。"""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        # 既定はjpeg
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    # 公式ガイドに従い data URL を構築
    return f"data:{mime};base64,{b64}"

def extract_text_content(chat_completion) -> str:
    """
    DashScopeのOpenAI互換APIはmessage.contentが文字列または配列の可能性がある。
    両対応で文字列テキストを抽出。
    """
    # v1: choices[0].message.content (str)
    content = None
    try:
        content = chat_completion.choices[0].message.content
        if isinstance(content, list):
            # [{"type":"text","text":"..."}] 形式を想定
            texts = [c.get("text") for c in content if isinstance(c, dict) and c.get("type") == "text"]
            content = "\n".join([t for t in texts if t])
    except Exception:
        pass
    if not content:
        # モデルダンプ経由の保険
        as_dict = json.loads(chat_completion.model_dump_json())
        msg = as_dict["choices"][0]["message"]["content"]
        if isinstance(msg, list):
            texts = [x.get("text") for x in msg if isinstance(x, dict)]
            content = "\n".join([t for t in texts if t])
        else:
            content = msg
    return content

def call_qwen_bbox(
    api_key: str,
    base_url: str,
    model: str,
    image_data_urls: List[str],
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    top_p: float = 0.1,
    timeout_s: int = 120,
    max_retries: int = 2
) -> Dict[str, Any]:
    """Qwen2.5‑VLに画像＋指示を送り、JSON文字列をdictにパースして返す。"""
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            *[
                {"type": "image_url", "image_url": {"url": data_url}}
                for data_url in image_data_urls
            ],
            {"type": "text", "text": user_prompt}
        ]}
    ]
    last_err = None
    for _ in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
            )
            text = extract_text_content(resp).strip()
            # モデルが ```json フェンスを出す場合を除去
            if text.startswith("```"):
                text = text.strip("`")
                # 先頭に "json" が付いている可能性
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            # JSONへ
            return json.loads(text)
        except (APIError, APITimeoutError, json.JSONDecodeError) as e:
            last_err = e
            continue
    raise RuntimeError(f"Qwen呼び出しに失敗: {last_err}")