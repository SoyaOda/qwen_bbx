# -*- coding: utf-8 -*-
# src/dataset_utils.py
import os
from typing import List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def list_images(root_dir: str, max_items: int = 0) -> List[str]:
    """ディレクトリ配下の画像パス一覧を取得。max_items>0なら先頭N件に制限。"""
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in sorted(filenames):
            ext = os.path.splitext(fn)[1].lower()
            if ext in IMG_EXTS:
                paths.append(os.path.join(dirpath, fn))
    if max_items and len(paths) > max_items:
        paths = paths[:max_items]
    return paths