#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAM2の初期化テストスクリプト
公式のnotebook例を参考に実装
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# SAM2リポジトリをパスに追加
SAM2_REPO = "/home/soya/sam2_1_food_finetuning/external/sam2"
sys.path.append(SAM2_REPO)

# 作業ディレクトリを変更
os.chdir(SAM2_REPO)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def test_sam2():
    # チェックポイントとコンフィグのパス
    checkpoint = "./checkpoints/sam2.1_hiera_base_plus.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    
    print(f"Loading SAM2 model...")
    print(f"Config: {model_cfg}")
    print(f"Checkpoint: {checkpoint}")
    
    # モデルのビルド（相対パスで指定）
    predictor = build_sam2(model_cfg, checkpoint)
    
    # デバイスの設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = SAM2ImagePredictor(predictor)
    
    print(f"Model loaded successfully on {device}")
    
    # テスト画像を作成
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # 画像をセット
    predictor.set_image(test_image)
    
    # テストボックス
    test_box = np.array([100, 100, 200, 200])
    
    # 予測実行
    masks, scores, logits = predictor.predict(
        box=test_box,
        multimask_output=False
    )
    
    print(f"Prediction successful!")
    print(f"Mask shape: {masks.shape}")
    print(f"Score: {scores}")
    
    return True

if __name__ == "__main__":
    test_sam2()