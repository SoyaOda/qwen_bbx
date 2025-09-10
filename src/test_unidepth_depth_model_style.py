# -*- coding: utf-8 -*-
"""
UniDepth v2 テストスクリプト（depth_modelプロジェクトスタイル）
depth_modelプロジェクトと完全に同じビジュアライゼーション方法を使用
"""
import os
import numpy as np
import torch
from PIL import Image
import glob
from tqdm import tqdm
from pathlib import Path

# 自作モジュール
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vis_depth import (
    apply_colormap,
    save_depth_as_16bit_png,
    create_comparison_panel
)

def main():
    # 出力ディレクトリ（depth_modelと同じ構造）
    output_dir = Path("outputs_depth_model_style")
    output_dir.mkdir(exist_ok=True)
    
    # UniDepth v2モデルのロード
    try:
        from unidepth.models import UniDepthV2
        from torchvision import transforms
    except ImportError as e:
        print("エラー: UniDepthがインストールされていません。")
        return
    
    # デバイス設定（GPUを優先）
    if torch.cuda.is_available():
        device = "cuda"
        print("[Device] CUDA GPU detected")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("[Device] Apple Silicon GPU detected")
    else:
        device = "cpu"
        print("[Device] No GPU detected, using CPU")
    
    device = torch.device(device)
    
    # モデルロード（depth_modelと同じ）
    model_name = "lpiccinelli/unidepth-v2-vitl14"
    print(f"[UniDepth] Loading model: {model_name}")
    model = UniDepthV2.from_pretrained(model_name).to(device)
    model.eval()
    print(f"[UniDepth] Model loaded successfully on {device}")
    
    # ImageNet正規化（depth_modelと同じ）
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # test_images内の画像を取得
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join("test_images", ext)))
    
    if not image_paths:
        print("test_images/に画像が見つかりません")
        return
    
    print(f"\n[Processing] {len(image_paths)} images")
    
    # 各画像を処理
    for img_path in tqdm(image_paths, desc="UniDepth v2"):
        img_name = Path(img_path).stem
        
        # 画像ごとの出力ディレクトリ（depth_modelと同じ構造）
        image_output_dir = output_dir / img_name
        image_output_dir.mkdir(exist_ok=True)
        
        print(f"\n[Processing] {img_name}")
        
        # RGB画像を読み込み
        rgb_pil = Image.open(img_path).convert("RGB")
        rgb_np = np.array(rgb_pil)
        original_height, original_width = rgb_np.shape[:2]
        
        # テンソル変換と正規化（depth_modelと同じ）
        rgb_tensor = torch.from_numpy(rgb_np).float() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        rgb_tensor = normalize(rgb_tensor)
        
        # バッチ次元を追加
        if rgb_tensor.ndim == 3:
            rgb_tensor = rgb_tensor.unsqueeze(0)
        
        # デバイスに転送
        rgb_tensor = rgb_tensor.to(device)
        
        try:
            # 推論実行
            print("  Running UniDepth inference...")
            with torch.no_grad():
                predictions = model.infer(rgb_tensor)
            
            # 結果の取得（depth_modelと同じ処理）
            if isinstance(predictions, dict):
                depth = predictions.get('depth', predictions.get('pred_depth', None))
            elif hasattr(predictions, 'depth'):
                depth = predictions.depth
            else:
                depth = predictions
            
            # テンソルをnumpyに変換
            if isinstance(depth, torch.Tensor):
                depth_np = depth.squeeze().cpu().float().numpy()
            else:
                depth_np = depth
            
            # 元のサイズにリサイズ（必要な場合）
            if depth_np.shape != (original_height, original_width):
                from scipy.ndimage import zoom
                scale_y = original_height / depth_np.shape[0]
                scale_x = original_width / depth_np.shape[1]
                depth_np = zoom(depth_np, (scale_y, scale_x), order=1)
            
            # 16ビットPNGとして保存（depth_modelと同じ）
            png_path = image_output_dir / "UniDepth_depth16.png"
            save_depth_as_16bit_png(depth_np, str(png_path))
            
            # カラー可視化を保存（depth_modelと同じ処理）
            colored = apply_colormap(depth_np, model_name="UniDepth")
            colored_path = image_output_dir / "UniDepth_colored.png"
            Image.fromarray(colored).save(colored_path)
            
            # numpy形式でも保存
            np.save(image_output_dir / "UniDepth_depth.npy", depth_np)
            
            print(f"  ✓ Depth range: [{np.min(depth_np):.2f}, {np.max(depth_np):.2f}]")
            
            # 比較パネルを作成（単一モデルでも）
            depth_maps = {"UniDepth v2": depth_np}
            panel_path = image_output_dir / "panel.png"
            create_comparison_panel(rgb_np, depth_maps, str(panel_path))
            print(f"[Saved] Comparison panel: {panel_path}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[Complete] Results saved to: {output_dir}")
    print("\n出力ファイルの説明:")
    print("  - UniDepth_depth16.png: 16ビット深度PNG（depth_modelと同じ形式）")
    print("  - UniDepth_colored.png: Viridisカラーマップ（逆数変換済み、近い=明るい）")
    print("  - UniDepth_depth.npy: 生の深度値（メートル単位）")
    print("  - panel.png: 入力画像と深度の比較パネル")

if __name__ == "__main__":
    main()