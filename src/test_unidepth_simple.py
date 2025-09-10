# -*- coding: utf-8 -*-
"""
UniDepth v2 単体テストスクリプト
test_images/の画像で深度推定を実行し、可視化して保存
"""
import os
import numpy as np
import cv2
import torch
from PIL import Image
import glob
from tqdm import tqdm

def colorize_depth(depth_m: np.ndarray, q_lo=2, q_hi=98) -> np.ndarray:
    """深度[m]を2–98%分位で正規化し、TURBOカラーマップで着色"""
    d = depth_m.copy()
    mask = np.isfinite(d)
    if not np.any(mask):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    
    lo = np.percentile(d[mask], q_lo)
    hi = np.percentile(d[mask], q_hi)
    d = np.clip((d - lo) / max(1e-9, (hi - lo)), 0, 1)
    d8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)

def main():
    # 出力ディレクトリ
    outdir = "outputs/unidepth_test"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "depth_npy"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "depth_png"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "depth_viz"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "intrinsics"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "confidence"), exist_ok=True)
    
    # UniDepth v2モデルのロード
    try:
        from unidepth.models import UniDepthV2
    except ImportError as e:
        print("エラー: UniDepthがインストールされていません。")
        print("以下のコマンドでインストールしてください:")
        print("git clone https://github.com/lpiccinelli-eth/UniDepth.git")
        print("cd UniDepth")
        print("pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118")
        return
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # モデルロード（Hugging Faceから）
    model_name = "lpiccinelli/unidepth-v2-vitl14"  # ViT-L版
    print(f"モデルロード中: {model_name}")
    model = UniDepthV2.from_pretrained(model_name).to(device)
    model.eval()
    print("モデルロード完了")
    
    # test_images内の画像を取得
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_paths.extend(glob.glob(os.path.join("test_images", ext)))
    
    if not image_paths:
        print("test_images/に画像が見つかりません")
        return
    
    print(f"{len(image_paths)}枚の画像を処理します")
    
    # 各画像を処理
    for img_path in tqdm(image_paths, desc="UniDepth v2 推論"):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        
        # 画像読み込み（RGB）
        from torchvision import transforms
        
        # ImageNet正規化を定義（UniDepth v2推奨）
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        rgb_pil = Image.open(img_path).convert("RGB")
        rgb_np = np.array(rgb_pil)
        rgb = torch.from_numpy(rgb_np).float() / 255.0  # 0-1に正規化
        rgb = rgb.permute(2, 0, 1)  # C,H,W
        rgb = normalize(rgb)  # ImageNet正規化を適用
        rgb = rgb.unsqueeze(0)  # バッチ次元を追加 (1,C,H,W)
        rgb = rgb.to(device)
        
        # 推論実行
        with torch.inference_mode():
            pred = model.infer(rgb)
        
        # 結果取得
        depth = pred["depth"].detach().cpu().float().numpy()  # メートル単位
        # depthが4次元の場合は最初の2次元を削除 (B,C,H,W) -> (H,W)
        if depth.ndim == 4:
            depth = depth[0, 0]  # バッチ次元とチャンネル次元を削除
        elif depth.ndim == 3:
            depth = depth[0]  # バッチ次元を削除
        
        K = pred["intrinsics"].detach().cpu().float().numpy()  # カメラ内部パラメータ
        # Kが3次元の場合は最初の次元を削除 (B,3,3) -> (3,3)
        if K.ndim == 3:
            K = K[0]
        
        # confidence取得（V2の場合）
        conf = None
        for k in ("confidence", "confidence_map"):
            if k in pred:
                conf = pred[k].detach().cpu().float().numpy()
                # confが4次元の場合は最初の2次元を削除
                if conf.ndim == 4:
                    conf = conf[0, 0]
                elif conf.ndim == 3:
                    conf = conf[0]
                break
        
        # 1. 深度マップ保存（npy形式）
        np.save(os.path.join(outdir, "depth_npy", f"{stem}_depth_m.npy"), depth)
        
        # 2. 深度マップ保存（16bit PNG, mm単位）
        dmax = float(np.nanpercentile(depth, 99))
        depth_mm16 = (np.clip(depth, 0, dmax) * 1000.0).astype(np.uint16)
        cv2.imwrite(os.path.join(outdir, "depth_png", f"{stem}_depth_mm.png"), depth_mm16)
        
        # 3. 深度マップ可視化
        depth_viz = colorize_depth(depth)
        cv2.imwrite(os.path.join(outdir, "depth_viz", f"{stem}_depth_viz.jpg"), depth_viz)
        
        # 4. カメラ内部パラメータ保存
        np.save(os.path.join(outdir, "intrinsics", f"{stem}_K.npy"), K)
        
        # 5. 信頼度マップ保存（あれば）
        if conf is not None:
            np.save(os.path.join(outdir, "confidence", f"{stem}_conf.npy"), conf)
            c8 = (np.clip(conf, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(outdir, "confidence", f"{stem}_conf.png"), c8)
        
        # 統計情報を表示
        print(f"\n{stem}:")
        print(f"  画像サイズ: {depth.shape[1]}x{depth.shape[0]}")
        print(f"  深度範囲: {np.nanmin(depth):.3f}m ~ {np.nanmax(depth):.3f}m")
        print(f"  深度中央値: {np.nanmedian(depth):.3f}m")
        print(f"  焦点距離: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
        print(f"  主点: cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
        if conf is not None:
            print(f"  信頼度範囲: {np.min(conf):.3f} ~ {np.max(conf):.3f}")
    
    print(f"\n完了: 結果は {outdir} に保存されました")
    print("保存内容:")
    print("  - depth_npy/: 深度マップ（メートル単位、npy形式）")
    print("  - depth_png/: 深度マップ（ミリメートル単位、16bit PNG）")
    print("  - depth_viz/: 深度マップ可視化（カラーマップ）")
    print("  - intrinsics/: カメラ内部パラメータ")
    print("  - confidence/: 信頼度マップ（V2のみ）")

if __name__ == "__main__":
    main()