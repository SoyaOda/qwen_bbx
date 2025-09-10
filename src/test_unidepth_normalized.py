# -*- coding: utf-8 -*-
"""
UniDepth v2 テストスクリプト（正規化版）
ImageNet正規化を適用して推論を実行
"""
import os
import numpy as np
import cv2
import torch
from PIL import Image
import glob
from tqdm import tqdm
from torchvision import transforms

def colorize_depth_viridis(depth_m: np.ndarray, q_lo=2, q_hi=98) -> np.ndarray:
    """深度[m]を2–98%分位で正規化し、Viridisカラーマップで着色（depth_modelと同じ）"""
    d = depth_m.copy()
    mask = np.isfinite(d) & (d > 0)
    if not np.any(mask):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    
    lo = np.percentile(d[mask], q_lo)
    hi = np.percentile(d[mask], q_hi)
    d = np.clip((d - lo) / max(1e-6, (hi - lo)), 0, 1)
    
    # Viridisカラーマップを使用
    import matplotlib.cm as cm
    cmap = cm.get_cmap('viridis')
    colored = cmap(d)[:, :, :3]  # RGBAからRGBを取得
    colored = (colored * 255).astype(np.uint8)
    # MatplotlibはRGB、OpenCVはBGRなので変換
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    return colored_bgr

def colorize_depth_turbo(depth_m: np.ndarray, q_lo=2, q_hi=98) -> np.ndarray:
    """深度[m]を2–98%分位で正規化し、TURBOカラーマップで着色（現在の実装）"""
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
    outdir = "outputs/unidepth_normalized"
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "depth_npy"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "depth_viz_viridis"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "depth_viz_turbo"), exist_ok=True)
    os.makedirs(os.path.join(outdir, "comparison"), exist_ok=True)
    
    # UniDepth v2モデルのロード
    try:
        from unidepth.models import UniDepthV2
    except ImportError as e:
        print("エラー: UniDepthがインストールされていません。")
        return
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # モデルロード
    model_name = "lpiccinelli/unidepth-v2-vitl14"
    print(f"モデルロード中: {model_name}")
    model = UniDepthV2.from_pretrained(model_name).to(device)
    model.eval()
    print("モデルロード完了")
    
    # ImageNet正規化を定義（depth_modelプロジェクトと同じ）
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # test_images内の画像を取得（最初の3枚のみ）
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join("test_images", ext)))
    image_paths = image_paths[:3]  # テスト用に3枚のみ
    
    if not image_paths:
        print("test_images/に画像が見つかりません")
        return
    
    print(f"{len(image_paths)}枚の画像を処理します")
    
    # 各画像を処理
    for img_path in tqdm(image_paths, desc="UniDepth v2 推論（正規化版）"):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        
        # 画像読み込み
        pil_img = Image.open(img_path).convert("RGB")
        rgb_np = np.array(pil_img)
        
        # === 方法1: 正規化なし（現在の実装） ===
        rgb_tensor_raw = torch.from_numpy(rgb_np).permute(2, 0, 1).float()
        rgb_tensor_raw = rgb_tensor_raw.unsqueeze(0).to(device)
        
        with torch.inference_mode():
            pred_raw = model.infer(rgb_tensor_raw)
        
        depth_raw = pred_raw["depth"].detach().cpu().float().numpy()
        if depth_raw.ndim == 4:
            depth_raw = depth_raw[0, 0]
        elif depth_raw.ndim == 3:
            depth_raw = depth_raw[0]
        
        # === 方法2: ImageNet正規化あり（depth_modelと同じ） ===
        rgb_tensor_norm = torch.from_numpy(rgb_np).float() / 255.0
        rgb_tensor_norm = rgb_tensor_norm.permute(2, 0, 1)
        rgb_tensor_norm = normalize(rgb_tensor_norm)
        rgb_tensor_norm = rgb_tensor_norm.unsqueeze(0).to(device)
        
        with torch.inference_mode():
            pred_norm = model.infer(rgb_tensor_norm)
        
        depth_norm = pred_norm["depth"].detach().cpu().float().numpy()
        if depth_norm.ndim == 4:
            depth_norm = depth_norm[0, 0]
        elif depth_norm.ndim == 3:
            depth_norm = depth_norm[0]
        
        # 結果を保存
        np.save(os.path.join(outdir, "depth_npy", f"{stem}_raw.npy"), depth_raw)
        np.save(os.path.join(outdir, "depth_npy", f"{stem}_normalized.npy"), depth_norm)
        
        # 可視化（両方のカラーマップで）
        # 正規化なし
        viz_raw_viridis = colorize_depth_viridis(depth_raw)
        viz_raw_turbo = colorize_depth_turbo(depth_raw)
        
        # 正規化あり
        viz_norm_viridis = colorize_depth_viridis(depth_norm)
        viz_norm_turbo = colorize_depth_turbo(depth_norm)
        
        # 保存
        cv2.imwrite(os.path.join(outdir, "depth_viz_viridis", f"{stem}_raw.jpg"), viz_raw_viridis)
        cv2.imwrite(os.path.join(outdir, "depth_viz_viridis", f"{stem}_normalized.jpg"), viz_norm_viridis)
        cv2.imwrite(os.path.join(outdir, "depth_viz_turbo", f"{stem}_raw.jpg"), viz_raw_turbo)
        cv2.imwrite(os.path.join(outdir, "depth_viz_turbo", f"{stem}_normalized.jpg"), viz_norm_turbo)
        
        # 比較画像を作成
        comparison = np.hstack([
            cv2.resize(rgb_np[:,:,::-1], (256, 192)),  # RGB->BGR
            cv2.resize(viz_raw_turbo, (256, 192)),
            cv2.resize(viz_norm_turbo, (256, 192)),
            cv2.resize(viz_norm_viridis, (256, 192))
        ])
        
        # ラベルを追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(comparison, "Original", (10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(comparison, "Raw/Turbo", (266, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(comparison, "Norm/Turbo", (522, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(comparison, "Norm/Viridis", (778, 20), font, 0.5, (255, 255, 255), 1)
        
        cv2.imwrite(os.path.join(outdir, "comparison", f"{stem}_compare.jpg"), comparison)
        
        # 統計情報を表示
        print(f"\n{stem}:")
        print(f"  正規化なし - 深度範囲: {np.nanmin(depth_raw):.3f}m ~ {np.nanmax(depth_raw):.3f}m")
        print(f"  正規化あり - 深度範囲: {np.nanmin(depth_norm):.3f}m ~ {np.nanmax(depth_norm):.3f}m")
        print(f"  差の最大値: {np.nanmax(np.abs(depth_raw - depth_norm)):.3f}m")
    
    print(f"\n完了: 結果は {outdir} に保存されました")
    print("比較画像で以下を確認してください:")
    print("  - Raw/Turbo: 正規化なし＋Turboカラーマップ（現在の実装）")
    print("  - Norm/Turbo: ImageNet正規化＋Turboカラーマップ")
    print("  - Norm/Viridis: ImageNet正規化＋Viridisカラーマップ（depth_modelと同じ）")

if __name__ == "__main__":
    main()