# -*- coding: utf-8 -*-
"""
UniDepth v2 メインスクリプト
Qwen/SAM2の結果を基に深度推定→平面フィッティング→体積計算
"""
import os
import json
import glob
import numpy as np
import cv2
import yaml
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# 自作モジュール
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unidepth_runner import UniDepthEngine
from plane_fit import estimate_plane_from_depth
from volume_estimator import (
    height_map_from_plane,
    pixel_area_map,
    estimate_volumes
)
from vis_depth import (
    apply_colormap,
    save_depth_as_16bit_png,
    colorize_height
)
from visualize import ensure_dir

def load_sam2_summary(json_path: str):
    """SAM2のサマリJSONを読み込み"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_binary_mask(path: str) -> np.ndarray:
    """バイナリマスクPNGを読み込み"""
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127)

def find_mask_files(mask_dir: str, stem: str, det_idx: int, label: str, source: str):
    """マスクファイルのパスを生成"""
    # ラベルを安全なファイル名に変換
    safe_lab = "".join([c if c.isalnum() else "_" for c in label])[:40]
    return os.path.join(mask_dir, f"{stem}_det{det_idx:02d}_{safe_lab}_{source}.png")

def main():
    # 設定ファイル読み込み
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # UniDepth設定
    uni_cfg = cfg.get("unidepth", {})
    plane_cfg = cfg.get("plane", {})
    vol_cfg = cfg.get("volume", {})
    paths = cfg.get("paths", {})
    
    # マスクソース（bplus or large）
    mask_source = cfg.get("mask_source", "large")
    
    # 出力ディレクトリ
    out_root = cfg.get("unidepth_paths", {}).get("out_root", "outputs/unidepth")
    ddir = os.path.join(out_root, "depth")
    cdir = os.path.join(out_root, "conf")
    kdir = os.path.join(out_root, "intrinsics")
    hdir = os.path.join(out_root, "height")
    vdir = os.path.join(out_root, "viz")
    jdir = os.path.join(out_root, "json")
    
    for d in (ddir, cdir, kdir, hdir, vdir, jdir):
        ensure_dir(d)
    
    # UniDepthモデルを初期化
    print("UniDepth v2 モデルを初期化中...")
    engine = UniDepthEngine(
        model_repo=uni_cfg.get("model_repo", "lpiccinelli/unidepth-v2-vitl14"),
        device=uni_cfg.get("device", "cuda")
    )
    
    # 入力画像とSAM2結果のパスを設定
    img_dir = paths.get("input_dir", "test_images")
    sam2_json_dir = paths.get("sam2_json_dir", "outputs/sam2/json")
    mask_dir = paths.get("sam2_mask_dir", "outputs/sam2/masks")
    
    # 処理する画像を取得
    stems = []
    for p in glob.glob(os.path.join(img_dir, "*")):
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            stems.append(os.path.splitext(os.path.basename(p))[0])
    stems.sort()
    
    print(f"\n{len(stems)}枚の画像を処理します")
    
    for stem in tqdm(stems, desc="UniDepth v2 → 平面 → 体積"):
        # 画像パスを検索
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            p = os.path.join(img_dir, f"{stem}{ext}")
            if os.path.exists(p):
                img_path = p
                break
        
        if not img_path:
            continue
        
        print(f"\n処理中: {stem}")
        
        # 1) UniDepth推論
        print("  深度推定...")
        K_scale = uni_cfg.get("K_scale_factor", 6.0)
        pred = engine.infer_image(img_path, K_scale_factor=K_scale)
        depth = pred["depth"]
        K = pred["intrinsics"]
        points = pred["points"]
        conf = pred["confidence"]
        
        # 次元が多い場合は削減
        if depth.ndim == 4:
            depth = depth[0, 0]  # (B,C,H,W) -> (H,W)
        elif depth.ndim == 3:
            depth = depth[0]  # (B,H,W) or (C,H,W) -> (H,W)
        
        if K.ndim == 3:
            K = K[0]  # (B,3,3) -> (3,3)
        
        if conf is not None:
            if conf.ndim == 4:
                conf = conf[0, 0]
            elif conf.ndim == 3:
                conf = conf[0]
        
        H, W = depth.shape
        
        # 深度データを保存
        if uni_cfg.get("save_npy", True):
            np.save(os.path.join(ddir, f"{stem}.npy"), depth)
            np.save(os.path.join(kdir, f"{stem}.K.npy"), K)
            if conf is not None:
                np.save(os.path.join(cdir, f"{stem}.conf.npy"), conf)
        
        if uni_cfg.get("save_png", True):
            save_depth_as_16bit_png(depth, os.path.join(ddir, f"{stem}.png"))
            if conf is not None:
                c8 = (np.clip(conf, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(cdir, f"{stem}.png"), c8)
        
        # 2) SAM2の検出結果を読み込み
        sam2_json_path = os.path.join(sam2_json_dir, f"{stem}.sam2.json")
        if not os.path.exists(sam2_json_path):
            print(f"  SAM2結果が見つかりません: {sam2_json_path}")
            continue
        
        summ = load_sam2_summary(sam2_json_path)
        dets = summ.get("detections", [])
        
        if len(dets) == 0:
            print("  検出結果がありません")
            continue
        
        # マスクを読み込み
        masks = []
        labels = []
        for i, det in enumerate(dets):
            label = det.get("label_ja", det.get("label_en", f"object_{i}"))
            labels.append(label)
            
            # マスクファイルを探す
            mpath = find_mask_files(mask_dir, stem, i, label, mask_source)
            if os.path.exists(mpath):
                m = load_binary_mask(mpath)
                masks.append(m)
            else:
                print(f"  警告: マスクファイルが見つかりません: {mpath}")
                # 空のマスクを追加
                masks.append(np.zeros((H, W), dtype=bool))
        
        # 3) 平面フィッティング
        print("  平面推定...")
        try:
            plane_n, plane_d, points_xyz = estimate_plane_from_depth(
                depth, K, masks,
                margin_px=plane_cfg.get("ring_margin_px", 40),
                dist_th=plane_cfg.get("ransac_threshold_m", 0.006),
                max_iters=plane_cfg.get("ransac_max_iters", 2000),
                min_support=plane_cfg.get("min_support_px", 2000)
            )
        except Exception as e:
            print(f"  平面推定エラー: {e}")
            continue
        
        # 4) 高さマップ生成
        height = height_map_from_plane(points_xyz, plane_n, plane_d, 
                                      clip_negative=vol_cfg.get("clip_negative_height", True))
        
        # 高さマップを保存
        if uni_cfg.get("save_npy", True):
            np.save(os.path.join(hdir, f"{stem}.height.npy"), height)
        
        # 5) 体積計算
        print("  体積計算...")
        volumes = estimate_volumes(
            depth, K, plane_n, plane_d,
            masks, labels,
            confidence=conf,
            use_conf_weight=vol_cfg.get("use_confidence_weight", False)
        )
        
        # 6) 可視化
        print("  可視化...")
        
        # 元画像を読み込み
        img_bgr = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # 深度マップの可視化
        depth_viz = apply_colormap(depth, model_name="UniDepth")
        depth_viz_bgr = cv2.cvtColor(depth_viz, cv2.COLOR_RGB2BGR)
        
        # 高さマップの可視化
        height_viz = colorize_height(height, max_h_m=0.05)  # 5cm上限
        
        # パネル画像を作成（元画像｜深度｜高さ）
        panel = np.concatenate([img_bgr, depth_viz_bgr, height_viz], axis=1)
        
        # ラベルを追加
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, "Original", (10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, "Depth", (W + 10, 30), font, 0.7, (255, 255, 255), 2)
        cv2.putText(panel, "Height", (2*W + 10, 30), font, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(os.path.join(vdir, f"{stem}_panel.jpg"), panel)
        
        # 7) JSON保存
        result_json = {
            "image": os.path.basename(img_path),
            "width": W,
            "height": H,
            "intrinsics": K.tolist(),
            "plane": {
                "normal": plane_n.tolist(),
                "d": float(plane_d)
            },
            "mask_source": mask_source,
            "detections": volumes
        }
        
        with open(os.path.join(jdir, f"{stem}.unidepth.json"), "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)
    
    print(f"\n完了: 結果は {out_root} に保存されました")

if __name__ == "__main__":
    main()