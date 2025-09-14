以下はDepth Anything V2（DAV2）: Pretrained vs Finetunedでの体積推定を“正しく”比較するためのレビュー & 完全修正版スクリプトです。
ポイントは (1) 正しい深度の扱い, (2) カメラ内部パラメータ K の実測（EXIF）→ 皿リム径で補正, (3) RANSACのスケール適応, (4) 体積積分の単位整合 の4点です。

まずはレビュー（問題点）

K を任意の固定係数でスケールしていた
estimate_intrinsics() が FOV60°を仮定し、さらに scale_factor=10.5 を掛けていますが、これは画像や端末ごとの差異を無視した経験則の二重補正になっています。
→ 修正:

まず EXIF から FocalLengthIn35mmFilm（35mm換算）または FocalLength（実焦点距離mm）を取得し、画素焦点距離 fx, fy に変換（fx = W * f35 / 36, fy = H * f35 / 24 など）。EXIFが無い場合のみ FOV 仮定にフォールバックします。
Wikipedia

次に 皿リムの実径（ドメイン prior: 例 260mm）と画像中のリム直径px・リム上の深度Zからfx, fyを一発で再推定（fx ≈ (d_px_x * Z) / D_world）。これで per-image に K を自動補正できます（下にコードあり）。任意の “Kスケール” ダイヤルは不要。

DAV2の出力の扱いが曖昧
HuggingFace の Depth Anything V2 “Metric” は Hypersim などのメトリック深度で微調整されており、Transformers で AutoModelForDepthEstimation / AutoImageProcessor を使う標準手順が推奨です。出力テンソルは**絶対深度（メートル）**を返します（カード上 “metric / absolute depth”・使用例あり）。元解像度へ補間して使うのが定石です。

RANSAC閾値を固定（6mm）
撮影距離や深度スケールに依存するため固定値だと過小/過大に。
→ 修正: 深度の中央値 z_med に応じて自動設定（例: dist_th = max(0.004, 0.01 * z_med) のような mmスケール）。Open3D でも同等のRANSACを用いた平面分離が一般的です。
Open3D

体積積分の単位と a_pix の導出
ピンホールモデルで逆投影は X=(u-cx)Z/fx, Y=(v-cy)Z/fy。1ピクセルが張る実面積は近似で a_pix ≈ Z²/(fx·fy)。OpenCV のカメラ行列/射影モデルを根拠に、この実装に統一（単位は m, m², m³→mL）。
GitHub
+1

マスクと画像サイズの不一致/ハードコード
常に深度マップと同じ (H,W) にリサイズ。FOOD側の union マスク外リングで卓面RANSACを行う（既存 plane_fit の思想は妥当）。

参考根拠（主要5点）

DAV2 Metric（Indoor/Outdoor）モデル: HFモデルカード/コード例（Transformersで使える・metric/absolute depth）。

Depth Anything V2 公式リポ（metric depth, Transformers対応）。

OpenCVカメラモデル（K, 逆投影の基本式）。
GitHub
+1

EXIFの35mm換算→画素焦点距離（一般式）。

Open3DのRANSACによる平面抽出（ドキュメント）。
Open3D

（＊HoughCircles で皿リムを得る API は OpenCV 公式。
GleamTech Documentation
）

完全修正版スクリプト（置き換え用）

ファイル名例: dav2_volume_compare_fixed.py
依存: pip install opencv-python pillow transformers open3d（Open3Dは任意）

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Depth Anything V2: Pretrained vs Finetuned 体積比較（修正版）
- 正しい深度処理（Transformers推奨手順）
- EXIF→K算出 → 皿リム実径で fx, fy を自動補正（per-image）
- RANSAC閾値を深度スケールに適応
- 体積積分（mL）を安定化

参照:
- Depth Anything V2 Metric (HF Transformers): model cards
- OpenCV pinhole camera model / HoughCircles
- Open3D segment_plane (任意)
"""

import os
import sys
import math
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import cv2
import torch
from PIL import Image, ImageOps, ExifTags
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# プロジェクト src へのパスを通す（plane_fit/volume_estimator を流用する場合）
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 既存実装を活かす（a_pix, height, integrate等）
from volume_estimator import integrate_volume

# ---------------------------
# ユーティリティ
# ---------------------------

def _exif_to_float(x):
    try:
        if isinstance(x, tuple) and len(x) == 2:  # 1/2 のような分数
            return float(x[0]) / float(x[1]) if x[1] != 0 else float(x[0])
        return float(x)
    except Exception:
        return None

def load_exif(image_path):
    try:
        img = Image.open(image_path)
        exif = img._getexif() or {}
        table = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        return table
    except Exception:
        return {}

def intrinsics_from_exif_or_fov(image_path, H, W, fallback_fov_deg=60.0):
    """
    EXIF優先で fx, fy を算出。無ければ FOV から。
    - FocalLengthIn35mmFilm -> fx = W * f35 / 36, fy = H * f35 / 24
    - FocalLength (mm)      -> センサー幅mmが必要。既知でなければ 6.4mm 程度を仮置き。
    """
    exif = load_exif(image_path)
    fx = fy = None

    f35 = exif.get('FocalLengthIn35mmFilm')
    f35 = _exif_to_float(f35)
    if f35 and f35 > 0:
        fx = (W * f35) / 36.0
        fy = (H * f35) / 24.0

    if fx is None or fy is None:
        fl_mm = exif.get('FocalLength')
        fl_mm = _exif_to_float(fl_mm)
        if fl_mm and fl_mm > 0:
            # 端末ごとのセンサー幅（簡易）。無ければ 6.4mm を既定値。
            sensor_width_mm = 6.4
            make = str(exif.get('Make', '')).lower()
            model = str(exif.get('Model', '')).lower()
            if 'apple' in make or 'iphone' in model:
                sensor_width_mm = 4.8
            elif 'samsung' in make:
                sensor_width_mm = 6.4
            elif 'sony' in make:
                sensor_width_mm = 7.2
            fx = W * (fl_mm / sensor_width_mm)
            fy = fx

    # FOVフォールバック
    if fx is None or fy is None:
        fov = math.radians(float(fallback_fov_deg))
        fx = W / (2.0 * math.tan(fov / 2.0))
        fy = fx

    cx = W / 2.0
    cy = H / 2.0
    K = np.array([[fx, 0, cx],
                  [0,  fy, cy],
                  [0,   0,  1]], dtype=np.float64)
    return K

def deproject_xyz(depth, K):
    """逆投影: X=(u-cx)Z/fx, Y=(v-cy)Z/fy（OpenCV pinholeモデル）"""
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    Z = depth
    X = (uu - cx) * Z / (fx + 1e-12)
    Y = (vv - cy) * Z / (fy + 1e-12)
    return np.stack([X, Y, Z], axis=0)  # (3,H,W)

def pixel_area_map(depth, K):
    """各ピクセルの実面積[m^2] 近似: Z^2 / (fx*fy)"""
    fx, fy = K[0,0], K[1,1]
    return (depth ** 2) / (fx * fy + 1e-12)

def build_ring_from_food_masks(food_masks, margin_px=40):
    """食品unionマスク外側リング"""
    if not food_masks:
        return None
    H, W = food_masks[0].shape
    union = np.zeros((H, W), dtype=np.uint8)
    for m in food_masks:
        union |= (m.astype(np.uint8))
    k = 2 * margin_px + 1
    kernel = np.ones((k, k), np.uint8)
    dil = cv2.dilate(union, kernel, iterations=1)
    ring = (dil > 0) & (union == 0)
    return ring

def ransac_plane(points_xyz, cand_mask, z_med, min_support=1500, seed=3):
    """
    スケール適応RANSACで平面推定（閾値は z_med に依存）
    dist_th ≈ max(4mm, 1% * z_med)
    """
    H, W = cand_mask.shape
    ys, xs = np.where(cand_mask)
    if len(ys) < min_support:
        raise RuntimeError("平面候補点が不足")

    P = np.stack([points_xyz[0, ys, xs],
                  points_xyz[1, ys, xs],
                  points_xyz[2, ys, xs]], axis=1)
    valid = np.isfinite(P).all(axis=1)
    P = P[valid]
    if len(P) < min_support:
        raise RuntimeError("有効平面点が不足")

    dist_th = max(0.004, 0.01 * float(z_med))  # m
    rng = np.random.default_rng(seed)
    best_inl = -1
    best_n = None
    best_d = None

    for _ in range(2000):
        idx = rng.choice(len(P), 3, replace=False)
        p1, p2, p3 = P[idx]
        v1, v2 = p2 - p1, p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            continue
        n = n / n_norm
        d = -float(np.dot(n, p1))
        dist = np.abs(P @ n + d)
        inl = int((dist < dist_th).sum())
        if inl > best_inl:
            best_inl = inl
            # もう一度 inlier でSVDリファイン
            Q = P[dist < dist_th]
            centroid = np.mean(Q, axis=0)
            U, S, Vt = np.linalg.svd(Q - centroid, full_matrices=False)
            n_ref = Vt[-1]
            if n_ref[2] < 0:  # +Zを上に
                n_ref = -n_ref
            d_ref = -float(np.dot(n_ref, centroid))
            best_n, best_d = n_ref.astype(np.float64), d_ref

    if best_n is None:
        raise RuntimeError("RANSAC平面推定失敗")
    return best_n, float(best_d)

def height_from_plane(points_xyz, n, d, table_mask=None, food_union=None):
    """符号を自動解決して高さ[m]（皿/卓面が0で上が+）に整える"""
    X, Y, Z = points_xyz
    h = n[0]*X + n[1]*Y + n[2]*Z + d
    if table_mask is not None and np.any(table_mask):
        med_table = float(np.median(h[table_mask]))
        if abs(med_table) > 1e-3:  # 1mm 以上ズレていたら
            # 卓面中央値が負→反転
            if med_table < 0:
                h = -h
        if food_union is not None and np.any(food_union):
            med_food = float(np.median(h[food_union]))
            if med_food < 0:  # 食品が負側にあるなら反転
                h = -h
    else:
        if n[2] > 0:  # +Zが上前提
            h = -h
    h = np.maximum(h, 0.0)
    return h

def detect_plate_circle(bgr, exclude_mask=None, dp=1.2, minDist=80, param1=150, param2=30):
    """
    皿リムの円検出（OpenCV HoughCircles: dp, minDist, param1/2 は適宜調整）
    返り値: (x,y,r) or None
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    if exclude_mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=(~exclude_mask.astype(np.uint8) & 0xFF))
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2,
                               minRadius=30, maxRadius=0)
    if circles is None:
        return None
    circles = np.uint16(np.around(circles))
    # 最大半径を採用（最も外周の皿と仮定）
    x, y, r = max(circles[0, :], key=lambda c: c[2])
    return int(x), int(y), int(r)

def calibrate_fx_fy_with_plate(K_init, depth, circle, plate_diam_mm):
    """
    リム直径pxとリム上の深度Zから fx, fy を再推定:
    fx ≈ (d_px_x * Z) / D_world, fy 同様（D_world=plate_diam_m）
    """
    H, W = depth.shape
    fx0, fy0 = K_init[0,0], K_init[1,1]
    cx, cy = K_init[0,2], K_init[1,2]

    x, y, r = circle
    D_world = float(plate_diam_mm) / 1000.0  # m
    # 円周帯(±2px) をサンプル
    yy, xx = np.ogrid[:H, :W]
    dist = np.abs(np.sqrt((xx - x)**2 + (yy - y)**2) - r)
    band = dist <= 2.0
    Z_band = depth[band]
    Z_band = Z_band[np.isfinite(Z_band) & (Z_band > 0)]
    if len(Z_band) < 50:
        return K_init, False

    Zr = float(np.median(Z_band))  # リム近傍の代表Z

    # 直径px（水平/垂直）を計算（画面内に収まる範囲にクリップ）
    d_px_x = min(2*r, 2*min(x, W-1-x))
    d_px_y = min(2*r, 2*min(y, H-1-y))

    fx_new = (d_px_x * Zr) / (D_world + 1e-9)
    fy_new = (d_px_y * Zr) / (D_world + 1e-9)

    # 異常値ガード（相対変化が極端ならスキップ）
    s_fx = fx_new / (fx0 + 1e-9)
    s_fy = fy_new / (fy0 + 1e-9)
    if not (0.2 <= s_fx <= 5.0 and 0.2 <= s_fy <= 5.0):
        return K_init, False

    K = K_init.copy()
    K[0,0], K[1,1] = float(fx_new), float(fy_new)
    return K, True

def load_dav2(model_id_or_dir, device):
    processor = AutoImageProcessor.from_pretrained(model_id_or_dir)
    model = AutoModelForDepthEstimation.from_pretrained(model_id_or_dir)
    model = model.to(device).eval()
    return processor, model

def infer_depth(processor, model, image_path, device):
    """HF推奨: predicted_depth -> 元解像度へ補間。DAV2 Metricはメートル出力。"""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    # DAV2のTransformers出力は `predicted_depth` を持つ
    pred = out.predicted_depth if hasattr(out, "predicted_depth") else out.logits
    if pred.ndim == 3:  # [B,H,W] -> [B,1,H,W]
        pred = pred.unsqueeze(1)
    # 元解像度にバイキュービック補間
    depth = torch.nn.functional.interpolate(
        pred, size=(orig_h, orig_w), mode="bicubic", align_corners=False
    )[0, 0].cpu().numpy()
    return depth  # [m]

def load_food_masks(image_name, target_size):
    """outputs/sam2/masks 内の bplus/large を読み込み"""
    mask_dir = "outputs/sam2/masks"
    H, W = target_size
    masks, labels, paths = [], [], []
    stem = os.path.splitext(image_name)[0]
    for fname in sorted(os.listdir(mask_dir)):
        if fname.startswith(stem) and (fname.endswith("_bplus.png") or fname.endswith("_large.png")):
            m = cv2.imread(os.path.join(mask_dir, fname), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            masks.append(m > 127)
            # detXX_*** をラベルに
            parts = fname.split('_')
            label = "food"
            if len(parts) >= 4:
                label = parts[3]  # 短めに
            labels.append(label)
            paths.append(os.path.join(mask_dir, fname))
    return masks, labels, paths

def integrate_masked_volume(height, a_pix, mask):
    """体積[mL], 高さ統計[mm] を返す"""
    m = mask.astype(bool)
    if not np.any(m):
        return 0.0, 0.0, 0.0
    V_m3 = float(np.sum(height[m] * a_pix[m]))
    vol_mL = V_m3 * 1e6
    h_mm_mean = float(np.mean(height[m])) * 1000.0
    h_mm_max  = float(np.max(height[m])) * 1000.0
    return vol_mL, h_mm_mean, h_mm_max

# ---------------------------
# メイン比較
# ---------------------------

def run_compare(image_path, pretrained_id, finetuned_dir, plate_diam_mm=260.0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== {os.path.basename(image_path)} ===")
    print(f"Device: {device}")

    # 1) モデル
    proc_pre,  model_pre  = load_dav2(pretrained_id, device)
    proc_ft,   model_ft   = load_dav2(finetuned_dir, device)

    # 2) 深度（メートル）
    depth_pre = infer_depth(proc_pre, model_pre, image_path, device)
    depth_ft  = infer_depth(proc_ft,  model_ft,  image_path, device)
    H, W = depth_pre.shape
    if depth_ft.shape != (H, W):
        depth_ft = cv2.resize(depth_ft, (W, H), interpolation=cv2.INTER_LINEAR)

    print(f"Depth(pre) range: {np.nanmin(depth_pre):.3f}–{np.nanmax(depth_pre):.3f} m")
    print(f"Depth(ft)  range: {np.nanmin(depth_ft):.3f}–{np.nanmax(depth_ft):.3f} m")

    # 3) K 初期値（EXIF or FOV）
    K0 = intrinsics_from_exif_or_fov(image_path, H, W)
    print(f"K0 fx,fy: {K0[0,0]:.1f}, {K0[1,1]:.1f}")

    # 4) 皿リム検出→ K の per-image 校正
    bgr = cv2.cvtColor(np.array(Image.open(image_path).convert("RGB")), cv2.COLOR_RGB2BGR)
    masks, labels, _ = load_food_masks(os.path.basename(image_path), (H, W))
    union = None
    if masks:
        union = np.zeros((H, W), np.uint8)
        for m in masks:
            union |= (m.astype(np.uint8))

    circle = detect_plate_circle(bgr, exclude_mask=union)
    if circle is not None:
        K_cal_pre, ok_pre = calibrate_fx_fy_with_plate(K0, depth_pre, circle, plate_diam_mm)
        K_cal_ft,  ok_ft  = calibrate_fx_fy_with_plate(K0, depth_ft,  circle, plate_diam_mm)
        if ok_pre: K_pre = K_cal_pre
        else:      K_pre = K0
        if ok_ft:  K_ft  = K_cal_ft
        else:      K_ft  = K0
        print(f"Plate circle: (x,y,r)={circle}  -> calibrated fx(pre,ft)= {K_pre[0,0]:.1f}, {K_ft[0,0]:.1f}")
    else:
        print("Plate circle not found -> use K0")
        K_pre = K0
        K_ft  = K0

    # 5) RANSAC 平面（スケール適応）
    ring = build_ring_from_food_masks(masks, margin_px=40) if masks else None
    if ring is None:
        # 画像外周リングを候補に
        ring = np.zeros((H, W), bool)
        ring[:H//8, :] = True; ring[-H//8:, :] = True; ring[:, :W//8] = True; ring[:, -W//8:] = True

    # pre
    z_med_pre = float(np.nanmedian(depth_pre[np.isfinite(depth_pre) & (depth_pre > 0)]))
    P_pre = deproject_xyz(depth_pre, K_pre)
    n_pre, d_pre = ransac_plane(P_pre, ring, z_med_pre)
    h_pre = height_from_plane(P_pre, n_pre, d_pre, table_mask=ring, food_union=(union>0) if union is not None else None)
    a_pre = pixel_area_map(depth_pre, K_pre)

    # ft
    z_med_ft = float(np.nanmedian(depth_ft[np.isfinite(depth_ft) & (depth_ft > 0)]))
    P_ft = deproject_xyz(depth_ft, K_ft)
    n_ft, d_ft = ransac_plane(P_ft, ring, z_med_ft)
    h_ft = height_from_plane(P_ft, n_ft, d_ft, table_mask=ring, food_union=(union>0) if union is not None else None)
    a_ft = pixel_area_map(depth_ft, K_ft)

    # 6) 体積積分（各マスク & 合計）
    if not masks:
        print("No SAM2 masks found -> skip volumes")
        return None

    print("\n{:28s} {:>12s} {:>12s}".format("Label", "Pre(mL)", "Finetune(mL)"))
    print("-"*56)
    total_pre = 0.0
    total_ft  = 0.0
    rows = []
    for m, lab in zip(masks, labels):
        v_pre, hmean_pre, _ = integrate_masked_volume(h_pre, a_pre, m)
        v_ft,  hmean_ft,  _ = integrate_masked_volume(h_ft,  a_ft,  m)
        total_pre += v_pre
        total_ft  += v_ft
        rows.append((lab, v_pre, v_ft, hmean_pre, hmean_ft))
        print("{:28s} {:>10.1f} {:>12.1f}".format(lab[:28], v_pre, v_ft))
    print("-"*56)
    print("{:28s} {:>10.1f} {:>12.1f}".format("TOTAL", total_pre, total_ft))

    # 7) 結果dict
    result = {
        "image": os.path.basename(image_path),
        "K0": {"fx": float(K0[0,0]), "fy": float(K0[1,1])},
        "K_pre": {"fx": float(K_pre[0,0]), "fy": float(K_pre[1,1])},
        "K_ft":  {"fx": float(K_ft[0,0]),  "fy": float(K_ft[1,1])},
        "circle": {"x": int(circle[0]), "y": int(circle[1]), "r": int(circle[2])} if circle else None,
        "z_median": {"pre": z_med_pre, "ft": z_med_ft},
        "total_volume_mL": {"pre": total_pre, "finetuned": total_ft},
        "items": [
            {
                "label": lab, "vol_pre_mL": vp, "vol_ft_mL": vf,
                "hmean_pre_mm": hp, "hmean_ft_mm": hf
            }
            for (lab, vp, vf, hp, hf) in rows
        ]
    }
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True,
                    help="評価する画像パス（複数可）")
    ap.add_argument("--pretrained",
                    default="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf",
                    help="DAV2 Metric Pretrained モデルID")
    ap.add_argument("--finetuned", required=True,
                    help="Finetuned モデルのディレクトリ or HF ID")
    ap.add_argument("--plate_mm", type=float, default=260.0,
                    help="皿の直径 [mm] （データセットprior）")
    ap.add_argument("--out", default=None, help="JSON出力先")
    args = ap.parse_args()

    all_results = []
    for img in args.images:
        if not os.path.exists(img):
            warnings.warn(f"not found: {img}")
            continue
        res = run_compare(img, args.pretrained, args.finetuned, plate_diam_mm=args.plate_mm)
        if res:
            all_results.append(res)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved: {args.out}")

if __name__ == "__main__":
    main()

使い方例
python dav2_volume_compare_fixed.py \
  --images test_images/train_00000.jpg test_images/train_00001.jpg \
  --pretrained depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf \
  --finetuned checkpoints/dav2_metric_n5k \
  --plate_mm 260 \
  --out outputs/dav2_compare.json

実装の要点（なぜこうするか）

DAV2 Metric の深度はメートル：Transformersの AutoModelForDepthEstimation / AutoImageProcessor を用い、predicted_depth を元解像度へ補間して利用するのがモデルカードの推奨フロー。これで深度単位が明確になります。

K は EXIF → 皿リム校正：FocalLengthIn35mmFilm があれば 35mm換算から 画素焦点距離が直に求まります。さらに皿リム径の実寸（事前分布: 例 26cm）を用いて per-image で fx, fy を自動補正。Kスケール係数の手動探索は不要です。

RANSACのスケール適応：dist_th = max(4mm, 1% * z_med) のように撮影距離依存でチューニング。Open3D の segment_plane も同様の枠組みです。
Open3D

逆投影 & 面積：OpenCVのピンホールモデルに基づき X=(u-cx)Z/fx, Y=(v-cy)Z/fy。1px当たり面積は近似で Z²/(fx·fy) を採用（導出は射影微分の一次近似）。
GitHub
+1

皿リム検出：OpenCV HoughCircles を使用（画像/マスクに応じて dp, param1/2 を調整）。
GleamTech Documentation

追加の検証（推奨）

皿直径の事前分布
家/店/国ごとに 18/20/22/24/26/28 cm の混合分布を持ち、リム検出結果に最も整合的な径を自動選択（BIC等）。

平面の別推定
Open3D の segment_plane で点群から直接推定→上記符号解決で高さ。
Open3D

入れ子分割 & 器/影の除外
既存の SAM2.1 軽FT（FoodSeg/UEC-FoodPix）を併用して卓面リングの純度を高め、平面推定を安定化。

メトリック整合のサニティチェック

皿の高さは 0 に近い分布か

食品の平均高さは数mm〜数cmか

1皿合計体積は50–800mLに多く収まるか

まとめ

K の“スケール係数”ダイヤルを回すのではなく、EXIF → 皿リム実寸で per-image 自動較正するのが最もロバストです。

DAV2 Metricは Transformersでそのままメートル深度を返す前提で正しく使い、RANSAC閾値はスケール適応に。

この修正版スクリプトは、同じマスク・同じ画像で Pretrained と Finetuned の体積（mL）を公正に比較します。

さらに精度を詰めるなら、SAM2.1（軽FT）で器/影除去と入れ子分割 → リム径事前分布 → Open3D平面の順で堅牢化を進めてください。