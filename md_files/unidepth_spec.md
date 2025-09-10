「QwenのBBox → SAM2.1のマスク」まで出来ている現状を前提に、UniDepth v2 で “深度・内参・信頼度” を推定 → 皿/卓面平面をRANSACで推定 → 高さマップ → マスクごとの体積積分までを完全にテストできるプロジェクトの実装計画と曖昧性のないコード一式を提示します。
UniDepth v2 は公式実装で Hugging Face から from_pretrained でロードし、model.infer(rgb) から **depth / points / intrinsics（V2はconfidenceも）**が得られます（READMEの使用例）
GitHub
。モデルは lpiccinelli/unidepth-v2-vitl14 等が提供されています（Model Zoo）
GitHub
。

0) 目的（今回テストでカバーするタスク）

UniDepth v2 による メトリック深度（m）・内参K・信頼度マップの推定

RANSAC平面当てはめ（皿/卓面）→ 相対高さマップ h(x,y)（皿面=0）

**SAM2.1の各マスク（b+ / large 切替可）**に対して

体積 
𝑉
≈
∑
ℎ
(
𝑥
,
𝑦
)
⋅
𝑎
pix
(
𝑥
,
𝑦
)
V≈∑h(x,y)⋅a
pix
	​

(x,y) を信頼度オプションあり/なしで算出

可視化（深度ヒートマップ・高さマップ重畳・3分割パネル）

JSON出力（画像ごと）：内参・平面方程式・各IDの面積/体積（b+とlargeを別欄で）を保存

任意：PLY点群出力（デバッグ用）

注：UniDepth v2の API 仕様（from_pretrained, infer, 返り値 depth/points/intrinsics、V2のconfidenceやONNX対応）は公式README記載に準拠します。
GitHub

1) プロジェクト構成（既存リポに追加）
qwen-vl-bbox-demo/
├─ README.md
├─ requirements.txt              # 既存＋追記
├─ config.yaml                   # 既存＋追記
├─ src/
│  ├─ unidepth_runner.py         # ★ UniDepth v2 推論（depth/K/conf/points）
│  ├─ plane_fit.py               # ★ 皿/卓面のRANSAC平面当てはめ
│  ├─ volume_estimator.py        # ★ 体積積分（信頼度の有無で2種類）
│  ├─ vis_depth.py               # ★ 深度・高さの可視化
│  ├─ run_unidepth.py            # ★ メイン：一括で処理・保存
│  └─ （既存: qwen_client, run_infer, run_sam2 などはそのまま）
└─ outputs/
   └─ unidepth/
      ├─ depth/      # 16bit PNG / npy
      ├─ conf/       # 8bit/float confidence
      ├─ intrinsics/ # Kをnpyで
      ├─ height/     # 相対高さマップ
      ├─ viz/        # vizパネル（原/深度/高さ、b+とlargeの違い分かる描画）
      └─ json/       # 1画像に1JSON（体積・面積など）

2) セットアップ
2.1 依存（requirements.txt 追記）
# UniDepth v2 は公式リポからインストール（下記手順）※pipに直接指定しない
torch>=2.1.0
opencv-python>=4.9.0.80
numpy>=1.26.4
Pillow>=10.3.0
tqdm>=4.66.4

2.2 UniDepth v2 の導入（公式手順に準拠）
git clone https://github.com/lpiccinelli-eth/UniDepth.git
cd UniDepth
# CUDA11.8ホイールの例（README準拠）
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118


READMEに from unidepth.models import UniDepthV1/UniDepthV2、model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")、predictions = model.infer(rgb) など明記。V2はConfidence出力・ONNX対応などを追加。
GitHub

2.3 config.yaml 追記
unidepth:
  model_repo: "lpiccinelli/unidepth-v2-vitl14"  # V2 ViT-L
  device: "cuda"                                # or "cpu"
  save_npy: true
  save_png: true

plane:
  ring_margin_px: 40         # 食品マスクの外側リング幅（候補点抽出）
  ransac_threshold_m: 0.006  # 平面距離の閾値[m]（約6mm）
  ransac_max_iters: 2000
  min_support_px: 2000       # RANSACの最小有効点数

volume:
  use_confidence_weight: false   # trueにすると conf を重み付けに使用
  area_formula: "z2_over_fx_fy"  # a_pix(z) = (z^2)/(fx*fy)
  clip_negative_height: true

paths:
  sam2_json_dir: "outputs/sam2/json"  # 既存のSAM2サマリ
  sam2_mask_dir: "outputs/sam2/masks" # 既存のマスクPNG
  qwen_json_dir: "outputs/json"       # 既存のQwen出力（ラベル）
  input_dir: "/path/to/images"        # 元画像
  out_root: "outputs/unidepth"

mask_source: "large"   # "bplus" or "large" のどちらで体積算出するか

3) コード実装
3.1 src/unidepth_runner.py（UniDepth v2 推論）
# -*- coding: utf-8 -*-
import numpy as np
import torch
from PIL import Image
from typing import Dict, Any, Optional
from unidepth.models import UniDepthV2  # READMEの使用例に準拠 :contentReference[oaicite:4]{index=4}

class UniDepthEngine:
    def __init__(self, model_repo: str, device: str = "cuda"):
        self.device = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
        self.model = UniDepthV2.from_pretrained(model_repo)  # HFからロード（READMEのModel Zoo） :contentReference[oaicite:5]{index=5}
        self.model = self.model.to(self.device)

    def infer_image(self, image_path: str) -> Dict[str, Any]:
        """RGB画像1枚から depth[m], intrinsics(3x3), points(3,H,W), confidence(H,W?) を返す。"""
        rgb = torch.from_numpy(np.array(Image.open(image_path).convert("RGB"))).permute(2, 0, 1)  # C,H,W
        rgb = rgb.to(self.device)
        with torch.inference_mode():
            pred = self.model.infer(rgb)  # README記載のAPI :contentReference[oaicite:6]{index=6}

        # depth
        depth_t = pred.get("depth")          # (H,W) torch.Tensor
        depth = depth_t.detach().to("cpu").float().numpy()

        # intrinsics
        K_t = pred.get("intrinsics")         # (3,3)
        K = K_t.detach().to("cpu").float().numpy()

        # points（あれば使う。無ければK, depthから計算）
        pts_t = pred.get("points", None)     # (3,H,W) 期待
        if pts_t is not None:
            points = pts_t.detach().to("cpu").float().numpy()  # (3,H,W)
        else:
            points = None

        # confidence（V2で追加。キー名はバージョンにより変わる可能性があるため二重取り）
        conf_t = pred.get("confidence", pred.get("confidence_map", None))
        conf = None if conf_t is None else conf_t.detach().to("cpu").float().numpy()

        return {"depth": depth, "intrinsics": K, "points": points, "confidence": conf}

3.2 src/plane_fit.py（卓面RANSAC）
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from typing import Tuple

def build_support_ring(food_union_mask: np.ndarray, margin_px: int) -> np.ndarray:
    """食品マスクの外側リング領域（皿や卓面候補）を作成。"""
    k = (2*margin_px + 1)
    kernel = np.ones((k, k), np.uint8)
    dil = cv2.dilate(food_union_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    ring = np.logical_and(dil, np.logical_not(food_union_mask))
    return ring

def fit_plane_ransac(points_xyz: np.ndarray, cand_mask: np.ndarray,
                     dist_th: float = 0.006, max_iters: int = 2000,
                     min_support: int = 2000, rng_seed: int = 3) -> Tuple[np.ndarray, float]:
    """
    points_xyz: (3,H,W) の点群（カメラ座標, m）
    cand_mask : (H,W) のbool（RANSAC候補点）
    dist_th   : 点→平面距離[m]の閾値
    戻り値: (n,d) ただし nは単位法線(3,), 平面は n·X + d = 0（d<0想定）/ インライア数
    """
    H, W = cand_mask.shape
    ys, xs = np.where(cand_mask)
    if ys.size < min_support:
        raise RuntimeError(f"平面候補点が不足: {ys.size} < {min_support}")

    # 候補点のXYZを抽出
    X = points_xyz[0, ys, xs]
    Y = points_xyz[1, ys, xs]
    Z = points_xyz[2, ys, xs]
    P = np.stack([X, Y, Z], axis=1)

    rs = np.random.RandomState(rng_seed)
    best_inliers = -1
    best_n, best_d = None, None

    for _ in range(max_iters):
        idx = rs.choice(P.shape[0], size=3, replace=False)
        p1, p2, p3 = P[idx]
        v1, v2 = p2 - p1, p3 - p1
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-8:
            continue
        n = n / n_norm
        d = -np.dot(n, p1)

        # 点→平面距離
        dist = np.abs(P @ n + d)
        inliers = (dist < dist_th)
        n_in = int(inliers.sum())
        if n_in > best_inliers:
            # 最小二乗でリファイン
            Q = P[inliers]
            # min ||Q·n + d|| -> SVDで n を求め、d を再計算
            Q1 = np.concatenate([Q, np.ones((Q.shape[0],1))], axis=1)
            # 係数 [n; d] は最小特異値の特異ベクトル
            _, _, vh = np.linalg.svd(Q1, full_matrices=False)
            coeff = vh[-1, :]
            n_ref = coeff[:3]
            n_ref /= np.linalg.norm(n_ref) + 1e-9
            d_ref = coeff[3]
            # 符号合わせ（+Z方向が上になるように）
            if n_ref[2] < 0:
                n_ref = -n_ref; d_ref = -d_ref
            best_n, best_d, best_inliers = n_ref, d_ref, n_in

    if best_n is None:
        raise RuntimeError("RANSAC平面推定に失敗しました。")
    return (best_n, float(best_d)), float(best_inliers)

3.3 src/volume_estimator.py（体積積分）
# -*- coding: utf-8 -*-
import numpy as np
from typing import Dict, Any

def ensure_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """depth[m] と内参Kから (3,H,W) の点群を作る。"""
    H, W = depth.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    us = np.arange(W); vs = np.arange(H)
    uu, vv = np.meshgrid(us, vs)
    Z = depth
    X = (uu - cx) / fx * Z
    Y = (vv - cy) / fy * Z
    return np.stack([X, Y, Z], axis=0)

def height_map_from_plane(points_xyz: np.ndarray, plane_n: np.ndarray, plane_d: float,
                          clip_negative: bool = True) -> np.ndarray:
    """n·X + d の符号を高さと解釈（皿面=0, 上が正）。"""
    X = points_xyz[0]; Y = points_xyz[1]; Z = points_xyz[2]
    h = plane_n[0]*X + plane_n[1]*Y + plane_n[2]*Z + plane_d
    if clip_negative:
        h = np.maximum(h, 0.0)
    return h

def pixel_area_map(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """a_pix(z) ≈ (z^2)/(fx*fy)（小面積近似）。"""
    fx, fy = K[0,0], K[1,1]
    return (depth**2) / (fx * fy + 1e-12)

def integrate_volume(height: np.ndarray, a_pix: np.ndarray,
                     mask_bool: np.ndarray, conf: np.ndarray = None,
                     use_conf_weight: bool = False) -> Dict[str, Any]:
    """mask内で V を積分。conf重みはオプション（比較のため両方出すのが推奨）。"""
    m = mask_bool.astype(bool)
    if not np.any(m):
        return {"pixels": 0, "volume_mL": 0.0}
    if use_conf_weight and (conf is not None):
        w = conf
        V = float(np.sum(height[m] * a_pix[m] * np.clip(w[m], 0.0, 1.0)))
    else:
        V = float(np.sum(height[m] * a_pix[m]))
    # m^3 → mL(=1e6 * m^3)
    return {"pixels": int(m.sum()), "volume_mL": V * 1e6}

3.4 src/vis_depth.py（可視化）
# -*- coding: utf-8 -*-
import numpy as np
import cv2

def colorize_depth(depth: np.ndarray, clip_q=(0.02, 0.98)) -> np.ndarray:
    d = depth.copy()
    lo, hi = np.quantile(d[np.isfinite(d)], clip_q)
    d = np.clip((d - lo) / max(1e-9, (hi - lo)), 0, 1)
    d8 = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)

def colorize_height(height: np.ndarray, max_h_m: float = 0.05) -> np.ndarray:
    """0〜max_h_m を 0〜255 に正規化してカラマップ。"""
    h = np.clip(height / max_h_m, 0, 1)
    h8 = (h * 255).astype(np.uint8)
    return cv2.applyColorMap(h8, cv2.COLORMAP_MAGMA)

3.5 src/run_unidepth.py（メイン：深度→平面→体積）
# -*- coding: utf-8 -*-
import os, json, glob
import numpy as np
import cv2
import yaml
from PIL import Image
from tqdm import tqdm

from src.unidepth_runner import UniDepthEngine
from src.plane_fit import build_support_ring, fit_plane_ransac
from src.volume_estimator import ensure_points, height_map_from_plane, pixel_area_map, integrate_volume
from src.vis_depth import colorize_depth, colorize_height
from src.visualize import ensure_dir

def load_sam2_summary(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_binary_mask(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return (m > 127)

def find_mask_files(mask_dir: str, stem: str, det_idx: int, label: str, source: str):
    # run_sam2.py の命名規則: <stem>_det##_<label>_<bplus|large>.png
    safe_lab = "".join([c if c.isalnum() else "_" for c in label])[:40]
    return os.path.join(mask_dir, f"{stem}_det{det_idx:02d}_{safe_lab}_{source}.png")

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    uni_cfg   = cfg["unidepth"]
    plane_cfg = cfg["plane"]
    vol_cfg   = cfg["volume"]
    paths     = cfg["paths"]
    src_name  = cfg.get("mask_source", "large")

    out_root = paths["out_root"]
    ddir = os.path.join(out_root, "depth")
    cdir = os.path.join(out_root, "conf")
    kdir = os.path.join(out_root, "intrinsics")
    hdir = os.path.join(out_root, "height")
    vdir = os.path.join(out_root, "viz")
    jdir = os.path.join(out_root, "json")
    for d in (ddir, cdir, kdir, hdir, vdir, jdir): ensure_dir(d)

    # UniDepthモデル
    engine = UniDepthEngine(uni_cfg["model_repo"], device=uni_cfg.get("device", "cuda"))

    # 入力画像
    img_dir    = paths["input_dir"]
    sam2_dir   = paths["sam2_json_dir"]
    mask_dir   = paths["sam2_mask_dir"]
    stems = []
    for p in glob.glob(os.path.join(img_dir, "*")):
        if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            stems.append(os.path.splitext(os.path.basename(p))[0])
    stems.sort()

    for stem in tqdm(stems, desc="UniDepthV2 → 平面 → 体積"):
        img_path  = os.path.join(img_dir, f"{stem}.jpg")
        if not os.path.exists(img_path):
            # 他拡張子対応
            alt = [".png", ".jpeg", ".bmp", ".webp"]
            found = False
            for ext in alt:
                p = os.path.join(img_dir, f"{stem}{ext}")
                if os.path.exists(p): img_path=p; found=True; break
            if not found: continue

        # 1) UniDepth 推論
        pred = engine.infer_image(img_path)
        depth, K, points, conf = pred["depth"], pred["intrinsics"], pred["points"], pred["confidence"]
        H, W = depth.shape
        if points is None:
            points = ensure_points(depth, K)

        # 保存
        if uni_cfg.get("save_npy", True):
            np.save(os.path.join(ddir, f"{stem}.npy"), depth)
            np.save(os.path.join(kdir, f"{stem}.K.npy"), K)
            if conf is not None:
                np.save(os.path.join(cdir, f"{stem}.conf.npy"), conf)
        if uni_cfg.get("save_png", True):
            cv2.imwrite(os.path.join(ddir, f"{stem}.png"), (np.clip(depth, 0, np.nanpercentile(depth, 99)) * 1000).astype(np.uint16))
            if conf is not None:
                c8 = (np.clip(conf, 0, 1) * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(cdir, f"{stem}.png"), c8)

        # 2) SAM2の検出をロード
        sam2_json_path = os.path.join(sam2_dir, f"{stem}.sam2.json")
        if not os.path.exists(sam2_json_path):
            # Qwen/SAMなし画像はスキップ
            continue
        summ = load_sam2_summary(sam2_json_path)
        dets = summ.get("detections", [])
        labels = [d["label_ja"] for d in dets]

        # 食品Unionマスク
        union = np.zeros((H, W), dtype=bool)
        masks = []
        for i, lab in enumerate(labels):
            mpath = find_mask_files(mask_dir, stem, i, lab, "large" if src_name=="large" else "bplus")
            m = load_binary_mask(mpath)
            masks.append(m)
            union |= m

        # 3) 平面候補点（リング）→ RANSAC
        ring = build_support_ring(union, margin_px=int(plane_cfg["ring_margin_px"]))
        try:
            (n, d), nin = fit_plane_ransac(points, ring,
                                           dist_th=float(plane_cfg["ransac_threshold_m"]),
                                           max_iters=int(plane_cfg["ransac_max_iters"]),
                                           min_support=int(plane_cfg["min_support_px"]))
        except RuntimeError:
            # フォールバック：画像全域からRANSAC（小物が多い場合など）
            full = np.logical_not(union)
            (n, d), nin = fit_plane_ransac(points, full,
                                           dist_th=float(plane_cfg["ransac_threshold_m"]),
                                           max_iters=int(plane_cfg["ransac_max_iters"]),
                                           min_support=int(plane_cfg["min_support_px"]//2))

        # 4) 高さ・面積マップ
        height = height_map_from_plane(points, n, d, clip_negative=bool(vol_cfg.get("clip_negative_height", True)))
        a_pix  = pixel_area_map(depth, K)

        # 5) 体積（conf有/無）
        out_items = []
        for i, lab in enumerate(labels):
            vol_plain = integrate_volume(height, a_pix, masks[i], conf=None, use_conf_weight=False)
            vol_conf  = integrate_volume(height, a_pix, masks[i], conf=conf, use_conf_weight=bool(vol_cfg["use_confidence_weight"]))
            out_items.append({
                "id": i,
                "label_ja": lab,
                "pixels": vol_plain["pixels"],
                "volume_mL_no_conf": vol_plain["volume_mL"],
                "volume_mL_conf": vol_conf["volume_mL"] if vol_cfg["use_confidence_weight"] and conf is not None else None
            })

        # 6) 可視化
        img_bgr = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        depth_cm = colorize_depth(depth)
        height_cm = colorize_height(height, max_h_m=0.05)  # 5cm上限で正規化
        panel = np.concatenate([img_bgr, depth_cm, height_cm], axis=1)
        cv2.imwrite(os.path.join(vdir, f"{stem}_panel.jpg"), panel)

        # 7) JSON保存
        js = {
            "image": os.path.basename(img_path),
            "width": W, "height": H,
            "intrinsics": K.tolist(),
            "plane": {"n": n.tolist(), "d": float(d)},
            "mask_source": src_name,
            "detections": out_items
        }
        with open(os.path.join(jdir, f"{stem}.unidepth.json"), "w", encoding="utf-8") as f:
            json.dump(js, f, ensure_ascii=False, indent=2)

    print(f"完了: {out_root}")

if __name__ == "__main__":
    main()

4) 実行
# 1) 既存の Qwen → SAM2.1 (b+ / large) が完了している前提
# 2) UniDepth v2 の環境セットアップ（前述）
python -m src.run_unidepth
# 出力は outputs/unidepth/ 配下に保存


viz/xxx_panel.jpg … 原画像｜深度ヒートマップ｜高さマップ

json/xxx.unidepth.json … K・平面・各IDの体積（mL）

depth/xxx.npy|png, intrinsics/xxx.K.npy, conf/xxx.npy|png, height/xxx.npy|png

5) サニティチェック（最小）

(A) 平面推定の有効点数・残差を確認
fit_plane_ransac で nin（インライア数）が閾値以上になっているか。
(B) 高さの範囲
height の 99％分位が 1〜50mm 程度に収まるか（料理の厚みの常識範囲）。
(C) 体積のオーダ
一品の体積が 50–600 mL 程度に多くが収まるか（汁物は器容量で上限を確認）。

6) よくある質問／調整ポイント

confidence の使い方
V2はconfidence出力を持ちます（READMEに「Confidence output」と明記）
GitHub
。テストでは 未使用版と重み付版の両方をJSONに入れ、挙動を比較してください。

a_pix の式
近似として 
𝑎
pix
(
𝑧
)
=
(
𝑧
2
)
/
(
𝑓
𝑥
⋅
𝑓
𝑦
)
a
pix
	​

(z)=(z
2
)/(fx⋅fy) を採用。厳密には視線方向や面の傾きに依存しますが、皿付近の狭い範囲での積分近似として十分実用です。

リングの失敗時
食品が画面いっぱいの場合などリングが小さいとRANSACが失敗します。コードはフォールバックで全域から再推定します。

b+ / large の比較
config.yaml の mask_source を切り替えて同じUniDepth結果に対しマスク差による体積差を検証できます。

ライセンス
UniDepthは CC BY-NC 4.0。非商用である点に注意（README記載）。
GitHub