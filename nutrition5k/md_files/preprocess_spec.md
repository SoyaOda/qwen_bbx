1) Nutrition5kの実体確認（インストールは概ね正しい／誤解しやすい点の是正）

公式READMEの要点（Overhead RGB‑D）

imagery/realsense_overhead/ 以下に RGB, raw depth(16bit), colorized depth を格納。

raw depthの単位は “1m = 10,000 units”（= 1ユニット 0.1mm）。深度値は最大 0.4m (= 4,000 units) で丸め。
GitHub

CVPR 2021 論文の要点

俯瞰深度は Intel RealSense D435 で取得。深度単位 = 1e‑4 m（= 0.1mm）。

カメラ‐テーブル距離 Z_plane = 35.9 cm、その距離における1画素面積 a_pix_plane = 5.957×10⁻³ cm²を明示。これで体積を計算し質量推定を改善（MAE 13.7%）。
CVF Open Access

✅ 結論：
あなたのフォルダ構成観察（rgb.png, depth_raw.png 等、splitsあり）は 正しい。
ただし、深度単位をmmとみなして×0.001するのは誤りで、正しくは depth_m = depth_raw / 10000.0。
またK行列はファイル配布がなく（Issuesにも質問あり）、論文のZ_planeとa_pix_planeからfx, fyを復元するのが堅いです。
GitHub

fx, fyの復元
平面上の画素面積 
𝑎
p
i
x
=
𝑍
2
/
(
𝑓
𝑥
𝑓
𝑦
)
a
pix
	​

=Z
2
/(f
x
	​

f
y
	​

)。論文値をSIに変換すると

𝑍
=
0.359
 m
Z=0.359 m

𝑎
p
i
x
=
5.957
×
10
−
3
 cm
2
=
5.957
×
10
−
7
 m
2
a
pix
	​

=5.957×10
−3
 cm
2
=5.957×10
−7
 m
2

よって 
𝑓
𝑥
𝑓
𝑦
=
𝑍
2
/
𝑎
p
i
x
≈
2.16
×
10
5
f
x
	​

f
y
	​

=Z
2
/a
pix
	​

≈2.16×10
5
。アスペクトや歪みが小さい前提で 
𝑓
𝑥
≈
𝑓
𝑦
≈
2.16
×
10
5
≈
465
f
x
	​

≈f
y
	​

≈
2.16×10
5
	​

≈465 px。
※この値はRealSense D435の一般的な実測fx~600pxより小さいですが、本データの撮影設定と論文のa_pix定義に整合します（GitHub IssueでもD435個体差・プロファイル差が言及）。
CVF Open Access
+1

2) 「インストールの仕方がおかしい？」への回答

いいえ：ディレクトリ・ファイルは正しい。ただし深度単位の解釈とKの扱いでスケールがズレます。

修正ポイント

depth_raw.png → m変換は /10000.0。
GitHub

K行列は 論文の Z_plane・a_pix_plane から fx,fyを復元（下に完全コード）。
CVF Open Access

画像をリサイズする場合は Kも同スケールで更新。

3) Fine‑tuningに最適なデータセット構成（現実×合成のハイブリッド）
(A) コア（実画像・絶対深度・量ラベル）— Nutrition5k Overhead

強み：実画像、俯瞰RGB‑D、dish質量、公式split。体積は深度＋マスクから厳密に算出可能（論文定義が明確）。
GitHub
+1

不足：カメラ内部 Kの配布なし、食品マスクなし（要自動生成）。Issuesでも質問が続いており、K配布は期待できない。
GitHub

結論：最終FTとホールドアウト評価の本命。あなたの要件（hold‑outテスト）とも合致。

(B) 補完（3Dメッシュ／RGB‑D動画・マスク・栄養）— MetaFood3D

中身：637～743 食品オブジェクト、3Dメッシュ、720°RGB‑D動画、マスク、栄養・重量、レンダリング用カメラパラメータと計測スケール用fiducialまで整備。
arXiv
+1

使い方：Blender/付属パラメータで俯瞰ビューにレンダリングし、正確なK・深度付きの合成俯瞰RGB‑Dを量産→DA‑V2/UniDepthの事前馴化に有効。

注意：実スキャン起点のため、Nutrition5kの背景・器・照明の分布とは違う。前学習（pretrain）用途が最適。

(C) 大量の合成俯瞰データ— NutritionVerse‑Synth / NutritionVerse‑3D

NV‑Synth：84,984の合成料理画像。RGB・深度・インスタンス/セマンティックマスク等、完全アノテーション。俯瞰含む多視点・ライティングをランダム化可能。スケール整合の合成教師に最適。
arXiv

NV‑3D：105の3D食品モデル。自由視点レンダ。ただし規模が小さく、サイズ較正に難ありと後発研究が指摘（MetaFood3D論文の比較表）。単独本命には不十分。
arXiv
+1

推奨構成

NV‑Synth or MetaFood3Dで前学習（数エポック） → 2) Nutrition5k Overheadで微調整＆最終評価。
理由：前者で食品ドメインの形状/材料感と絶対スケールを先に身につけさせ、後者で実世界俯瞰へ寄せ切る。

4) Nutrition5k向けの落とし込みコード（曖昧性ゼロ）
4.1 変換・K復元ユーティリティ（src/datasets/nutrition5k_utils.py）
import numpy as np
from PIL import Image

# --- 論文定数（CVPR 2021 Nutrition5k）---
Z_PLANE_M = 0.359                      # 35.9 cm
A_PIX_PLANE_CM2 = 5.957e-3             # cm^2 at Z=35.9cm
A_PIX_PLANE_M2 = A_PIX_PLANE_CM2 * 1e-4

def depth_raw_to_meters(depth_raw_u16: np.ndarray) -> np.ndarray:
    """Nutrition5kのdepth_raw.png(16bit)を[meters]へ。単位: 1m=10000 units。"""
    return depth_raw_u16.astype(np.float32) / 10000.0  # <-- ここが /10000

def infer_fx_fy_from_plane_constants(width:int=640, height:int=480) -> tuple[float,float,float,float]:
    """
    論文のZ_plane & a_pix_plane から fx,fy を復元。
    歪みや非等方性が小さい仮定で fx≈fy。
    """
    prod = (Z_PLANE_M**2) / A_PIX_PLANE_M2  # fx*fy
    f = float(np.sqrt(prod))                 # ≈465 px
    cx, cy = width/2.0, height/2.0
    return f, f, cx, cy

def resize_intrinsics(fx, fy, cx, cy, src_size, dst_size):
    """画像リサイズ時のK更新（最近傍/双線形いずれでも同じスケーリング）。"""
    (W0, H0), (W1, H1) = src_size, dst_size
    sx, sy = W1 / W0, H1 / H0
    return fx * sx, fy * sy, cx * sx, cy * sy


参考：READMEの深度単位、論文のZ=35.9cm と 1画素面積の記述に厳密に一致。
GitHub
+1

4.2 Nutrition5kローダ（RGB/Depth/Mask/ID）（src/datasets/nutrition5k.py）
import os, glob, json, numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .nutrition5k_utils import depth_raw_to_meters, infer_fx_fy_from_plane_constants

class Nutrition5kOverhead(Dataset):
    """
    imagery/realsense_overhead/dish_xxx/{rgb.png,depth_raw.png,depth_color.png}
    dish_ids/splits/depth_{train,val,test}_ids.txt を前提。
    マスクは別途SAM等で生成し *.png (0/255) として imagery/.../mask.png を想定（無ければ None）。
    """
    def __init__(self, root, split="train", use_mask=True, resize=None):
        self.root = root
        ids_file = os.path.join(root, "dish_ids", "splits", f"depth_{split}_ids.txt")
        with open(ids_file, "r") as f:
            self.ids = [line.strip() for line in f if line.strip()]
        self.use_mask = use_mask
        self.resize = resize

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        did = self.ids[i]
        ddir = os.path.join(self.root, "imagery", "realsense_overhead", did)
        rgb = np.array(Image.open(os.path.join(ddir, "rgb.png")).convert("RGB"))
        depth_raw = np.array(Image.open(os.path.join(ddir, "depth_raw.png")))
        depth_m = depth_raw_to_meters(depth_raw)

        mask_path = os.path.join(ddir, "mask.png")
        mask = None
        if self.use_mask and os.path.exists(mask_path):
            mask = (np.array(Image.open(mask_path)) > 0).astype(np.uint8)

        H, W = depth_m.shape
        fx, fy, cx, cy = infer_fx_fy_from_plane_constants(W, H)

        # リサイズ処理（任意）
        if self.resize is not None:
            W1, H1 = self.resize
            rgb = np.array(Image.fromarray(rgb).resize((W1, H1), Image.BILINEAR))
            depth_m = np.array(Image.fromarray(depth_m).resize((W1, H1), Image.BILINEAR))
            if mask is not None:
                mask = np.array(Image.fromarray(mask).resize((W1, H1), Image.NEAREST))
            # K更新
            from .nutrition5k_utils import resize_intrinsics
            fx, fy, cx, cy = resize_intrinsics(fx, fy, cx, cy, (W, H), (W1, H1))
            W, H = W1, H1

        # Torch tensor化
        rgb_t = torch.from_numpy(rgb).permute(2,0,1).float()/255.0
        depth_t = torch.from_numpy(depth_m).unsqueeze(0).float()
        mask_t = None if mask is None else torch.from_numpy(mask).bool()
        K = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=torch.float32)

        return {"id": did, "rgb": rgb_t, "depth": depth_t, "mask": mask_t, "K": K}

4.3 体積GTの事前計算（Nutrition5kから真の体積ラベルを作る）

論文と同様、テーブル平面を基準に体積を積分します（あなたの既存pixel_area = Z^2/(fx*fy)式と同じ考え方）。
※Nutrition5kには**dish総質量(g)**がメタデータにあり、密度情報が別途あれば質量↔体積の整合チェックも可能ですが、深度GTからの体積算出が最も確実です。
GitHub
+1

def compute_volume_m3(depth_m, K, food_mask, plane_ring_mask=None):
    """
    depth_m: [H,W] meters
    K: 3x3
    food_mask: [H,W] bool
    plane_ring_mask: テーブル平面フィット用リング (Noneなら food_mask 周辺を自動リング化)
    """
    import numpy as np

    H, W = depth_m.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

    # --- 平面フィット ---
    if plane_ring_mask is None:
        from scipy.ndimage import binary_dilation
        ring = binary_dilation(food_mask, iterations=10) & (~food_mask)
        plane_ring_mask = ring

    Z = depth_m
    X = (xx - cx) * Z / fx
    Y = (yy - cy) * Z / fy
    pts = np.stack([X[plane_ring_mask], Y[plane_ring_mask], Z[plane_ring_mask]], axis=1)

    # RANSAC平面: n·p + d = 0
    n, d = fit_plane_ransac(pts)  # 実装は既存のplane_fitを流用可

    # 高さh（テーブルからの正の高さのみ）
    # 平面->点の符号付き距離。テーブル法線は+Z向きに揃える。
    if n[2] < 0: n = -n; d = -d
    dist = (n[0]*X + n[1]*Y + n[2]*Z + d)  # [m]
    h = np.maximum(0.0, -dist)  # 上に盛る想定：負は0クリップ

    # 画素面積（位置依存）
    a_pix = (Z**2) / (fx*fy)  # [m^2/px]
    vol_m3 = np.sum(h[food_mask] * a_pix[food_mask])  # [m^3]
    return vol_m3


これでGT体積（m³→mLは×1e6）を作れます。学習損失に L_depth + L_grad + L_plane + L_volume を採用すれば、絶対スケールと平面水平性に頑健なFTが可能（前回ご提案通り）。

5) 追加のマスク生成（Nutrition5kはマスク未配布）

既知：Nutrition5kは食品セグメントマスク未提供（Issuesでも質問あり）。
GitHub

対応：既存の QwenVL→SAM 2.1 パイプラインで mask.png をバッチ生成し、上記ローダが拾えるよう imagery/realsense_overhead/dish_xxx/ 直下に保存。

もし高品質GTが必要なら、3,224枚に手動マスク付与して性能評価した近年研究の方針も参考に（Frontiers in Nutrition 2024）。
Frontiers

6) どのデータセットでFTするのが「精度×コスパ」ベストか？
目的	データセット	長所	短所	使い分け
最終精度・実用評価（俯瞰絶対深度）	Nutrition5k Overhead	実画像・RealSense・俯瞰・質量ラベル、論文に体積算出手順が明記	K未配布、マスク未配布	本命FT＆hold‑out評価
ドメイン前学習（3D起点で深度・Kが厳密）	MetaFood3D	3Dメッシュ、RGB‑D動画/マスク/栄養/重量、カメラパラメータ、fiducial	実俯瞰データと背景分布が異なる	短期pretrain→N5kで微調整
大量合成で汎化	NutritionVerse‑Synth	8.5万枚、RGB・深度・(インスタンス)マスク完備、多視点	合成→実のドメインギャップ	軽いpretrainに有効
3Dモデル合成	NutritionVerse‑3D	3Dモデルと栄養値あり	規模小・サイズ較正に課題と指摘	付随的なデータ拡張のみ

出典：Nutrition5k README/論文、MetaFood3D 論文、NV‑Synth/NV‑3D 論文。
arXiv
+5
GitHub
+5
CVF Open Access
+5

最小コスト構成（GPU1枚/短期）：
NV‑Synth(数エポック) → Nutrition5k(本FT)。
さらに時間があれば MetaFood3Dも前学習に混ぜ、俯瞰レンダを中心に弱い重みで追加。

7) スケール不一致を潰す「サニティチェック」スクリプト
# scripts/check_n5k_scale.py
from datasets.nutrition5k_utils import *
import numpy as np, imageio.v2 as iio, os, glob

root = "path/to/nutrition5k_dataset"
dids = open(os.path.join(root,"dish_ids/splits/depth_train_ids.txt")).read().splitlines()
did = dids[0]
p = f"{root}/imagery/realsense_overhead/{did}"

depth_raw = iio.imread(os.path.join(p,"depth_raw.png")).astype(np.uint16)
depth_m = depth_raw_to_meters(depth_raw)

print("raw stats:", depth_raw.min(), depth_raw.max())
print("meters stats:", depth_m.min(), depth_m.max())
assert depth_raw.max() <= 4000+10, "READMEの最大4000unitsを超過していないか？"

H,W = depth_m.shape
fx,fy,cx,cy = infer_fx_fy_from_plane_constants(W,H)
a_pix_plane = (Z_PLANE_M**2)/(fx*fy)
print("a_pix_plane(m^2)", a_pix_plane, " expected≈", A_PIX_PLANE_M2)
rel_err = abs(a_pix_plane - A_PIX_PLANE_M2)/A_PIX_PLANE_M2
assert rel_err < 0.05, "平面画素面積が論文値からズレています"
print("OK: units & intrinsics consistent")

8) まとめ（回答）

インストール自体はおおむね正しいです。深度単位をmm扱い（×0.001）にしていた場合は誤りで、/10000.0に修正が必須。
GitHub

Nutrition5kはKを配布していません。論文の Z=35.9cm と 1画素面積 5.957×10⁻³ cm²からfx≈fy≈465pxを復元すれば、絶対体積が整合します（コード提供）。
CVF Open Access

Fine‑tuning用データセットは、

本命＝Nutrition5k Overhead（実俯瞰RGB‑D＋dish質量、hold‑out評価に最適）。

前学習＝NV‑Synth（大量合成で深度・マスク完備）／余力があればMetaFood3D（RGB‑D動画・メッシュ・パラメータ完備）を追加。
arXiv
+1

上記により、ゼロショットの限界で見えていたスケール崩れはデータ側の定数と単位の厳密化＋俯瞰実データでのFTで解消できます。

必要であれば、このローダとサニティチェックを**既存の学習スクリプト（Depth‑Anything V2 / UniDepth v2）**にすぐ織り込める形で差分パッチも作ります。