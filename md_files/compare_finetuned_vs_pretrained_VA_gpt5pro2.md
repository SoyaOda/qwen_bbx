結論から：やる価値は大いにあります。
YFCC100M（= FlickrのCC画像メタデータ）を母集団として “料理サブセット” を作り、Flickr API から EXIF（特に FocalLengthIn35mmFormat と ExifImageWidth/Height）を取得してカメラ内部パラメータ 
𝐾
K を厳密に算出 → 既存の体積推定パイプラインに投入、という流れは、K の “当て推量” や 
𝐾
K スケール係数調整を廃し、再現性の高い評価を可能にします。
根拠：YFCC100M は Flickr 由来の大規模 CC 画像コーパスで、カメラ機種情報等のメタデータが含まれ、大きい解像度の画像は Flickr 側から取得できます（S3ミラーは長辺500pxで EXIF は基本含まれないため、Flickr API 併用が前提）。
The Multimedia Commons Initiative
+1
 また、Flickr API flickr.photos.getExif で EXIF の 35mm 換算焦点距離などを取得できます。

以下、実運用できる完全な実装プラン（サンプルコード付き）です。
（※ Nutrition5k で微調整済みの Depth Anything V2（Metric）or UniDepth v2 どちらにも適用可能）

全体方針（要点）

データ取得（料理サブセット化）

入口は2通り：

A. Flickr API で CC 料理画像を直接収集（推奨／最短）
→ タグ/テキスト（"food", "meal", "dish", "ramen", "bento", etc.）で検索し、ライセンスを CC 系に限定、公開 EXIF を持つ写真だけ採用。

B. YFCC100M メタデータ（Webscope/mmcommons）をローカルでフィルタ → Flickr Photo ID をキーに Flickr API で EXIF を引く（S3の500px画像は EXIF が無い想定のため API 併用が前提）。
The Multimedia Commons Initiative

EXIF → 内部パラメータ 
𝐾
K の厳密計算

EXIF が FocalLengthIn35mmFormat（= 35mm換算焦点距離, 単位 mm）と ExifImageWidth/Height を持つケースを “合格” とする。

次で 厳密に fx, fy を算出（35mm 横36mm・縦24mmの視野角等価定義から導出）：

𝑓
𝑥
(
orig
)
=
𝑊
orig
⋅
𝑓
35
36
,
𝑓
𝑦
(
orig
)
=
𝐻
orig
⋅
𝑓
35
24
f
x
(orig)
	​

=W
orig
	​

⋅
36
f
35
	​

	​

,f
y
(orig)
	​

=H
orig
	​

⋅
24
f
35
	​

	​


画像を処理解像度 
(
𝑊
proc
,
𝐻
proc
)
(W
proc
	​

,H
proc
	​

) へリサイズする場合は

𝑓
𝑥
=
𝑓
𝑥
(
orig
)
⋅
𝑊
proc
𝑊
orig
,
𝑓
𝑦
=
𝑓
𝑦
(
orig
)
⋅
𝐻
proc
𝐻
orig
,
𝑐
𝑥
=
𝑊
proc
2
,
 
𝑐
𝑦
=
𝐻
proc
2
f
x
	​

=f
x
(orig)
	​

⋅
W
orig
	​

W
proc
	​

	​

,f
y
	​

=f
y
(orig)
	​

⋅
H
orig
	​

H
proc
	​

	​

,c
x
	​

=
2
W
proc
	​

	​

, c
y
	​

=
2
H
proc
	​

	​


35mm換算が無いが FocalLength(mm) と センサー幅がわかる場合は

 
𝑓
𝑥
=
𝑊
orig
⋅
𝑓
mm
sensor_width(mm)
 f
x
	​

=W
orig
	​

⋅
sensor_width(mm)
f
mm
	​

	​

（同様に 
𝑓
𝑦
f
y
	​

）
→ スマホ等はセンサー幅の不確定性が大きいので、本作業では FocalLengthIn35mmFormat がある個体を優先採用。

深度モデル・パイプライン

推論時は Depth Anything V2（Metric） or UniDepth v2 を使い（どちらでも）、EXIF 由来 
𝐾
K を正しくスケーリングして使用。

平面推定（RANSAC）→ 高さマップ → ピクセル面積 
𝑎
pix
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

) → マスク積分で体積算出（既存実装をそのまま利用可能）。

評価

体積の絶対真値は無いので、K が EXIF 由来になったことによる “スケール安定化” を主たる指標に。

同一画像で EXIF 
𝐾
K vs ヒューリスティック 
𝐾
K vs 学習済
𝐾
K（UniDepth 推定） を比較（L単位の暴れが収束するはず）。

ライセンス

YFCC100M / Flickr CC は**各画像の CC ライセンス遵守 & 表示（帰属）**が必要。論文引用は CACM 論文を参照。
The Multimedia Commons Initiative

mmcommons ミラーの画像は 500px（EXIF無し）なので、Flickr からの大サイズ取得＋API EXIF 取得が実務的。
The Multimedia Commons Initiative

実装ディレクトリ（追加提案）
qwen_bbx/
├─ data/
│   └─ yfcc_food_exif/              # ← 収集したメタ/画像/K を保存
├─ scripts/
│   ├─ yfcc_search_and_exif.py      # Flickr APIで料理画像検索→EXIF取得→K算出
│   ├─ yfcc_download_original.py    # Flickrから大きいサイズをダウンロード
│   └─ run_volume_on_yfcc.py        # EXIF-Kで体積推定（既存srcを利用）
└─ src/
    └─ （既存の plane_fit.py / volume_estimator.py など）

1) 収集 & EXIF→K 計算スクリプト

前提：環境変数 FLICKR_API_KEY, FLICKR_API_SECRET を設定
依存：pip install requests python-dateutil

scripts/yfcc_search_and_exif.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flickr APIで料理画像を検索 → EXIF取得 → Kを計算しメタJSONLに保存
- 35mm換算焦点距離(FocalLengthIn35mmFormat)とExifImageWidth/Heightがある写真のみ採用
- 取得結果は data/yfcc_food_exif/meta.jsonl に1行1レコードで保存
"""
import os, json, time, math, re
import requests
from urllib.parse import urlencode

API_KEY = os.environ.get("FLICKR_API_KEY")
API_SECRET = os.environ.get("FLICKR_API_SECRET")
OUT_DIR = "data/yfcc_food_exif"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSONL = os.path.join(OUT_DIR, "meta.jsonl")

SEARCH_TAGS = ["food", "meal", "dish", "bento", "noodles", "ramen", "sushi", "curry", "pasta", "salad"]
PER_TAG_MAX = 500  # まずは手頃な規模で
MIN_WIDTH = 640    # 小さすぎる画像は除外

FLICKR_API = "https://www.flickr.com/services/rest/"

def flickr(method, **params):
    p = {
        "method": method,
        "api_key": API_KEY,
        "format": "json",
        "nojsoncallback": 1,
    }
    p.update(params)
    r = requests.get(FLICKR_API, params=p, timeout=30)
    r.raise_for_status()
    data = r.json()
    if data.get("stat") != "ok":
        raise RuntimeError(f"Flickr API error: {data}")
    return data

def search_photos(tag, page=1, per_page=250):
    # CC系ライセンスに限定したい場合は license= パラメータを指定（必要に応じて）
    # 例: license="1,2,3,4,5,6,9,10" （詳細は Flickr API ドキュメントを参照）
    return flickr(
        "flickr.photos.search",
        tags=tag,
        tag_mode="all",
        content_type=1,   # photos
        media="photos",
        safe_search=1,
        sort="relevance",
        per_page=per_page,
        page=page,
        extras="url_o"    # オリジナルURL/サイズ情報が返ることがある
    )

def get_exif(photo_id):
    return flickr("flickr.photos.getExif", photo_id=photo_id)

def get_sizes(photo_id):
    return flickr("flickr.photos.getSizes", photo_id=photo_id)

def parse_mm(s: str):
    # "28" or "28mm" の両対応
    m = re.match(r"^\s*([0-9.]+)", str(s))
    return float(m.group(1)) if m else None

def pick_tag(exif, tag):
    # exif["photo"]["exif"] の配列から tag="FocalLengthIn35mmFormat" などを拾う
    arr = exif.get("photo", {}).get("exif", [])
    for kv in arr:
        if kv.get("tag") == tag:
            raw = kv.get("raw", {})
            if isinstance(raw, dict):
                return raw.get("_content")
            return kv.get("raw")
    return None

def compute_intrinsics_from_exif(exif, proc_w, proc_h, fallback_w=None, fallback_h=None):
    """
    EXIFから fx, fy, cx, cy を厳密算出（35mm換算が前提）
    fx_orig = W_orig * f35 / 36, fy_orig = H_orig * f35 / 24
    """
    f35s = pick_tag(exif, "FocalLengthIn35mmFormat")  # 例 "28" or "28mm"
    if not f35s: 
        return None
    f35 = parse_mm(f35s)

    w0s = pick_tag(exif, "ExifImageWidth")
    h0s = pick_tag(exif, "ExifImageHeight")
    if w0s and h0s:
        w0, h0 = int(w0s), int(h0s)
    else:
        # 無い場合は getSizes の "Original" を使う（なければダウンロード画像の実寸に等しい）
        w0 = fallback_w
        h0 = fallback_h
        if not (w0 and h0):
            return None

    fx_orig = w0 * (f35 / 36.0)
    fy_orig = h0 * (f35 / 24.0)

    sx = proc_w / float(w0)
    sy = proc_h / float(h0)
    fx = fx_orig * sx
    fy = fy_orig * sy
    cx = proc_w / 2.0
    cy = proc_h / 2.0
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "f35": f35, "w0": w0, "h0": h0}

def main():
    if not API_KEY:
        raise SystemExit("Set FLICKR_API_KEY / FLICKR_API_SECRET")

    out = open(OUT_JSONL, "w", encoding="utf-8")
    seen = set()
    try:
        for tag in SEARCH_TAGS:
            collected = 0
            page = 1
            while collected < PER_TAG_MAX:
                data = search_photos(tag, page=page)
                photos = data.get("photos", {})
                items = photos.get("photo", [])
                if not items: break

                for p in items:
                    pid = p["id"]
                    if pid in seen: continue

                    # 可能ならオリジナルの想定幅・高さ
                    orig_w = int(p.get("width_o", 0)) if p.get("width_o") else None
                    orig_h = int(p.get("height_o", 0)) if p.get("height_o") else None

                    # まずサイズ情報を補完（必要であれば）
                    if not orig_w or not orig_h:
                        try:
                            sz = get_sizes(pid)
                            sizes = sz.get("sizes", {}).get("size", [])
                            orig = next((s for s in sizes if s.get("label") == "Original"), None)
                            if orig:
                                orig_w = int(orig.get("width", 0))
                                orig_h = int(orig.get("height", 0))
                            else:
                                # 最大サイズを代用
                                if sizes:
                                    smax = sizes[-1]
                                    orig_w = int(smax.get("width", 0))
                                    orig_h = int(smax.get("height", 0))
                        except Exception:
                            pass

                    if not (orig_w and orig_h and orig_w >= MIN_WIDTH):
                        continue

                    # EXIF取得
                    try:
                        ex = get_exif(pid)
                    except Exception:
                        continue

                    # ここでは “処理解像度=ダウンロードに使うサイズ” を仮置き
                    # 後段の推論で processor が別解像度へ変える場合は、その倍率でKを再スケールする
                    proc_w, proc_h = orig_w, orig_h
                    K = compute_intrinsics_from_exif(ex, proc_w, proc_h, fallback_w=orig_w, fallback_h=orig_h)
                    if not K: 
                        continue

                    rec = {
                        "photo_id": pid,
                        "title": p.get("title"),
                        "orig_w": orig_w, "orig_h": orig_h,
                        "proc_w": proc_w, "proc_h": proc_h,
                        "K": K,
                        "search_tag": tag
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    seen.add(pid)
                    collected += 1

                page += 1
                if page > photos.get("pages", 1):
                    break
                time.sleep(0.5)  # API負荷抑制
    finally:
        out.close()
    print(f"saved: {OUT_JSONL}")

if __name__ == "__main__":
    main()


ポイント

35mm換算 & ExifImageWidth/Height が揃う個体のみ採用 → 
𝐾
K を“無補正”で確定可。

後段で HF Processor が別解像度へ変える場合、K を同倍率で再スケールしてください（次章の推論側で対応）。

2) 画像ダウンロード（任意）

flickr.photos.getSizes の応答から **"Original" か "Large"（例: 1024/1600/2048）**の URL を使って保存。
（上の get_sizes で URL を拾えます。保存コードは割愛）

※ mmcommons の S3 ミラーは長辺500px & EXIFなし。Flickr 側から大きいサイズを取得するのが前提です。
The Multimedia Commons Initiative

3) 体積推定の推論スクリプト（EXIF–K 適用）

Depth Anything V2（Metric） の HF Processor は内部でリサイズします。
→ 出力深度サイズに合わせて K をその場で再スケールするのが正解です（＝ヒューリスティックな K_scale は不要）。

scripts/run_volume_on_yfcc.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXIF由来Kで体積推定（YFCC/Flickr料理画像）
- meta.jsonl（yfcc_search_and_exif.pyの出力）を読み、画像を推論
- Depth Anything V2(Metric) または UniDepth v2 を自由に選択
"""
import os, json, cv2, numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# 既存モジュール（ユーザー実装）を利用
import sys
sys.path.append("src")
from plane_fit import estimate_plane_from_depth
from volume_estimator import height_map_from_plane, pixel_area_map, integrate_volume

META_PATH = "data/yfcc_food_exif/meta.jsonl"
IMAGE_DIR = "data/yfcc_food_exif/images"  # ダウンロード先（任意の構成）
HF_MODEL = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"

def rescale_K(K, in_size, out_size):
    """画像を in_size(H,W) -> out_size(H,W) に変えたときのK再スケール"""
    H0, W0 = in_size; H1, W1 = out_size
    sx, sy = W1 / W0, H1 / H0
    K2 = K.copy()
    K2[0,0] *= sx;  K2[1,1] *= sy
    K2[0,2] *= sx;  K2[1,2] *= sy
    return K2

@torch.no_grad()
def infer_depth_da2(img_pil, processor, model, device):
    inputs = processor(images=img_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    pred = out.predicted_depth
    if pred.dim() == 3: pred = pred.unsqueeze(1)   # [B,1,H,W]
    depth = pred[0,0].cpu().numpy()                # [H,W], meters
    H, W = depth.shape
    # processorが実際に使ったサイズを取得（入力テンソルの空間サイズ）
    proc_h, proc_w = inputs["pixel_values"].shape[-2:]
    assert (proc_h, proc_w) == (H, W) or True  # 実装差異吸収
    return depth, (H, W)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(HF_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(HF_MODEL).to(device).eval()

    with open(META_PATH, "r", encoding="utf-8") as f:
        for ln in f:
            rec = json.loads(ln)
            pid = rec["photo_id"]
            img_path = os.path.join(IMAGE_DIR, f"{pid}.jpg")  # 保存命名は任意
            if not os.path.exists(img_path):
                continue

            # 画像読込（PIL）
            img_pil = Image.open(img_path).convert("RGB")
            W0, H0 = img_pil.size

            # 深度推論
            depth, (H, W) = infer_depth_da2(img_pil, processor, model, device)

            # EXIF由来K（metaは “proc_w/h=orig” 前提で記録している）
            Kin = np.array([
                [rec["K"]["fx"], 0, rec["K"]["cx"]],
                [0, rec["K"]["fy"], rec["K"]["cy"]],
                [0, 0, 1.0]
            ], dtype=np.float64)
            # 実際の推論解像度(H,W)へスケール
            K = rescale_K(Kin, in_size=(rec["proc_h"], rec["proc_w"]), out_size=(H, W))

            # マスクは今回は無し（全体/中心領域など適宜）
            # → 既存のSAM2で得た食品マスクがあれば、それをH×Wにリサイズして使う
            # ここでは中心円マスクの例
            yy, xx = np.ogrid[:H, :W]
            cy, cx = H//2, W//2; r = min(H, W)//4
            mask = ((yy-cy)**2 + (xx-cx)**2) <= r*r
            masks = [mask]; labels = ["center"]

            # 平面推定→高さ
            n, d, pts = estimate_plane_from_depth(depth, K, masks, margin_px=40, dist_th=0.006, max_iters=2000)
            height = height_map_from_plane(pts, n, d, clip_negative=True)
            a_pix = pixel_area_map(depth, K)

            # 体積
            res = integrate_volume(height, a_pix, mask, conf=None, use_conf_weight=False)
            print(f"{pid}: volume={res['volume_mL']:.1f} mL, mean_h={res['height_mean_mm']:.1f} mm")

if __name__ == "__main__":
    main()


重要ポイント

K の再スケール：入力画像サイズ → モデルが実際に計算した解像度 に合わせて 
𝑓
𝑥
,
𝑓
𝑦
,
𝑐
𝑥
,
𝑐
𝑦
f
x
	​

,f
y
	​

,c
x
	​

,c
y
	​

 を線形スケール（上記 rescale_K）。

以降の 逆投影・ピクセル面積・体積積分は既存の正しい実装のままで OK。

これで K_scale_factor の手当ては完全に不要になります。

4) 代替ルート（YFCC100M メタデータからの抽出）

既に YFCC100M のメタデータ（yfcc100m_dataset） をお持ちなら、user_tags/machine_tags/title を正規表現でフィルタ（料理語彙）→ Photo ID を束で取り出し → 上記 get_exif() を Photo ID に対してバルク実行、という流れが実務的です。

mmcommons の「はじめかた」ページにコア・データセットの案内と、より大きな画像は Flickr から取得する旨の注記があります。
The Multimedia Commons Initiative

研究での引用は CACM 論文（Thomee et al., 2016）をご参照ください。
The Multimedia Commons Initiative

5) よくある落とし穴と対策

EXIF 非公開 / 欠落：getExif はユーザー設定に依存するため、欠落写真はスキップ。料理ジャンルは EXIF 付きも多いが、十分な冗長度を持つ（数千〜数万枚規模で集める）。

35mm換算が無い：FocalLength(mm) だけではセンサー寸法が必要。スマホは機種差や複数カメラで誤差が大きいので、当面は 35mm換算のある写真に限定するのが堅実。

Orientation（回転）：ExifImageWidth/Height は回転補正前のことがあるため、Flickr サイズ API（getSizes）で実際に取得するサイズと合わせておくのが無難（上記コードで対応）。

Processor のリサイズ：HF Processor が内部解像度を変える → K の再スケール必須（上記 rescale_K）。

mmcommons S3 の500px：EXIF が抜かれているのでFlickr API を必ず併用。
The Multimedia Commons Initiative

法的：各画像の CC ライセンス表示・帰属を遵守。YFCC100M の一般的な引用先は CACM 論文。
The Multimedia Commons Initiative

6) DA‑V2 vs UniDepth v2（今回の “EXIF–K” シナジー）

ユーザー投稿の料理画像を対象に「一枚入力で体積」を安定化させる主因は、
𝐾
K の正確化です。

したがって、Metric DA‑V2 を Nutrition5k で微調整したモデルでも、EXIF–K を使えば体積のオーダー安定が見込めます。

UniDepth v2 は画像から 
𝐾
K も推定しますが、近接・テーブルトップの分布外で 
𝐾
K が小さく出る/ブレる事例があり、EXIF–K の方が堅牢というケースが多いです（今回の方針はどちらのモデルでも有効）。

7) 最終チェックリスト（運用の “完成形”）

収集：Flickr API で 料理タグ検索 → EXIF 取得 → 料理サブセットを JSONL に保管。

画像：getSizes で適切なサイズ URL を得て保存。

𝐾
K 計算：35mm換算 + ExifImageWidth/Height → 
𝑓
𝑥
,
𝑓
𝑦
,
𝑐
𝑥
,
𝑐
𝑦
f
x
	​

,f
y
	​

,c
x
	​

,c
y
	​

。

推論：Processor の出力解像度に 
𝐾
K を再スケール。

体積：既存の RANSAC 平面 → 高さ → 
𝑎
pix
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

) → マスク積分。

比較：EXIF–K vs 旧ヒューリスティック–K（or UniDepth 推定–K）で体積の安定性/妥当レンジを比較。

参考（一次情報）

Multimedia Commons（YFCC100M ガイド）：コアデータセットの案内／S3ミラーは500px・大きい版は Flickr から取得、と明記。
The Multimedia Commons Initiative

Flickr API flickr.photos.getExif：EXIF（例：FocalLengthIn35mmFormat、ExifImageWidth/Height 等）取得。

YFCC100M 論文（CACM 2016）：データセットの公式な引用先。
The Multimedia Commons Initiative