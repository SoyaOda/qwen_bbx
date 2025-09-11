Depth-Anything V2（メートル対応版）による体積推定テスト実装計画
背景と目的

既存のシステムでは、Qwen-VLで食品領域を検出し、SAM2.1で各料理のマスクを生成した後、UniDepth v2による深度推定を経て体積を算出していました
GitHub
GitHub
。しかし、UniDepth v2は絶対尺度（メートル単位）の深度推定に対応しておらず、推定されたカメラ内部パラメータ（特に焦点距離）が実際より約10倍小さい問題がありました
GitHub
。その結果、補正なしでは体積が極端に過大（例: 46L）になるため、データセット（Nutrition5k）に基づき経験的に約10.5倍のK_scale補正を適用する必要がありました
GitHub
。この問題に対処するため、AppleのDepth Proを用いると、LiDARによる正確な絶対距離の取得と高精度な内部パラメータの提供によりK_scale補正が不要となり、現実的な体積推定が可能になりました
GitHub
。

以上を踏まえ、本計画ではDepth-Anything V2のメートルスケール対応版を導入し、追加のファインチューニングなしで料理ごとの体積推定をテストします。Depth-Anything V2のMetricモデルは合成データセットで絶対尺度の深度学習が行われており
huggingface.co
、Depth Proと同様に推定深度がメートル単位で出力されるためK_scale補正が不要と期待できます
GitHub
huggingface.co
。これにより、UniDepth v2で問題となったスケール補正なしで体積算出が可能か検証することが目的です。

テストデータの準備

評価には、既存のtest_imagesディレクトリ内の画像と対応するマスクを使用します（例: test_images/train_00000.jpgおよび対応するマスクファイル群）
GitHub
。各画像にはSAM2.1で抽出済みの複数の食品マスクが存在し、ファイル名に料理名が含まれています（例: train_00000_det00_rice_bplus.png 等）
GitHub
。テストではこれら同一の画像・マスクセットを使用し、Depth-Anything V2 (Metric) を用いた場合の体積推定結果を比較検証します。

また、体積算出にはカメラの内部パラメータ行列K（焦点距離や光学中心）が必要です。Nutrition5kデータセット同様に各画像のK行列が既知で与えられることを前提とします。実際のiPhoneで撮影された画像であれば、撮影データから画像ごとのキャリブレーション（焦点距離やセンサーサイズなど）を取得できます。例えば、EXIF情報のFocalLengthやFocalLengthIn35mmFormatから推定するか、Apple ARKitが提供するキャリブレーション情報を利用します。テスト実装では簡略のため、想定される典型的な内部パラメータをハードコードするか、画像解像度と想定視野角から計算します（下記コード例では仮の値を使用）。なお、Depth-Anything V2 (Metric) はモデル自体が絶対尺度を出力しますが、ピクセル毎の体積計算には対応するK行列が不可欠である点に注意が必要です。

Depth-Anything V2 (Metric) モデルのセットアップ

まず、Depth-Anything V2のメートル対応版モデルを利用できるよう準備します。最新のTransformersライブラリではDepth-Anything V2が統合されており、Hugging Faceのリポジトリから直接モデルをロード可能です
huggingface.co
。例えば屋内環境向け大型モデル（Hypersimデータセットで学習済みのLargeモデル）を使用します。具体的な手順:

Depth-Anything V2のコードと依存ライブラリをインストールします（必要に応じてGitHubリポジトリをcloneしpip install -r requirements.txt）
huggingface.co
。

Transformers経由でモデルをロードする場合、AutoImageProcessorとAutoModelForDepthEstimationを利用します（HuggingFace上のモデル名は "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf" 等）
huggingface.co
。モデルとプロセッサをロードするコード例を以下に示します。

from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Depth-Anything V2 Metric (Indoor Large) のモデルと画像プロセッサをロード
processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")

device = "cuda"  # GPUが利用可能ならCUDAを使用
model = model.to(device).eval()


上記により、Depth-Anything V2 Metricモデルの推論準備が完了します。以降、このmodelとprocessorを用いて入力画像から深度マップを推定します。

深度推定と内部パラメータ取得

次に、各テスト画像についてDepth-Anything V2モデルで深度推定を行います。推定結果の深度マップはメートル単位の絶対距離となります
huggingface.co
。内部パラメータ行列Kは前述の通り既知であるとし、画像解像度に対応する値を使用します。具体的な処理フローは以下の通りです:

画像の読み込み: OpenCVやPILで画像ファイルを読み込みます。読み込んだ画像はモデルに入力できる形式（Tensor）に変換します。TransformersのImageProcessorを使う場合は、processor(images=img, return_tensors="pt")で前処理（リサイズや正規化）を行います。

Depth-Anything V2による深度推論: 前処理した画像テンソルをモデルに入力し、推論を実行します。例えば:

import torch
import numpy as np

# 画像読み込み（PIL形式からprocessorでテンソル化）
image = Image.open(image_path)
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
# 推定された深度マップ（1xHxWテンソル）を取り出し、numpy配列に変換
pred_depth = outputs.predicted_depth  # shape: [batch=1, 1, H_out, W_out]
# 出力深度を元の画像解像度にリサイズ（モデル内部でリサイズされている場合に備え）
orig_h, orig_w = image.size[1], image.size[0]  # PIL: size=(W,H)
pred_depth = torch.nn.functional.interpolate(
    pred_depth, size=(orig_h, orig_w),
    mode="bicubic", align_corners=False
)
depth_map = pred_depth[0, 0].cpu().numpy()  # HxWの深度マップ（単位: m）


Depth-Anything V2 Metricモデルは深度マップをメートル単位で出力するため、そのままdepth_mapの各値がカメラから各点までの距離[m]になります
huggingface.co
。上記コードでは、モデル内部でリサイズやパッチ処理が行われた場合でも体積計算のため元画像と同じ解像度の深度マップに復元しています。

内部パラメータKの取得: 対応する画像のカメラ内部パラメータ行列Kを取得します。Nutrition5kのようにキャリブレーションデータがある場合はそれを利用し、ない場合はEXIF情報や既定値から推定します。例えばiPhoneで4032×3024の写真を撮影した場合、水平視野角60°程度・焦点距離約4mm（35mm換算26mm）であることを踏まえると、ピクセル単位の焦点距離fx, fyはおおよそ2800～3000px前後と推定できます。この仮定に基づき下記コードでは簡易的にKを設定しています（実際の値はデータに合わせ調整してください）:

H, W = depth_map.shape  # 画像（および深度マップ）の高さ・幅
# 仮の焦点距離（ピクセル単位）: ここでは画像幅に対し水平視野約60°となる値を設定
fx = fy = 3000.0  
cx, cy = W / 2.0, H / 2.0  # 主点は画像中心と仮定
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])


※注意: 上記は一例であり、可能であれば実際のキャリブレーション値を使用してください。Depth Pro等ですでに各画像のintrinsicsが得られている場合はそれをそのまま利用するのが望ましいです（Depth Proの出力にはintrinsics行列が含まれます
GitHub
）。本テストではK行列が既知との前提なので、適切に設定されたKを用いることで、深度マップ上のピクセルを実空間の寸法に対応付けます。

平面検出と高さマップの生成

料理の体積を求めるには、テーブル面を基準とした高さマップを計算する必要があります。Depth-Anything V2で得たシーン全体の深度マップから、RANSACを用いてテーブル平面を推定し、その平面から各ピクセルの高さ差を算出します。具体的な処理は以下の通りです:

テーブル平面の推定: 深度マップと内部パラメータK、そして画像中の食品領域マスクを入力として平面推定を行います。既存実装のestimate_plane_from_depth関数を再利用し、RANSACによりテーブルと思われる平面を検出します
GitHub
。この際、食品が載っているテーブル表面のみを平面とみなすため、マスク領域を除外した周辺ピクセル（マージン40px程度）を平面検出に使うようにします
GitHub
。例えば以下のように呼び出します:

from src.plane_fit import estimate_plane_from_depth

# masks: 前段で読み込んだ食品マスクのブール配列リスト
plane_normal, plane_distance, points_xyz = estimate_plane_from_depth(
    depth_map, K, masks,
    margin_px=40,
    dist_th=0.006,    # 平面とみなす距離閾値 約6mm
    max_iters=2000    # RANSAC試行回数
)


ここでplane_normalとplane_distanceは推定されたテーブル平面の法線ベクトルと原点から平面までの距離（メートル）です。points_xyzは深度マップを点群に変換したもの（カメラ座標系）で、後続の処理で使用します。

高さマップの計算: 推定した平面に対し、各ピクセルが平面からどれだけ上方にあるか（=料理の高さ）を計算します。既存のheight_map_from_plane関数に点群データと平面情報を渡すことで、高さマップを取得できます
GitHub
。なお、テーブル面より下側の高さは不要なので、負の高さは0にクリップします（clip_negative=True）:

from src.volume_estimator import height_map_from_plane

# 平面からの高さマップを計算（points_xyzはdepth_mapと同サイズの3次元点群）
height_map = height_map_from_plane(points_xyz, plane_normal, plane_distance, clip_negative=True)


これにより、height_mapは画像内各ピクセルについてテーブル面からの高さ（m）を表すマップになります。テーブル上の食品領域では正の値、それ以外（テーブル面上または下）は0となります。

ピクセル面積マップの計算: 深度マップと内部パラメータKから、各ピクセルが実空間でカバーする面積を計算します。ピクセルの実面積は深度Zに依存し、式: $a_{\text{pix}} = \frac{Z^2}{f_x \cdot f_y}$ で与えられます
GitHub
。既存のpixel_area_map関数を使えば、この計算を各ピクセルについて行いマップを得られます
GitHub
:

from src.volume_estimator import pixel_area_map

area_map = pixel_area_map(depth_map, K)


ここでarea_map[y,x]は深度マップ上の各点が実世界で表す平面上の面積（平方メートル）です。例えばカメラに近いほど1ピクセル当たりの面積は小さく、遠いほど大きくなります。

料理ごとの体積計算と結果出力

最後に、各料理マスク領域について体積を数値化します。手順は以下の通りです:

体積の数値積分: 高さマップとピクセル面積マップを組み合わせて、各領域の体積を計算します。既存のintegrate_volume関数を利用し、食品マスク内のピクセルについて高さ [m] × 面積 [m^2] を積分します
GitHub
。Depth-Anything V2モデルは信頼度マップを出力しないため、Confidence重み付けはオフにします（conf=None, use_conf_weight=False）。体積は内部で立方メートルからmLへの単位変換（×1e6倍）が行われ、結果のディクショナリ中"volume_mL"として取得できます
GitHub
。

from src.volume_estimator import integrate_volume

total_volume = 0.0
for mask, label in zip(masks, labels):
    vol_result = integrate_volume(height_map, area_map, mask,
                                  conf=None, use_conf_weight=False)
    volume_mL = vol_result["volume_mL"]      # マスク領域の体積 [mL]
    height_mean = vol_result["height_mean_mm"]  # 平均高さ [mm]
    height_max = vol_result["height_max_mm"]    # 最大高さ [mm]
    total_volume += volume_mL
    print(f"{label:>30}: {volume_mL:7.1f} mL  (平均高さ: {height_mean:.1f} mm, 最大高さ: {height_max:.1f} mm)")
print(f"\n合計体積: {total_volume:.1f} mL")


上記ループにより、画像内の各料理について体積（mL単位）と高さ情報が表示されます。体積算出は、Depth-Anything V2の絶対深度を用いているためスケール補正不要で直接的に行われます。これは前述のDepth Pro使用時と同様の利点であり
GitHub
、出力される体積値は現実に即した数量になると期待できます。

結果の評価・出力形式: 得られた体積を元に簡易的な評価を行います。例えば、常識的な料理の体積範囲100～1500 mLに入っているかをチェックし、範囲内なら「✓適切」、極端に小さい場合「⚠小さすぎ」、大きすぎる場合「⚠大きすぎる可能性」などのマークを付与します
GitHub
。また、全料理の合計体積についても合計値と適切性を表示します
GitHub
。これらは既存実装に倣った出力形式で、下記はその一例です:

# 合計体積の適切性チェック
if 100 <= total_volume <= 1500:
    print("→ ✓ 妥当な範囲（100-1500mL）")
elif total_volume < 100:
    print("→ ⚠ 小さすぎる可能性")
else:
    print("→ ⚠ 大きすぎる可能性")


実行結果の形式は、Depth Proテスト時と同様にコンソール出力ベースで構いません。各画像について個別に上記の詳細を表示し、複数画像を処理する場合は最後に統計サマリ（体積の範囲・平均・中央値や、料理カテゴリごとの平均体積など）も算出できます
GitHub
GitHub
。

実装コード例

以上の計画に基づき、テストスクリプト（例: test_depthanything_metric.py）の主要部分コードを示します。既存のtest_depthpro_with_masks.py等と類似した構成で、異なる部分はDepth-Anything V2モデルの読み込み・推論部分と、K_scale補正が不要になった点です:

import os, sys
import numpy as np
import cv2
import torch
from PIL import Image
# Depth-Anything V2 Metricモデルのロード
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

from src.plane_fit import estimate_plane_from_depth
from src.volume_estimator import height_map_from_plane, pixel_area_map, integrate_volume

def infer_depth_anything(image_path):
    """Depth-Anything V2 Metricで画像から深度マップ推定"""
    # 画像読み込み
    img = Image.open(image_path)
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    pred_depth = outputs.predicted_depth  # Tensor [1,1,H_out,W_out]
    # 深度マップを元画像サイズにリサイズ
    orig_h, orig_w = img.size[1], img.size[0]
    pred_depth = torch.nn.functional.interpolate(
        pred_depth, size=(orig_h, orig_w),
        mode="bicubic", align_corners=False
    )
    depth_map = pred_depth[0, 0].cpu().numpy()  # (H, W) in meters
    return depth_map

def test_depthanything_volume(image_path, mask_paths, K):
    """Depth-Anything V2 Metricモデルで深度推定し体積計算"""
    img_name = os.path.basename(image_path)
    print(f"\n{'='*70}\n画像: {img_name}\n{'='*70}")
    # 深度推定
    print("Depth-Anything V2 (Metric)で深度推定中...")
    depth = infer_depth_anything(image_path)
    H, W = depth.shape
    print(f"深度範囲: {depth.min():.2f} - {depth.max():.2f} m")
    print(f"画像サイズ: {W}x{H}, 内部パラメータ fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    # マスク読み込み
    masks = []
    labels = []
    for m_path in mask_paths:
        if os.path.exists(m_path):
            mask_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            mask = mask_img > 127
            masks.append(mask)
            # ファイル名から料理名を抽出（例: train_00000_det00_rice_bplus.png -> "rice"）
            parts = os.path.basename(m_path).split('_')
            if len(parts) > 3:
                food_name = '_'.join(parts[2:-1])
            else:
                food_name = "food"
            labels.append(food_name)
    if not masks:
        print("マスクが見つかりません")
        return None
    print(f"{len(masks)}個のマスクを読み込みました")
    # テーブル平面推定
    print("テーブル平面を推定中...")
    n, d, points_xyz = estimate_plane_from_depth(depth, K, masks,
                                                 margin_px=40, dist_th=0.006, max_iters=2000)
    # 高さマップ・ピクセル面積マップ計算
    height_map = height_map_from_plane(points_xyz, n, d, clip_negative=True)
    area_map = pixel_area_map(depth, K)
    # マスクごとの体積計算
    total_vol = 0.0
    for mask, label in zip(masks, labels):
        vol = integrate_volume(height_map, area_map, mask, conf=None, use_conf_weight=False)
        vol_ml = vol["volume_mL"]
        h_mean = vol["height_mean_mm"]
        h_max = vol["height_max_mm"]
        total_vol += vol_ml
        # 結果表示
        status = "✓" if 10 <= vol_ml <= 1000 else ("⚠小" if vol_ml < 10 else "⚠大" if vol_ml > 1500 else "△")
        print(f"  {label:30s}: {vol_ml:7.1f} mL  (高さ: 平均{h_mean:.1f} mm, 最大{h_max:.1f} mm) {status}")
    # 合計体積と評価表示
    print(f"合計体積: {total_vol:.1f} mL", end="")
    if total_vol > 1000:
        print(f"  ({total_vol/1000:.2f} L)")
    else:
        print("")
    if 100 <= total_vol <= 1500:
        print("→ ✓ 妥当な範囲（100-1500 mL）")
    elif total_vol < 100:
        print("→ ⚠ 小さすぎる可能性")
    else:
        print("→ ⚠ 大きすぎる可能性")
    return total_vol

# テスト実行例（既存のtest_casesリストを利用）
test_cases = [
    ("test_images/train_00000.jpg", [
        "outputs/sam2/masks/train_00000_det00_rice_bplus.png",
        "outputs/sam2/masks/train_00000_det01_snow_peas_bplus.png",
        "outputs/sam2/masks/train_00000_det02_chicken_with_sauce_bplus.png"
    ]),
    # 必要に応じて他の画像も追加
]
# 仮の内部パラメータ行列（各画像に対して設定）
# ここでは全画像共通で近似値を使用しているが、本来は画像ごとに適切な値を用いる
example_fx = example_fy = 3000.0
for img_path, masks in test_cases:
    # 画像解像度取得（ここでは推定前のdepth_mapからではなく、cv2等で読んで取得してもよい）
    img = cv2.imread(img_path)
    H, W, _ = img.shape
    cx, cy = W/2.0, H/2.0
    K = np.array([[example_fx, 0, cx],
                  [0, example_fy, cy],
                  [0, 0, 1]])
    test_depthanything_volume(img_path, masks, K)


上記コードでは、Depth-Anything V2 Metricモデルを用いて各画像の深度推定と体積計測を行っています。要点を整理すると:

モデル推論部分: infer_depth_anything関数内でTransformersのモデルを使い、絶対距離の深度マップを得ています
huggingface.co
。

K行列の扱い: 簡易的に仮定した値を用いていますが、実際には画像に対応する正確なKを使用してください。Depth Proから取得済みの値がある場合はそれを流用できます
GitHub
。

K_scale補正: コード中に登場しないことに注目してください。Depth-Anything V2 (Metric)は絶対スケールで深度推定を行うため、UniDepth v2で必要だったK_scale係数の適用を省略しています
GitHub
。これはApple Depth Proを用いた場合と同様の利点です。

体積計算: integrate_volumeで得られたvolume_mLをそのまま報告し、単位は[mL]で統一しています（1立方メートル=1000L=1e6mLで換算）
GitHub
。ユーザの指定により、必要であればcm³表記も可能ですが、mLは数値が直感的で扱いやすいため本実装ではmLとしています。

以上、Depth-Anything V2 Metricモデルを組み込んだ体積推定テストの実装計画およびコード例を示しました。実行することで、各料理の体積がコンソールに表示されます。この結果をUniDepth v2やDepth Proを用いた場合と比較検証し、Depth-Anything V2 (Metric)モデルの有効性（スケールの信頼性や推定精度向上
GitHub
など）を評価できます。