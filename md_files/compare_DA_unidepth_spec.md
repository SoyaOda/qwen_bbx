食品体積推定向けDepthモデルFine-tuning実装計画
背景・目的

既存の実装では以下の2パターンを試みましたが、食品ごとの体積予測精度が不十分でした。

パターン1: QwenVLで料理領域を検出し、SAM 2.1でセグメンテーション、UniDepth v2で深度推定 → 体積算出（test_v2_implementation.py, test_with_masks_only.py）。

パターン2: 深度推定をApple Depth Proに置き換え、アルゴリズムを最適化して体積算出（test_depthpro_implementation.py, test_depthpro_optimized.py）。

どちらもドメインギャップ（一般シーン訓練のモデルを食品画像に適用）により大きな誤差が生じました
GitHub
。例えばDepth Proは食品データで訓練されておらず、体積推定で60倍以上の過大評価をしてしまうケースも報告されています
GitHub
。焦点距離調整や後処理では限界があり
GitHub
、食品画像ドメインに特化したFine-tuningが必要と判断されました
GitHub
。

目的: 食品画像に対して精度良く絶対的な深度マップを推定できるモデルを構築し、そこから各料理の体積を高精度に計算できるようにすることです。そのために、最新の深度推定モデルを食品ドメイン（Nutrition5kデータセット）でFine-tuningし、体積予測精度を向上・比較します。

モデル選定: Depth-Anything V2 と UniDepth v2

Fine-tuning対象として、以下の2つのオープンソース深度推定モデルを使用します
GitHub
GitHub
。

Depth-Anything V2: 2024年公開の最新SOTA単眼深度モデルです
GitHub
。大規模データで事前学習されゼロショット性能が高く、詳細な深度表現が可能です。実装も効率的で推論速度と精度のバランスに優れています
GitHub
。V1に比べ精細で頑健な深度を推定でき、Transformersライブラリ経由でモデル取得が可能です。

UniDepth v2: 単眼からメートルスケールの絶対深度を推定できるモデルです
GitHub
。カメラ内部パラメータ(K)も同時に推定し、画像ごとにスケールを調整できます。オープンソースで改変可能かつ軽量で、食品データでのFine-tuning事例も報告されています
GitHub
。既存コード(src/unidepth_runner_v2.pyなど)で利用しており、統合しやすい利点もあります。

上記2モデルは技術調査の比較でも推奨度が高く（★★★）
GitHub
、食品ドメインでのFine-tuning適性があると評価されています
GitHub
。Depth Proは精度自体は高いものの食品データ不足とブラックボックス性からFine-tuning困難であり、今回採用しません
GitHub
。

データセット準備 (Nutrition5k 他)

主要データセット: Nutrition5kを使用します。これは約5000枚の実世界の料理画像と詳細な栄養情報からなるデータセットです
GitHub
。各料理はGoogle社のリグで撮影されており、RGB画像と対応する深度マップ（RealSenseによる測定）および食品の質量・体積等のラベルが含まれます
openaccess.thecvf.com
openaccess.thecvf.com
。食品ごとの体積ラベル（質量や容積）は既知で、単一視点のRGB-Dペアが取得可能です。この実測データにより、モデルを食品ドメインの絶対距離感に適応させます
GitHub
。

Nutrition5k内でデータ分割を行います。例: 80% (約4000枚)を訓練用、10% (500枚)を検証用、10% (500枚)をホールドアウトのテスト用に割り当てます。ホールドアウトテストで各モデルの体積推定精度を比較評価します（後述）。

追加データ: 必須ではありませんが、精度向上のため以下も検討します。

オープンな別の食品画像データ（例: Food-101
GitHub
）を用いて学習前半で特徴を馴化させる。ただしFood-101は深度や体積情報が無いため、教師信号としては使えません。

合成データ: 必要に応じ、Blender/Unityで料理3Dモデルを用いたレンダリングでRGB-Dペアを生成します
GitHub
。様々な撮影角度(45°/60°/75°/90°)・距離(20–50cm)でレンダリングし、食品ごとの正確な深度マップと体積が得られる合成データを数千枚用意できます。

実測小規模データ: LiDAR付きスマホやKinect等で、既知体積の食品を撮影してRGB-Dデータを収集する方法もあります
GitHub
。例えば50mL, 100mL, 200mL, 300mLの既知容積の食品を複数角度・距離から撮影し、深度GTを取得することも可能です
GitHub
。ただし初期段階ではNutrition5kのみで十分です。

セグメンテーションマスク: 体積計算には各画像で食品ピクセルの領域を知る必要があります。Nutrition5kには食品のバイナリマスクが付属している場合があります（論文では前景/背景の食物セグメンテーションモデルを使用と記述
openaccess.thecvf.com
）。付属していない場合は、QwenVL+SAMを用いて各画像の料理領域マスクをあらかじめ作成・保存します。トレーニング時にはこのマスクを読み込み、食品領域の深度に対して損失計算・体積算出を行います。

データ前処理

前処理のポイント:

単位統一: RealSense深度マップはミリメートル単位のPNG等の場合があります。適切にスケールを読み取り、メートル(m)に変換します（例えば深度PNGの値×0.001でmに）。合成データも同様に単位を統一。

解像度調整: モデル入力解像度に合わせ、画像と深度マップを必要に応じてリサイズします。Depth-Anything V2は任意解像度入力が可能ですが、メモリ削減のため適度なサイズ（例えば長辺512pxや720px程度）に縮小することを検討します。UniDepth v2もViT-L14ベースのため高解像度入力はメモリ負荷が大きく、上限解像度を設定します（例えば640×480程度にダウンサンプリング）。リサイズ時にはアスペクト比を維持し、深度マップはバイリニア補間でスケール。

データオーグメンテーション: 食品画像に偏りがあるため、訓練データに以下の拡張を適用します
GitHub
（検証・テストには適用しません）。

幾何学的変換: ランダムクロップ・回転・スケーリングなどでカメラ視点の微妙な変化をシミュレート
GitHub
。例えば±10度程度の回転や±10-20%のズーム、ランダムなトリミング（ただし食品がフレームアウトしないよう注意）。

色・質感変換: 写真の色調や鮮明さの違いに対応するため、ColorJitter（明度・彩度・コントラスト変化）やGaussianBlur（ぼかし）を適用
GitHub
。食品画像特有の照明変化やカメラ差をロバストにします。

深度特有の拡張: 深度マップにガウスノイズを加える、あるいは平滑化フィルタをかけることでセンサー雑音を再現します
GitHub
。RealSenseの計測ノイズや欠損を学習時に再現し、モデルが過度にGTにフィットしすぎないようにします。

食品ドメイン特有拡張: 料理が皿に載っている前提なので、皿自体の向きをランダムに回転させるなど（トップビューなので回転しても見た目の変化は少ないですが、例えば器の模様などの視点変化になる）
GitHub
。また、マスクのエロージョン/ダイレーションを行い、境界を多少不正確にする
GitHub
。これにより、セグメンテーション誤差に対する頑健性も向上させます。
(実装: mask = random_erosion_dilation(mask, max_px=5) のように±数ピクセル範囲でマスクを膨張・縮小させる)

データセットクラス: 前処理とデータ読み込みを行うPyTorch Datasetを実装します。例えばsrc/depthany/dataset_nutrition5k.pyに以下のようなクラスを用意します。

import os, glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class Nutrition5kDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        # 画像、深度、マスク、体積ラベルのパスリストを作成
        img_dir = os.path.join(root_dir, split, "images")
        depth_dir = os.path.join(root_dir, split, "depths")
        mask_dir = os.path.join(root_dir, split, "masks")
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        self.depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform
        # 体積ラベル（mLなど）の読み込み: ファイル名対応のCSVやJSONから読み込む
        volume_label_path = os.path.join(root_dir, "volumes.csv")
        self.volumes = load_volume_labels(volume_label_path, split)  # 実装は省略

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 画像読み込み
        img = Image.open(self.img_paths[idx]).convert("RGB")
        depth = Image.open(self.depth_paths[idx])
        depth_np = np.array(depth).astype(np.float32)
        depth_np *= 0.001  # 例えば、mm→m変換（適宜データ仕様に合わせる）
        mask = Image.open(self.mask_paths[idx])
        mask_np = (np.array(mask) > 128).astype(np.uint8)  # 2値マスク

        vol = np.float32(self.volumes[idx])  # 体積の真値

        # 変換の適用（画像・深度・マスクに統一的な幾何変換をかける）
        if self.transform:
            img, depth_np, mask_np = self.transform(img, depth_np, mask_np)

        # Tensor変換
        img_t = torchvision.transforms.functional.to_tensor(img)            # [C,H,W], 0-1正規化
        depth_t = torch.from_numpy(depth_np).unsqueeze(0)                   # [1,H,W]
        mask_t = torch.from_numpy(mask_np)                                  # [H,W], 0/1

        return img_t, depth_t, mask_t, torch.tensor(vol, dtype=torch.float32)


上記では、事前にvolumes.csv等に各画像の体積ラベル（mL単位など）を用意し読み込んでいます。Nutrition5kでは質量から容積を算出可能ですが、ここでは深度GTとマスクから基準平面に対する積分で体積を計算し、その値をラベルとすることもできます
openaccess.thecvf.com
（後述）。

体積ラベルの取得: Nutrition5kには各皿の**質量 (grams)や体積 (mL)**が付与されている可能性があります
medium.com
。無い場合は、深度GTから体積を計算してラベルとします。計算方法は文献[1]にならい、上方視点の深度図から食品体積を求める方法を用います
openaccess.thecvf.com
。具体的には:

テーブル面の深度 (Z_plane) を決定: セグメンテーションマスク外の領域、またはマスク周囲のリングから平面フィットし、テーブルの平均高さを算出します。

各画素の体積寄与を計算: 食品領域内の各画素について、テーブル面からの高さ $h = Z_{\text{plane}} - Z_{\text{pred}}(x,y)$ を求めます（上からの俯瞰で食品はテーブルより手前にあるため、この差が食品の高さになります）。

画素当たりの面積 (a_pix) を算出: 深度カメラの内部パラメータを用い、その画素の実世界での面積を算定します。例えば焦点距離$f_x, f_y$および画素座標$(x, y)$、深度$Z$から $a_{\text{pix}}(x,y) = \frac{Z^2}{f_x \cdot f_y}$
GitHub
と求められます（これは各ピクセルがカメラ座標系でどれだけの面積を表すかを示す指標です）。

総体積の積分: 各画素について体積寄与 $v(x,y) = a_{\text{pix}}(x,y) \times h(x,y)$ を計算し、マスク領域内で総和を取ります
GitHub
。これが予測体積$V_{\text{pred}}$となります。同様にGT深度マップから$V_{\text{gt}}$を計算可能です。

上記の計算ではカメラ内部パラメータ($K$行列)が必要です。Nutrition5kでは撮影条件が固定で、例えばカメラとテーブル距離35.9cm, 焦点距離から計算されるピクセル面積$5.957\times10^{-3}$cm²といった値が論文に示されています
openaccess.thecvf.com
。これはカメラの$K$が既知であることを意味します。そこで:

Depth-Anything V2には内部パラメータ推定機構が無いので、**既知の$K$（固定カメラのキャリブレーション値）**を使用します。データセットに記載のFOVや焦点距離から$K$を算出してプログラムにハードコードします。

UniDepth v2は推論時に$K$を予測しますが、学習時には**$K$の真値**（固定値）を与えて誤差を測ることも可能です。今回は、モデルには$K$予測も学習させつつ、体積計算には真の$K$を使用する方針とします（真の$K$を使えば体積誤差の勾配は主に深度に伝搬し、$K$予測誤差も深度経由で矯正されます）。必要なら$K$予測そのものに対して$fx, fy$の差のL2損失を追加し、モデルが実キャリブレーションに近い$K$を出力するよう促します。

損失関数設計とFine-tuning目標

Fine-tuningの目標: モデルが絶対的に正確な深度マップを予測できるようにし、そこから得られる食品体積の誤差を最小化することです。すなわち、単にピクセル単位の誤差を減らすだけでなく、テーブル平面の水平性や食品全体のスケールも正しく推定できるようにします
GitHub
。これにより実用上重要な体積計算の精度向上を図ります。

損失関数: 複数の要素を組み合わせた総合的な損失を設計します
GitHub
GitHub
。各項の狙いは以下の通りです。

深度の絶対誤差 (L_depth): 深度マップの各ピクセル値とGTの誤差を計算します。食品領域のピクセルについて、$L_1$損失または$L_2$損失を適用します
GitHub
（L1の方が外れ値にロバストなため一般的に選択されます）。これによりローカルな深度値の精度を高めます。

実装: L_depth = F.l1_loss(pred_depth[mask], gt_depth[mask])
GitHub
。ここでmaskは食品ピクセルを示すブールマスクです。

深度勾配の一致 (L_grad): 予測深度とGT深度の勾配（梯度）を一致させる損失です
GitHub
。具体的には、画像上で隣接ピクセル間の深度差（エッジの強度）をGTと比較し、L1損失を計算します。これによりモデルがエッジを鋭敏に検出しつつ、平坦な領域ではスムーズな出力を保つようになります。食品ドメインでは器の縁や食品の境界で深度が不連続になるため、勾配損失でその表現力を高めます
GitHub
。

実装: Sobelフィルタ等でpred_depthとgt_depthのx方向・y方向の差分画像を計算し、それらのL1差を取ります（カーネル手計算でもOK）。gradient_loss(pred, gt, mask)として実装し、食品領域内の勾配だけでなく境界周辺も見るようにします
GitHub
。

平面の水平性 (L_plane): テーブル面が水平になるよう促す損失です
GitHub
。マスク外周のリング状領域（食品の周囲のテーブル部分）に対して平面当てはめを行い、その平面法線ベクトル$n$のz成分($n_z$)を取り出します。理想的にはテーブルは真横（水平）なので$n = (0,0,1)$、すなわち$n_z = 1$です
GitHub
。そこで損失をL_plane = 1 - |n_z|と定義し、$n_z$が1に近づく（平面が水平に近づく）ほど損失が減るようにします
GitHub
。これによりモデルがテーブル面の傾きを最小化する方向に調整されます。食品撮影ではカメラを真上から向けることが多いため、現実にも$n_z$は概ね1のはずです。水平面がずれている場合、深度推定の系統誤差（中心部過小評価など）が疑われるため
GitHub
、その是正につながります。

実装: mask_ring = dilate(mask) - mask 等でマスクを数ピクセル膨張させた領域から食品領域を除き、テーブル上の輪郭リングを抽出します。fit_plane(pred_depth, mask_ring)でその領域の点群を平面近似し、法線$n=(n_x,n_y,n_z)$を得ます。損失をL_plane = 1 - abs(n_z)とします
GitHub
。

体積の一致 (L_volume): 予測体積とGT体積の比誤差をとる損失です
GitHub
。食品全体のスケール誤差を直接是正する目的で導入します。体積$V_{\text{pred}}$と$V_{\text{gt}}$の差の相対誤差$|V_{\text{pred}} - V_{\text{gt}}| / V_{\text{gt}}$を損失とし、これを小さくするよう学習させます
GitHub
。相対値にすることで、大容量と小容量で同等の比率精度を狙います（例えば10mL誤差でも、小盛りスープ100mLでは10%のエラー、大皿パスタ1000mLでは1%のエラーとなり、後者は相対的には許容できる）。

実装: 上述した体積計算関数compute_volume(depth, K, mask)を用い、vol_pred = compute_volume(pred_depth, K, mask)で予測体積を算出。GT体積vol_gtはデータセット付属値またはGT深度から事前計算した値を使用。損失はL_volume = abs(vol_pred - vol_gt) / vol_gtとなります
GitHub
。

各損失はスケールが異なるため重み付けを行います。参考値として、深度L1を1.0基準とし、勾配損失に0.1、平面損失に0.05、体積損失に0.1程度を乗じます
GitHub
。したがって総損失は次の形になります:

𝐿
total
=
𝐿
depth
+
0.1
𝐿
grad
+
0.05
𝐿
plane
+
0.1
𝐿
volume
.
L
total
	​

=L
depth
	​

+0.1L
grad
	​

+0.05L
plane
	​

+0.1L
volume
	​

.

(上記重みは経験的な初期値であり、学習の進行を見て調整します。例えば初期エポックで深度L1が大きく減少し勾配が荒れるようなら$L_{\text{grad}}$比重を上げる等)。

備考: もしデータセット内にGT深度がない場合（例えばNutrition5kで深度が使えず体積ラベルのみの場合）は、$L_{\text{depth}}$を除き残りの損失で学習します。その場合$L_{\text{volume}}$がグローバルスケールの唯一の教師になるため、まず合成データ等で粗くスケールを合わせてから微調整する戦略も考えられます。ただしNutrition5kでは深度も利用可能なため、今回は直接$L_{\text{depth}}$で強力に教師信号を与え、$L_{\text{volume}}$は補助的に全体スケールを引き締める役割とします。

実装計画・フォルダ構成

既存コード（src/volume_depthpro.py, src/depth/unidepth.pyなど）と共存させるため、新たに**src/depthany/ディレクトリ**を作成し、この中にFine-tuningと評価に必要なコード一式を実装します。ファイル構成と役割は以下のとおりです。

src/depthany/                 # 新規ディレクトリ（Depth-Anything + UniDepth関連コード）
├── __init__.py
├── dataset_nutrition5k.py    # Nutrition5k用のDataset実装（画像・深度・マスク・体積の読み込み・前処理）
├── model_depth_anything.py   # Depth-Anything V2モデルの読み込み・推論用ラッパー
├── model_unidepth_v2.py      # UniDepth v2モデルの読み込み・推論用ラッパー
├── train_depth_models.py     # Fine-tuning実行スクリプト（学習ループ実装）
└── evaluate_volume.py        # テストデータで各モデルの体積精度を評価するスクリプト


各コンポーネントの詳細実装について順に説明します。

モデルの読み込み (Depth-Anything V2 / UniDepth v2)

Depth-Anything V2: Hugging Face Transformers経由で事前学習モデルをロードします
github.com
github.com
。例えばvit-large版モデルを用いる場合、transformersライブラリから専用クラスを利用できます（最新TransformersにはDepthAnythingV2モデルのクラスが統合されています
github.com
）。

# model_depth_anything.py
import torch
from transformers import DepthAnythingV2Config, DepthAnythingV2Model

def load_depth_anything(model_size="vitl", checkpoint_path=None, device="cuda"):
    # 設定を選択: vit-small/base/large
    config = DepthAnythingV2Config(encoder=model_size)
    model = DepthAnythingV2Model(config)  # モデルインスタンス作成
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    else:
        # HuggingFace Hubから事前学習モデルをロード（例: vit-largeのweights）
        hf_repo = f"DepthAnything/Depth-Anything-V2-{model_size}"  # 仮のリポ名
        model = DepthAnythingV2Model.from_pretrained(hf_repo)
    model.to(device)
    return model


モデルサイズ: 初期は軽量化のためvitb(Base)やvits(Small)版でFine-tuningを開始し、余力があればvitl(Large)版も試します。パラメータ数はSmallで約25M、Baseで97M、Largeで335Mとされています
github.com
。まずは中程度のBaseモデルで学習し、結果次第でLargeにも適用します（小さいモデルの方が学習が安定しやすく、初期検証に適しています）。

前処理: Depth-Anythingモデルは入力画像を正規化して内部に入れる必要があります。Transformers版では内部でinfer_image等の補助関数がありますが、訓練時はバッチ処理するため、自前でforwardを呼び出します。入力テンソルは[B,3,H,W]のピクセル値0-1正規化済みで問題ありません（公式実装ではOpenCVで読み込んだBGR画像を渡していますが
github.com
、PyTorchでは0-1正規化済みTensorをそのまま与え、モデル内部でscaleしてくれる想定です）。

出力: Depth-Anythingモデルのforward出力は深度マップ（おそらく入力と同解像度、またはスケール後リサイズ）です。Transformers版の場合、DepthAnythingV2Modelのforwardメソッドを呼ぶと深度マップTensorを返すと仮定します（もしくは辞書形式）。実際の実装ではmodel(pixel_values=img_tensor).predicted_depthのようなプロパティで得られるかもしれません。必要に応じてドキュメント
huggingface.co
を参照して取得します。

注意: Depth-Anythingは相対深度モデルですが、事前学習である程度スケール感も持っています。Fine-tuningで体積誤差を学習させることで、絶対スケールに調整されることを期待します。学習中は既知のカメラ内部パラメータKを用いて体積損失を計算し、モデルが自動的に出力スケールを合わせるよう誘導します。

UniDepth v2: 公式実装のPyTorch Hubを利用し、事前学習済みモデルをロードします
GitHub
。UniDepthはHuggingFace Hubにモデルウェイトがあるため、それを利用します
github.com
。例えばViT-L/14モデルを使う場合:

# model_unidepth_v2.py
import torch
from unidepth.models import UniDepthV2

def load_unidepth_v2(model_repo="lpiccinelli/unidepth-v2-vitl14", device="cuda"):
    try:
        model = UniDepthV2.from_pretrained(model_repo)
    except Exception as e:
        raise RuntimeError("UniDepthモデルのロードに失敗しました。リポジトリURLや依存関係を確認してください。") from e
    model.to(device)
    return model


上記のUniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")により、HuggingFace上の事前学習済み重みを読み込みます
huggingface.co
。モデルがロードされると、modelは推論モードではmodel.infer(...)を使いますが、訓練では直接forwardを呼び出して出力を取得します。

出力形式: UniDepthのforwardは、辞書{"depth": 深度Tensor, "intrinsics": K行列Tensor, "confidence": 信頼度マップ}等を返す仕様です
GitHub
。訓練ではpred = model(rgb_tensor)で出力を受け取り、pred_depth = pred["depth"], pred_K = pred["intrinsics"]のように取り出します。

カメラ内部パラメータ: pred_Kは$3\times3$行列で、モデルが推定した焦点距離等を含みます
GitHub
GitHub
。Nutrition5kでは真の$K$が分かっているため、必要に応じてpred_Kと真値$K_{\text{gt}}$を比較し、例えば$fx$の比率差$(fx_{\text{pred}}/fx_{\text{gt}} - 1)$にペナルティを課すことも検討します。ただし、メイン損失は深度と体積なので、$K$の誤差は間接的に修正されると期待できます
GitHub
GitHub
。訓練中ログとしてpred_KとK_{\text{gt}}の値を出力し、学習が進むにつれ収束していくか確認します。

モデルの微調整設定: 両モデルとも事前学習済みモデルをベースにファインチューニングします。効率と安定性のため、段階的学習戦略を採ります
GitHub
。

Stage 1: バックボーンを凍結し、最後のデコーダ層やヘッドのみ学習（学習率小さめ）。例えばTransformerエンコーダ部分のパラメータrequires_grad=Falseに設定します
GitHub
。これにより一般物体で学習された大局的な深度認識能力を保持しつつ、最終出力だけを食品用にスケール調整します。

Stage 2: バックボーンの一部を解凍し、全体を微調整します。特に食品領域の深度精度に関わる部分（中間層以降）を学習させます
GitHub
。Optimizerの学習率を若干下げ、微小な勾配で全層を調整します。

（必要に応じてStage 3/4も設定可能: 平面検出や体積項にフォーカスした微調整
GitHub
。しかし本実装ではStage 1/2の2段階で十分とします。）

例えばDepth-Anythingの場合、エンコーダ（ViT部分）をStage 1で固定し、デコーダ（Depthヘッド）のみfine-tune、続いてStage 2でエンコーダも含めfine-tuneという流れです。UniDepthの場合も同様に、Stage 1では主幹ネットワークを固定し$K$予測層や出力層のみ更新→Stage 2で全体更新とします。

トレーニングスクリプト (train_depth_models.py)

このスクリプトで実際の学習ループを実装します。疑似コードでフローを示します。

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from depthany.dataset_nutrition5k import Nutrition5kDataset
from depthany import model_depth_anything, model_unidepth_v2
from depthany.losses import gradient_loss, fit_plane, compute_volume  # 必要に応じ実装

# ハイパーパラメータ
EPOCHS = 10
BATCH_SIZE = 4   # 小さめのバッチ（VRAMに合わせ調整）
lr = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

# データセットの用意
train_data = Nutrition5kDataset(root_dir="data/Nutrition5k", split="train", transform=augmentations)
val_data   = Nutrition5kDataset(root_dir="data/Nutrition5k", split="val")
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

# モデルのロード（Depth-Anything V2とUniDepth v2それぞれ）
model_da = model_depth_anything.load_depth_anything(model_size="vitb", device=device)
model_ud = model_unidepth_v2.load_unidepth_v2(device=device)

# Stage 1: backbone凍結
for param in model_da.encoder.parameters():
    param.requires_grad = False
for param in model_ud.backbone.parameters():
    param.requires_grad = False
# （※ 実際のモデル属性名に合わせてencoder/backbone部分を指定）

# オプティマイザ設定（それぞれ別で学習）
optimizer_da = torch.optim.Adam(filter(lambda p: p.requires_grad, model_da.parameters()), lr=lr)
optimizer_ud = torch.optim.Adam(filter(lambda p: p.requires_grad, model_ud.parameters()), lr=lr)

# 学習ループ（モデルごとに別々に回すのがシンプル。先にDepth-Anythingから）
for epoch in range(EPOCHS):
    model_da.train()
    for imgs, depths, masks, volumes in train_loader:
        imgs, depths, masks, volumes = imgs.to(device), depths.to(device), masks.to(device), volumes.to(device)
        optimizer_da.zero_grad()
        # 順伝播
        pred_depths = model_da(imgs)           # [B,1,H,W] Depth予測
        # Volume計算用に内部パラメータ取得（Depth-Anythingでは固定K使用）
        K = get_intrinsics_matrix()           # Nutrition5k固定のK (deviceに載せる)
        # ロス計算（各項マスク内で計算）
        L_depth = F.l1_loss(pred_depths[masks==1], depths[masks==1])
        L_grad  = gradient_loss(pred_depths, depths, masks)
        plane_n = fit_plane(pred_depths, masks)  # マスク周囲リングは関数内で計算
        L_plane = 1 - torch.abs(plane_n[2])
        vol_pred = compute_volume(pred_depths, K, masks)
        L_volume = torch.abs(vol_pred - volumes) / (volumes + 1e-8)
        # 総損失
        loss = L_depth + 0.1*L_grad + 0.05*L_plane + 0.1*L_volume
        loss.backward()
        optimizer_da.step()
    # 検証ループ（省略: val_loaderで同様に損失計算しログ）

    # エポック終了ごとに学習率減衰やログ出力
    ...
# Stage 1終了 → Stage 2開始: バックボーンを解凍して再学習
for param in model_da.encoder.parameters():
    param.requires_grad = True
optimizer_da = torch.optim.Adam(model_da.parameters(), lr=lr*0.5)  # LR少し下げ
# Stage 2ループ実行（同様にfew epochs学習）

# モデル保存
torch.save(model_da.state_dict(), "depth_anything_v2_finetuned.pth")


上記はDepth-Anything用のループですが、UniDepth用もmodel_udに対して同様の処理を行います（もしくは引数でモデル種類を選択できるよう1つのループに汎用化します）。重要な差異は以下です。

UniDepthの場合、順伝播結果がpred = model_ud(imgs)で辞書として返る点です。例えば:

pred = model_ud(imgs)
pred_depths = pred["depth"]         # [B,1,H,W]
pred_Ks = pred["intrinsics"]        # [B,3,3]


ここでpred_Ksは各画像ごとのK行列ですが、Nutrition5kでは全てほぼ同一の想定なので、平均的に扱います。損失計算時には基本的にpred_depthsのみ使い、体積計算には真のKを使用します（pred_Kはログ用）。

損失計算はDepth-Anythingと同様ですが、UniDepthのL_depth計算時には全画素に対して行うことも検討します。UniDepthは画像全体の絶対距離を学習するので、テーブル面部分も含めて誤差をとった方が安定する可能性があります。しかし食品外の領域は平面損失で拘束しているため、基本は食品領域中心で問題ありません。

Volume計算には真のKを利用しますが、pred_Kを使っても可です。Fine-tuningが進めばpred_K ≒ 真のKとなることが期待され
GitHub
GitHub
、pred_Kで体積計算すればモデル自身にK誤差も含めたエンドツーエンドなフィードバックがかかります。実装では、開始直後の不安定さを避けるため前半エポックは真Kで計算し、後半エポックでpred_Kに切り替える、といったスケジューリングも考えられます（ただし今回は真K固定でも充分に精度向上すると想定します）。

学習ハイパーパラメータ:

最初は1エポックあたり全データを回す設定で、10エポック程度様子を見ます。Nutrition5kは5000枚程度と規模が大きくないため、過学習に注意しつつ繰り返し回数を調整します。必要ならEarlyStopping（検証損失が改善しなくなったら打ち切り）を導入します。

Optimizer: AdamあるいはAdamWを使用し、初期学習率$1e-4$程度から開始します。Stage 2では学習率を半減または1/5程度に下げ、細かく調整します。

バッチサイズ: VRAMに合わせ小さめ（2～8）で設定します。BatchNormはモデルに無いかLayerNorm主体と思われますが、もしBatchNorm層があれば小バッチでの不安定に注意します（場合によってはGroupNormへ置換）。

AMP自動混合精度: torch.cuda.amp.autocastを利用して半精度訓練を行い、メモリ削減と高速化を図ります。特にLargeモデルの場合有効です。

ロギング: TensorBoardやログ出力で、エポックごとのL_depth, L_volumeなど主要損失の推移を監視します。併せて検証セットでの**体積誤差(絶対/相対)**も計算し、過学習していないか確認します。

評価スクリプト (evaluate_volume.py)

学習後、ホールドアウトテストセットで各モデルの体積推定精度を比較します。以下の手順で評価します。

モデル読み込み: Fine-tuning済みのDepth-AnythingモデルとUniDepthモデルの重みをロードします。推論モードmodel.eval()に設定し、GPUに載せます。

テストデータ準備: Nutrition5kのテスト用Datasetを用意しDataLoader化（バッチサイズ1でも可）。テストではデータ拡張は行わず、生画像をそのまま評価します。

推論と体積計算: 各テスト画像について以下を実施します。

画像と食品マスクを取得（深度GTもあるが評価では使用しない、隠す）。

Depth-Anything推定: 画像テンソルをモデルに入力し、深度マップD_daを得ます。内部パラメータ$K$は既知のものを使用。

UniDepth推定: 画像テンソルをモデルに入力し、深度マップD_udおよび推定K行列K_udを得ます。

体積算出: それぞれについてV_da = compute_volume(D_da, K_{\text{known}}, mask)、V_ud = compute_volume(D_ud, K_{\text{known}}, mask)を計算します。※ここでは公平を期すため、**双方とも真のK（既知のカメラ内部パラメータ）**で体積計算します。UniDepth側はpred_Kもありますが、結果論的に正しい体積が得られるかを評価するため、敢えて同一条件に揃えます。

評価指標計算: 各サンプルについて、絶対誤差$|V_{\text{pred}} - V_{\text{gt}}|$と相対誤差$\frac{|V_{\text{pred}} - V_{\text{gt}}|}{V_{\text{gt}}}$を算出します。ここで$V_{\text{gt}}$はGTの体積ラベルです（Nutrition5kでは深度GTから算出可能）。絶対誤差はmL単位、相対誤差は%で記録します。併せて、深度マップそのものの評価（例えば平均絶対誤差、RMSEなど）も算出可能ですが、今回の目的は体積なので主には見ません。

集計: 全テストデータの誤差を集計し、モデルごとに平均絶対誤差 (MAE)と平均相対誤差を報告します。例えば:

Depth-Anything V2 (fine-tuned): MAE = 20 mL, 相対誤差平均 8%

UniDepth v2 (fine-tuned): MAE = 25 mL, 相対誤差平均 10% （仮の数値）

また誤差分布を見るため、各モデルの誤差ヒストグラムや、体積サイズ別の誤差（小容量ほど誤差%大きい傾向がないか）も分析します。

比較: 上記結果から、どちらのモデルがより正確に体積推定できるかを評価します。例えば相対誤差が小さい方が優秀と判断します。必要なら有意差検定など行いますが、本ケースでは相対誤差の数%の差で十分判断できるでしょう。

評価指標: 今回は体積の絶対誤差・相対誤差が中心です。実用上、例えば100mLの食品で誤差10mL(10%)以内、500mLで誤差25mL(5%)以内といった水準を目標とします。Nutrition5k論文では質量推定に体積を利用した結果、質量MAE約13.7%が得られています
openaccess.thecvf.com
。体積直接の評価指標は論文中明示されていませんが、我々のゴールは相対誤差10%未満程度を目指します。

評価スクリプトでは、結果をCSVやログに保存し、両モデルの指標を一目で比較できるレポートを出力します。例えば以下のような表を出力可能です。

モデル	平均絶対誤差 (mL)	平均相対誤差 (%)
Depth-Anything V2 (FT後)	20.5 mL	8.2%
UniDepth v2 (FT後)	27.3 mL	11.0%

加えて、興味深いケース（誤差の大きかった画像など）について、深度マップと誤差を可視化することも検討します。例えばあるテスト画像で両モデルの深度マップを比較し、どの部分で差が出たか分析します。こうした定性的評価も、モデル改良のヒントになります。

共通API設計と拡張性

新規実装部分はsrc/depthany/以下にまとめ、既存のvolume_depthpro.pyやdepth/unidepth.pyとは直接干渉しないようにします。必要なら既存コードから新モデルを呼び出すインターフェースを用意します。例えば、共通の関数としてpredict_depth(image, model_type)を実装し、model_typeに応じてDepth-AnythingまたはUniDepthをロードして推論する、といったラッパーを提供可能です。これにより、将来的に他のモデル（DPTやMetric3D等）にも簡単に拡張できます。

また、Fine-tuningコード自体も汎用性を意識します。例えばtrain_depth_models.pyではコマンドライン引数で--model=depthanyまたは--model=unidepthを指定できるようにし、一つのスクリプトで両モデルの学習に対応させます。あるいは共通部分（データ読み込み・損失計算）を関数化し、モデル固有処理のみ分ける構造にします。

インフラについては、まずは1枚のGPU上で軽量に実験し、効果を確認します。メモリに余裕が無い場合、中期策としてLoRAアダプタによる微調整や知識蒸留のアプローチも検討できます
GitHub
が、まずは上述のように直接Fine-tuningで十分な結果が得られる想定です。精度向上が頭打ちになれば、合成データの追加やマルチタスク学習（深度+セグメンテーション+体積の同時学習）
GitHub
など、さらなる発展も可能です。

以上が、Depth-Anything V2とUniDepth v2をNutrition5kデータセットでFine-tuningし、体積推定性能を比較評価するための詳細実装プランです。計画に沿って実装・学習を行い、モデルの改善効果を検証していきます。各モデルが食品ドメインでどこまで体積を正確に推定できるようになるか、上記の評価プロセスで明らかにします。
GitHub
GitHub