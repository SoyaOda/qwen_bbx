Depth Proへの移行と実装プラン
背景: UniDepth v2の課題とDepth Pro導入の目的

現在、画像から料理ごとの体積を推定するパイプラインではQwen-VLによる認識結果をもとにSAM 2.1で領域マスクを生成し、UniDepth v2で深度マップを推定して体積計算を行っています
GitHub
GitHub
。しかしUniDepth v2にはスケール誤差の問題があり、推定されるカメラ内部パラメータ（特に焦点距離fx, fy）が実際より約10倍も小さいため、深度から計算される体積が極端に大きく（例: 46リットル）なってしまうことが判明しています
GitHub
。この問題に対処するためにUniDepthではK_scale_factorによる補正を導入し、約10.5倍のスケーリングで理論上は正しい値に近づけました
GitHub
。しかし、最適なK_scale値は画像によって1.0〜12.0と変動し、深度に応じた適応調整でも成功率は33%程度に留まっています
GitHub
。その結果、現在の実装（unidepth_runner_final.py）では深度の中央値に基づきK_scaleを自動決定するロジックが含まれています
GitHub
。

以上の課題から、Apple社の最新深度推定モデル「Depth Pro」を統合し、絶対スケール精度の高い深度マップを取得することで体積推定の信頼性向上を図ります。Depth Proは事前のカメラパラメータ無しでメートル単位の絶対距離の深度マップをゼロショットで生成できるモデルであり
github.com
、高解像度かつ高周波のディテールも捉える優れた性能を持ちます
github.com
。このモデルを用いることで、UniDepth v2で必要だった煩雑なK_scale補正を省き、料理の体積を直接現実スケール（容積の絶対値）で算出できると期待されます。

Apple Depth Proモデルの概要と入出力形式

Apple Depth ProはAppleがICLR 2025で発表した最先端の単眼深度推定モデルです
machinelearning.apple.com
。入力はRGB画像（通常のカラー画像）で、出力は対応する深度マップです。深度マップは各ピクセルについてカメラからの距離を表し、単位はメートル（m）となっています
github.com
。特筆すべきは、この深度値が絶対スケールで提供される点で、カメラの焦点距離などのメタデータに依存しなくても実際の距離スケールを推定できることです
github.com
。モデル内部で単一画像から焦点距離を推定する機構を持ち、これによって絶対距離の精度を確保しています
github.com
。

Depth ProのAPI仕様については、GitHub上で公開されている実装コードおよびHugging Faceのモデルカードに詳しい使用例があります
github.com
github.com
。例えばPythonから以下のように利用できます
github.com
:

import depth_pro
model, transform = depth_pro.create_model_and_transforms()
model.eval()
# 画像読み込みと前処理（EXIFから焦点距離を取得）
image, _, f_px = depth_pro.load_rgb("input.jpg")
image_tensor = transform(image)
# 推論実行
prediction = model.infer(image_tensor, f_px=f_px)
depth_map = prediction["depth"]             # 深度マップ（単位:m）
estimated_focal = prediction["focallength_px"]  # 推定または利用された焦点距離[pixels]


上記のように、Depth Proでは入力画像を前処理する際にEXIFから35mm換算の焦点距離を自動抽出し、ピクセル単位の焦点距離 (f_px) を算出できます
GitHub
。モデル推論時にこのf_pxを渡すことで、もし入力画像のカメラ情報が得られる場合にはそれを活用し、より正確なスケールで深度を推定します（カメラメタデータがなくともモデル自身が焦点距離を推定可能です）。推論結果として返されるdepthは実シーンのメートル単位の距離マップであり、例えばdepth[y,x] = 1.20ならその地点まで約1.20mという意味になります。またpredictionには推定された焦点距離focallength_pxも含まれます
github.com
。精度に関して、Depth Proは従来モデルに比べて高い絶対距離精度と細部の表現力を持つことが報告されており
github.com
、高解像度（約2.25メガピクセル）の深度マップを0.3秒で生成できる高速性も特徴です
github.com
。

以上より、Depth Proの入力形式は特別なものではなく通常のカラー画像で、出力はメートルスケールの深度マップです。Depth Pro自体は点群や体積を直接出力する機能はありませんが、深度マップさえ取得できれば後段の処理で体積推定が可能です。本プロジェクトでは、このDepth Proを既存パイプラインに組み込み、絶対単位での体積算出を実現します。

Depth Proモデル導入の準備 (依存関係の追加)

まず、Depth Proをプロジェクトで使用できるよう環境にモデルをインストールします。Appleの公式GitHubリポジトリapple/ml-depth-proからコードを取得し、パッケージをインストールします
github.com
。例えば、開発環境にて:

# Depth Proリポジトリをクローンしてインストール
git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro
pip install -e .
# or, if using conda:
# conda create -n depth-pro -y python=3.9 && conda activate depth-pro && pip install -e .


これにより、Pythonパッケージdepth_proとしてモデルを利用できるようになります
github.com
。また、学習済みモデルのチェックポイントを入手する必要があります。リポジトリ内のget_pretrained_models.shスクリプトを実行するか、Hugging Face Hubのapple/DepthProモデルからダウンロードしてください
github.com
huggingface.co
。例えばHugging Face CLIを使う場合:

huggingface-cli download --local-dir checkpoints apple/DepthPro


ダウンロードした重みファイル（例: depthpro_model.ptなど）を所定のパスに配置し、モデルロード時に読み込めるようにします。以降のコードではこのDepth Proモデルと重みを使用することを前提とします。

補足: 本リポジトリ（SoyaOda/qwen_bbx）には現時点でDepth Pro関連のコードや依存関係は含まれていないため、上記のようにゼロから統合する形になります。必要に応じてrequirements.txtやenvironment.yamlにdepth_proや関連ライブラリ（例: pillow-heif for HEIC読み込み
GitHub
）を追記してください。

Depth Pro推論エンジンの実装

UniDepth v2に対するUniDepthEngineクラス（例えばunidepth_runner.py等）にならい、Depth Pro用に新たな推論エンジンDepthProEngine（仮称）を実装します。これにより既存コードを大きく変更せず、差し替えが可能になります。以下に実装方針とコード例を示します:

モデルのロード: コンストラクタ__init__でDepth Proモデルと前処理transformをロードします。Hugging Face経由でインストールしたdepth_proパッケージを用い、depth_pro.create_model_and_transforms()でモデル本体と前処理器を取得します
github.com
。また、実行デバイス(cuda/cpu)の指定も行います（UniDepthEngineと同様に）。

画像入力と前処理: Depth Proの前処理では、画像をPILで読み込みつつEXIF情報から焦点距離を取得するユーティリティdepth_pro.load_rgbを利用します
github.com
。これによりf_px（ピクセル単位の焦点距離）が得られます。返される画像はnumpy配列になりますので、Depth Pro付属のtransformを適用してモデル入力用テンソルに変換します
github.com
。UniDepthではImageNet正規化の有無を選べましたが
GitHub
、Depth Proでは内部で適切に正規化されるため追加の正規化は不要です。

推論実行: 準備したテンソルをモデルに入力し、model.infer(image_tensor, f_px=f_px)で深度推定を行います
github.com
。推論結果は辞書で返り、キー"depth"に深度マップ（torchテンソル）、"focallength_px"に推定焦点距離が含まれます
github.com
。必要に応じてtorchテンソルを.detach().cpu().numpy()でnumpy配列に変換します。Depth Proはカメラ内部行列K自体は返さないため、我々の側で適切な内部パラメータ行列を構成する必要があります。

内部パラメータKの構築: Depth Proから得られた焦点距離focallength_pxを用いて、画像の幅W・高さHに合わせた3x3のカメラ内参照行列Kを構築します。例えば、principal point(cx, cy)には画像中心(W/2, H/2)を仮定し、fx=fy=focallength_px、他の要素はゼロ/1として以下のように設定します:

H, W = depth_map.shape  # 深度マップの高さ・幅
f = focallength_px
cx, cy = W / 2.0, H / 2.0
K = np.array([[f, 0, cx],
              [0, f, cy],
              [0, 0, 1]], dtype=np.float64)


これにより、Depth Proモデルが想定したカメラ内部パラメータを明示的な行列形式にできます。なお、EXIFに基づく正確な光学中心が不明な場合でも、スマートフォン画像であれば画像中心を仮定して大きな問題はありません。

3D点群の算出: 深度マップと内部行列Kが得られたら、UniDepth実装と同様に各画素をカメラ座標系の3次元点に逆投影します。既存の_unproject_depth_to_xyz関数を再利用し（または同等の処理を実装し）
GitHub
GitHub
、X座標(u-cx)*Z/fx、Y座標(v-cy)*Z/fy、Z座標= Zとして点群xyzを計算します。ここでu,vは画素座標（横方向・縦方向のインデックス）、Zはその画素の深度値です。Depth Proの深度はメートル単位なので、得られる点群座標もメートル単位になります。

結果の構造: UniDepthEngineと互換性を保つため、DepthProEngine.infer_image()は戻り値として以下の辞書を返します:

{
    "depth": depth_map_numpy,        # 深度マップ (H,W) numpy配列 [m]
    "intrinsics": K,                # カメラ内部パラメータ行列 (3,3)
    "intrinsics_raw": K_raw_or_None,# 元のK（今回はDepth Proでは調整前後の差がないため同じ値を入れる）
    "points": xyz,                  # 3D点群 (H,W,3) numpy配列 [m]
    "confidence": None              # Depth Proは信頼度マップ未提供
}


intrinsics_rawについては、UniDepthではスケーリング前の推定Kでしたが、Depth ProではモデルからK自体が出ないためintrinsicsと同一の行列を入れておきます（もしくはNoneでも構いませんが、後段処理がキー存在を仮定しているなら同じ値を入れておくと安全です）。ConfidenceマップもUniDepth v2のみの機能
GitHub
なので、Depth ProではNoneまたは省略とします。
こうした実装によって、Depth Proによる深度推定エンジンが完成します。以下に疑似コード形式で実装イメージを示します:

# 擬似コード: DepthProEngine.py
import depth_pro
import numpy as np
class DepthProEngine:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Depth Proデバイス: {self.device}")
        # モデルとtransformの取得
        self.model, self.transform = depth_pro.create_model_and_transforms()
        self.model = self.model.to(self.device).eval()
        # チェックポイント読み込み（必要に応じて）
        # self.model.load_state_dict(torch.load("checkpoints/depthpro_model.pt"))
    def infer_image(self, image_path: str) -> Dict[str, Any]:
        # 画像読み込みとEXIF処理
        image_np, _, f_px = depth_pro.load_rgb(image_path)
        # 前処理transform適用（テンソル化等）
        input_tensor = self.transform(image_np).to(self.device)
        # Depth Pro推論実行
        with torch.inference_mode():
            pred = self.model.infer(input_tensor, f_px=f_px)
        depth_torch = pred["depth"]            # torch tensor
        depth = depth_torch.squeeze().cpu().numpy()  # (H,W) numpy配列
        # 焦点距離の取得（提供されたか推定されたかに関わらず取得）
        if "focallength_px" in pred:
            f_px_val = float(pred["focallength_px"])  # tensor->float
        else:
            f_px_val = f_px if f_px is not None else estimate_focal_from_image(image_np)
        # 内部行列Kの構築
        H, W = depth.shape
        cx, cy = W/2.0, H/2.0
        K = np.array([[f_px_val, 0,      cx],
                      [0,        f_px_val, cy],
                      [0,        0,        1]], dtype=np.float64)
        # 3次元点群の算出
        X_idx, Y_idx = np.meshgrid(np.arange(W), np.arange(H))
        Z = depth
        X = (X_idx - cx) * Z / f_px_val
        Y = (Y_idx - cy) * Z / f_px_val
        xyz = np.stack([X, Y, Z], axis=-1)  # (H,W,3)
        # 結果を辞書で返す
        return {
            "depth": depth,
            "intrinsics": K,
            "intrinsics_raw": K.copy(),  # 同じ値
            "points": xyz,
            "confidence": None
        }


上記では簡潔のため例外処理等は省いていますが、実際にはモデルロード失敗時のエラー表示（UniDepthEngine同様
GitHub
）や、EXIFが無い場合のf_px推定（Depth Pro内部の機構で推定されるためここでは気にしない）などを考慮してください。

メインパイプラインの修正 (UniDepth v2 → Depth Pro)

次に、既存のパイプラインコードをDepth Proエンジンに差し替えます。主な変更箇所は深度推定部分と、それに付随するK_scale補正の除去です。

Engineの置き換え: これまでunidepth_runner_v2.pyやunidepth_runner_final.pyでUniDepthEngineを初期化・利用していた箇所を、DepthProEngineに置き換えます。例えばrun_unidepth.py（Qwen→SAM→UniDepth→体積のメインパイプライン
GitHub
）内で:

# 変更前 (UniDepthEngineの初期化)
uni_engine = UniDepthEngineFinal(model_repo="lpiccinelli/unidepth-v2-vitl14")
pred = uni_engine.infer_image(image_path, K_mode="adaptive")
depth = pred["depth"]
K = pred["intrinsics"]
...


となっていた部分を:

# 変更後 (DepthProEngineの初期化と利用)
depth_engine = DepthProEngine(device="cuda")
result = depth_engine.infer_image(image_path)
depth = result["depth"]
K = result["intrinsics"]
xyz = result["points"]


のように修正します。これにより、以降の処理（平面検出や体積計算）にはDepth Pro由来のdepthおよびK、pointsが供給されます。

K_scale関連処理の削除: UniDepth用に実装されていたK_scale_factor調整ロジックは不要となるため完全に削除します。具体的には、UniDepthEngineで出力されたintrinsics_raw行列に対しK_scaleを乗じてintrinsicsを得ていた処理
GitHub
GitHub
を取り除きます。例えばinfer_image内で:

# 変更前 (UniDepthEngineFinalでのKスケーリング例)
K_raw = pred.get("intrinsics")
if K_mode == "adaptive":
    K_scale = self.estimate_K_scale_for_food(depth)
elif K_mode == "fixed":
    K_scale = fixed_K_scale
else:
    K_scale = 1.0
K = K_raw * np.diag([K_scale, K_scale, 1])  # fx, fyのみスケール


のようなコードブロックを削除します。K_mode引数やfixed_K_scale引数自体ももはや意味を持たないため、関連する引数定義や設定値も除去します（例えばconfig.yaml内のK_scale_factor設定
GitHub
GitHub
は使用しないか、デフォルト1.0に固定します）。Depth Proでは常に実測スケールで深度を出力するため、このような補正は不要になります
github.com
。

Confidenceの扱い: Depth Proには信頼度マップの出力がありません。現行のUniDepthEngineではconfidenceがオプションで存在し、一部設定で体積計算時に信頼度を重み付けする機能がありました（configのuse_confidence_weight
GitHub
）。今回Depth Proではconfidence=Noneとなりますので、仮に後段のVolumeEstimator等でuse_confidence_weight=Trueになっていても無視されるかFalseに設定する必要があります（幸いデフォルトfalse
GitHub
）。基本的にはConfidenceに依存した処理はオフにしておけば問題ありません。

以上の変更により、深度推定ステージはDepth Proベースに置き換わります。コード全体としてはUniDepth用処理が減るため簡潔になり、K_scaleの適切値を模索する試行錯誤（例えばtest_all_images.pyでの最適K_scale探索
GitHub
）も不要になります。

体積推定アルゴリズムの適用とDepth Pro向け最適化

Depth Proで得られた深度マップと点群を用いて、各料理領域の体積を計算するアルゴリズムを実行します。基本的な手順は従来と同じですが、Depth Proではスケールが正確であることを踏まえ、以下の点でシンプルかつ確実な実装となります。

平面検出 (テーブル面の推定): まず、画像中のテーブル平面をRANSACで推定します。これはUniDepth時と同様、料理マスクのすぐ外側のリング領域や背景領域の点群から平面をフィッティングする処理です
GitHub
GitHub
。既存のplane_fit.py/plane_fit_v2.pyをDepth Proの出力点群 (xyz) に対して適用します。パラメータ（距離閾値dist_thなど）は従来通りでまず問題ありません。Depth Proの深度マップは高精度なため、むしろ平面フィッティングの精度も向上し、外れ値に強くなります。必要に応じてRANSACの距離閾値を微調整します。例えば従来は深度中央値に応じて閾値を0.006m(6mm)程度に適応調整していました
GitHub
。Depth Proでも同様の閾値算出式を使い、より安定した平面推定を行います。
GitHub

体積計算: 推定されたテーブル平面と料理領域の点群を用いて、各画素がテーブルからどれだけ高さがあるかを求めます。具体的には、テーブル平面を法線ベクトル$\mathbf{n}$・距離$d$の形式で得ているとすると、料理領域内の各点$\mathbf{p}=(x,y,z)$について平面からの垂直距離を$\Delta h = (\mathbf{n}\cdot \mathbf{p} - d)$で計算します（$\mathbf{n}$は単位法線ベクトルと仮定）
GitHub
。テーブル上にある点では$\mathbf{n}\cdot\mathbf{p}\approx d$となり$\Delta h\approx0$、料理の表面の点では$\Delta h$が正の値（高さ）になります。

次に、各点に対応する平面上の面積要素を求めます。従来は**各画素の面積 $a_{\text{pix}}$を深度$Z$と内部パラメータから近似的に算出していました
GitHub
:

𝑎
pix
=
𝑍
2
𝑓
𝑥
⋅
𝑓
𝑦
a
pix
	​

=
f
x
	​

⋅f
y
	​

Z
2
	​


ここで$Z$はその画素の深度値、$f_x, f_y$はカメラの焦点距離[pixel]です
GitHub
（画素サイズなどの係数はこの式に含まれています）。Depth Proでは$f_x, f_y$ともに実測スケールに合った値になっているため、この式で求めた$a_{\text{pix}}$は実空間でその画素がカバーする面積（平方メートル）**を表すことになります
GitHub
。例えば深度$Z=1.0$[m]で$f_x=f_y=10500$なら、$a_{\text{pix}}\approx 1/(10500^2)\approx9.07\times10^{-8}$ m²となり、これはその深度での1ピクセルの床面積が約$0.09,\text{mm}^2$に相当することを意味します。

各画素の高さ$\Delta h$と面積$a_{\text{pix}}$が分かれば、その画素柱の体積$\Delta V$は単に $\Delta V = \Delta h \times a_{\text{pix}}$ となります。実装上は、料理マスク内の全ピクセルについてこの$\Delta V$を積分（=総和）することで料理全体の体積$V$を求めます。式で書けば:

𝑉
=
∑
(
𝑢
,
𝑣
)
∈
mask
𝑍
(
𝑢
,
𝑣
)
2
𝑓
𝑥
𝑓
𝑦
(
𝑛
⋅
𝑝
(
𝑢
,
𝑣
)
−
𝑑
)
V=∑
(u,v)∈mask
	​

f
x
	​

f
y
	​

Z(u,v)
2
	​

(n⋅p(u,v)−d)
となります（$\mathbf{p}(u,v)$はその画素の3D座標）。UniDepth時代は$f_x, f_y$のスケール誤差のせいでこの計算から得られる$V$が不正確でしたが、Depth Proではモデル自体が$f_x,f_y$を推定してくれるため
github.com
、追加のスケール補正なしでこの体積算出式がそのまま適用できます。

実装メモ: 既存のVolumeEstimatorモジュールが上記の計算を担っている場合、その中で使われる画素面積計算式を確認します（configのarea_formulaが"z2_over_fx_fy"となっている通り、この式が採用されています
GitHub
）。従来この処理はintrinsicsから直接$fx,fy$を取得していましたが、Depth Pro統合後も同様にresult["intrinsics"]から使えます。K_scale_factorの考慮はもう不要なので、VolumeEstimator内でそれに関する処理があれば削除します。例えば以前は$fx,fy$が小さすぎる場合に体積過大となる懸念がありましたが
GitHub
GitHub
、Depth Proでは適切な値になるため過剰補正せずとも大丈夫です。

体積値の単位と出力: 算出された体積$V$は、深度[m]と面積[m²]の積分なので**立方メートル（m³）で得られます。ユーザが要求する出力形式に合わせ、必要なら単位変換を行います。今回「絶対値であれば単位は何でもよい」とのことですが、分かりやすさから立方センチメートル(cm³)またはミリリットル(ml)**で出力するのが良いでしょう（1 cm³ = 1 ml）。例えば$V=5.2\times10^{-4}$ m³であれば、それは$5.2\times10^{-4}\times(100^3)$ cm³ = 520 cm³に等しいので、**約520 cm³（=520 ml）**と表現できます。コード上は$V_{\text{cm3}} = V_{\text{m3}} \times 1e6$で変換できます。特に食物の体積は数百～数千cm³程度のオーダーが予想されるため、cm³表示が適切と考えます。出力フォーマットの例:

volume_m3 = total_volume  # 上の計算で得た体積[m^3]
volume_cm3 = volume_m3 * 1e6
print(f"推定体積: {volume_cm3:.1f} cm^3")


とすれば小数1桁程度で容量を表示できます（必要に応じて四捨五入やリットル換算も追加）。

以上がDepth Proを用いた体積推定アルゴリズムの流れです。重要なのは、Depth Proに合わせてスケール周りの不確実性が解消される点で、これにより複雑な補正処理が省け、アルゴリズム自体はシンプルになります。**「Depth Proに最適化されたアルゴリズム」**とはすなわち、入力段階で可能な限り正確なカメラ情報を渡し（EXIFの焦点距離を利用）、出力された深度マップをそのまま実スケールの点群・体積計算に使うことを意味します。LiDARとの融合や特別なモデルは現時点では考慮せず、Depth Proモデル単体で完結する方針です（Apple提供のMLモデル自体が高精度なので追加の深度センサーは不要と判断）。ただし将来的にiPhoneのTrueDepthやLiDARデータが利用可能であれば、テーブル平面検出の初期値に使うなど拡張の余地はあります。

テストと検証計画

実装後は、既存のテストスクリプト（例えばtest_with_masks_only.py）や新規テスト画像を用いて、体積推定結果を検証します。具体的には:

いくつか既知の大きさの物体でテストし、概算体積が現実と大きく乖離しないことを確認します。例えば500mlペットボトルを撮影し体積推定させた場合に、おおよそ500ml前後の値が出るか検証します。

Depth Pro統合前後で同じ料理画像に対する体積出力を比較し、明らかに不自然な巨大値（リットル級）になっていた問題が解決していることを確認します。期待される効果として、従来46Lなどと出ていたケースは数百mL程度に収まるはずです。

デバッグ出力の確認: unidepth_runner_final.pyには推定結果に対するサニティチェック（典型的な画素面積$a_{\text{pix}}$のオーダー）があります
GitHub
。Depth Pro導入後はおそらく$a_{\text{pix}}$が非常に小さい値（例えば1e-8以下）になるため、「a_pixが小さすぎます（体積が過小になる可能性）」という警告が出るかもしれません
GitHub
。しかし、これは正しくスケールされた状態であり心配は不要です。必要ならこの閾値を引き下げる（例えば警告条件を1e-9未満に変更する）か、チェック自体を無効化してもよいでしょう。逆に$a_{\text{pix}}$が1e-5より大きいような警告
GitHub
が出なくなれば、スケール問題が解決した証拠と考えられます。

性能面の確認: Depth Proは高精度な分計算量もありますが、論文によれば2.25MPの画像を0.3秒で処理可能です
github.com
。本システムで扱う画像サイズやリアルタイム性要求に照らし、処理時間が許容範囲か確認します。必要であればGPUを活用したり、モデルサイズ（バックボーン）を小さいものに変更することも検討します。

最終的に、以上のプロセスによりUniDepth v2からDepth Proへの置き換えが完了し、料理体積の推定が曖昧さ無く絶対単位で出力できるようになります。Depth Proの活用で焦点距離スケールの問題が解消されるため、これまでボトルネックであったスケール補正のチューニングから解放され、開発効率と推定精度の向上が見込まれます
github.com
。今後は本実装プランに沿ってコードを修正・追加し、順次テストを行って本番システムへ統合していきます。