了解です。ここでは **Nutrition5k（Google Research 提供）を“フル”でローカルに入手**するための、**macOS（MacBook）向け手順**と **Google Colab 向け手順**を、容量の目安や検証コマンドまで含めてまとめます。Nutrition5k は **約181.4 GB の .tar.gz** もしくは **GCS バケットの全体同期**で取得できます（展開後は **200GB 以上**を見込んでください）。公式の配布元・構成・ライセンスは GitHub の README にまとまっています。([GitHub][1])

---

## 事前に知っておくこと（要点）

* **配布元**：Google Research の公式リポジトリ（GCS バケット `gs://nutrition5k_dataset/nutrition5k_dataset`）と 1 ファイル版 `nutrition5k_dataset.tar.gz`（181.4 GB）。([GitHub][1])
* **内容**：5,006 皿分のデータ（**側面4方向の回転動画**, **上面RGB-D**, **食材ごとの重量**, **総カロリー/三大栄養素** など）と **train/test split・評価スクリプト**。([GitHub][1])
* **ライセンス**：**Creative Commons 4.0**（商用含め再配布・改変可、クレジット表記要）。論文の引用情報も README に記載。([GitHub][1])

---

## A. macOS（ローカル）で“フル”ダウンロードする

> 推奨：**GCS 同期（rsync）**。途中中断しても再開しやすく、部分的な取りこぼしを自動で補完できます。Google は現在 `gsutil` より **`gcloud storage`** の利用を推奨しています（どちらでも可）。([Google Cloud][2])

### 1) 必要なツールを入れる

Homebrew が入っている前提で、Google Cloud CLI（`gcloud`/`gsutil` 同梱）をインストール：

```bash
brew install --cask gcloud-cli
gcloud --version
```

（`gcloud-cli` は旧 `google-cloud-sdk` の後継トークン）([Homebrew Formulae][3])

> 初回のみ `gcloud init` / `gcloud auth login` を実行しておくと確実です（公開バケット読み取りのみなら認証なしでも動くことがありますが、CLI 初期化を済ませておくとトラブルが減ります）。([Google Cloud][4])

### 2) 保存先を用意（外付けSSD推奨）

```bash
mkdir -p ~/data/nutrition5k
```

> ダウンロード181.4GB＋展開後200GB超を想定。余裕のある APFS もしくは exFAT のドライブを推奨（大容量ファイル可）。サイズの数字は公式 README の .tar.gz 181.4GB から。([GitHub][1])

### 3) （推奨）バケットから丸ごと同期

**新推奨コマンド（速くて堅牢）**：

```bash
# 丸ごと同期（再実行で差分だけ取る）
gcloud storage rsync -r gs://nutrition5k_dataset/nutrition5k_dataset ~/data/nutrition5k
```

**従来の gsutil 版（READMEの記載に準拠）**：

```bash
# 一括コピー
gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/*" ~/data/nutrition5k

# あるいは同期運用
gsutil -m rsync -r gs://nutrition5k_dataset/nutrition5k_dataset ~/data/nutrition5k
```

（`-m` は並列、`rsync` は再実行で取りこぼしを補完できるのが利点）([GitHub][1])

### 4) （代替）181.4GB の単一アーカイブで入手 → 展開

```bash
# GCS から .tar.gz を取得（ブラウザ経由だと Google サインインを求められる場合あり）
gcloud storage cp gs://nutrition5k_dataset/nutrition5k_dataset.tar.gz ~/data/
cd ~/data && tar -xzf nutrition5k_dataset.tar.gz
```

> `.tar.gz` の存在とサイズは公式 README に明記（181.4GB）。([GitHub][1])

### 5) ダウンロード後の検証（中身の確認）

```bash
cd ~/data/nutrition5k
# 上位ディレクトリをざっと確認
ls -1
# メタデータの先頭を見る
head -n 5 metadata/dish_metadata_cafe1.csv
head -n 5 metadata/dish_metadata_cafe2.csv
```

フォルダ構成（README より）：

* `imagery/side_angles/`（側面動画・皿IDごと）
* `imagery/realsense_overhead/`（上面RGB・深度）
* `metadata/`（食材メタ・皿レベル栄養情報 CSV）
* `dish_ids/splits/`（公式 train/test）
* `scripts/`（評価スクリプトなど）([GitHub][1])

### 6) 公式スクリプトとフレーム抽出（任意）

```bash
# リポジトリのスクリプトを取りたい場合
git clone https://github.com/google-research-datasets/Nutrition5k.git
brew install ffmpeg  # フレーム抽出に必要
# READMEの例：動画→フレーム
ffmpeg -i imagery/side_angles/dish_XXXXXXXXXX/angle_A.mp4 output_%03d.jpeg
```

（抽出の手順・スクリプト名は README に記載）([GitHub][1])

---

## B. Google Colab で扱う（注意：フル取得は非推奨）

Colab の一時ストレージは**容量が不足**しやすいため、**Google Drive をマウントしてそちらに同期**するか、**一部だけ取得**する運用を推奨します。

```python
# Colab セル例：Drive をマウント
from google.colab import drive
drive.mount('/content/drive')

# GCS から Drive へ同期（時間・容量に注意）
!mkdir -p /content/drive/MyDrive/datasets/nutrition5k
!gsutil -m rsync -r gs://nutrition5k_dataset/nutrition5k_dataset /content/drive/MyDrive/datasets/nutrition5k
```

> Nutrition5k は巨大（.tar.gz 181.4GB）なので、Colab では **metadata だけ**や **overhead 画像だけ**といったサブセット同期が現実的です：

```bash
# 例：メタデータだけ
gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/metadata" /content/drive/MyDrive/datasets/nutrition5k/

# 例：上面RGB-Dだけ
gsutil -m cp -r "gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead" /content/drive/MyDrive/datasets/nutrition5k/
```

（公式 GCS パスは README に記載）([GitHub][1])

---

## よくあるハマりどころと対処

* **`storage.cloud.google.com` がブラウザでサインインを要求**：CLI で `gcloud storage cp` / `gsutil cp` を使うと回避しやすい。([GitHub][1])
* **途中で失敗した**：`gcloud storage rsync -r` か `gsutil -m rsync -r` を再実行（差分のみ同期）。([Google Cloud][5])
* **転送が遅い**：`-m`（並列）を付ける。ネットワーク安定化のため有線/高速回線・外付けSSD推奨。([GitHub][1])
* **フルでなくサブセットでもよい？**：Kaggle に**部分版**のミラーや派生データがありますが、**完全版は公式 GCS が確実**です。([Kaggle][6])

---

## 参考（一次情報）

* 公式 README（配布方法・内容・スクリプト・ライセンス）([GitHub][1])
* 論文（CVPR 2021）：データのスケールや取得方法の詳細（20k短編動画／5k皿、RGB‑D、食材重量など）([CVF Open Access][7])
* Google Cloud CLI / `gcloud storage rsync` ドキュメント（`gsutil` より推奨）([Google Cloud][8])

---

### これで “フルインストール” は完了です

Mac ローカルなら **A-3 の同期**がもっとも安全・堅牢です。もし「**この構成（メタデータだけ／RGB‑D だけ／動画フレーム化だけ等）で取りたい**」といった希望があれば、目的別の**最小サブセット同期コマンド**をすぐに作成します。

[1]: https://github.com/google-research-datasets/Nutrition5k "GitHub - google-research-datasets/Nutrition5k: Detailed visual + nutritional data for over 5,000 plates of food."
[2]: https://cloud.google.com/storage/docs/gsutil?utm_source=chatgpt.com "gsutil tool | Cloud Storage"
[3]: https://formulae.brew.sh/cask/gcloud-cli?utm_source=chatgpt.com "gcloud-cli"
[4]: https://cloud.google.com/sdk/docs/install-sdk?utm_source=chatgpt.com "Quickstart: Install the Google Cloud CLI"
[5]: https://cloud.google.com/sdk/gcloud/reference/storage/rsync?utm_source=chatgpt.com "gcloud storage rsync | Google Cloud SDK Documentation"
[6]: https://www.kaggle.com/datasets/zygmuntyt/nutrition5k-dataset-side-angle-images?utm_source=chatgpt.com "Nutrition5k dataset side angle images"
[7]: https://openaccess.thecvf.com/content/CVPR2021/papers/Thames_Nutrition5k_Towards_Automatic_Nutritional_Understanding_of_Generic_Food_CVPR_2021_paper.pdf?utm_source=chatgpt.com "Nutrition5k: Towards Automatic Nutritional Understanding ..."
[8]: https://cloud.google.com/sdk/docs/install?utm_source=chatgpt.com "Install the gcloud CLI | Google Cloud SDK Documentation"
