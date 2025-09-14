はい、**EXIF（撮影端末・焦点距離・露出・撮影時刻・Orientation などのメタデータ）が残っている/残りやすい料理画像**のデータセットはあります。
下に「EXIF が期待できる度」の高い順でまとめます（※配布元からそのまま入手した“オリジナル画像”を使うことが重要です。ミラーや再エンコード版だと EXIF が落ちることがあります）。

---

## ◎ EXIF が“ほぼ”期待できる

* **UNIMIB2016（大学食堂トレイ画像・セグメンテーション付き）**
  ミラノ・ビコッカ大の学食で **Samsung Galaxy S3** で撮影された実写トレイ画像 1,027 枚。スマホ撮影であることが明記されており、オリジナル JPEG からの配布なら EXIF（焦点距離・露出・撮影時刻など）が残っている可能性が高いです。セグメンテーション用アノテーションあり。([ResearchGate][1])

* **ECUST Food Dataset（ECUSTFD：単品 19 クラス／体積推定向け）**
  **スマートフォンでトップビュー＋サイドビュー**を撮った画像で構成。**“オリジナル画像”版**と“リサイズ版”の2種を提供＝オリジナル版を使えば EXIF が残る可能性が高い（リサイズ版は落ちていることが多い）。体積推定用に **コインを写し込むプロトコル**があるのも利点。([arXiv][2])

* **NutritionVerse‑Real（実写 889 枚・分割マスクあり）**
  **iPhone 13 Pro Max** で各料理を多視点撮影した実写データ。Kaggle で公開。オリジナル JPEG を配布している場合は EXIF が残っているケースが多い（実機 iPhone 撮影が明記）。([arXiv][3])

* **Open Images（Flickr 由来・“Food”“Dish”ほかクラスで抽出）**
  9M 超の汎用画像コーパス。**Flickr の“オリジナル解像度画像”が取得可能**で、Flickr はアップロード時の EXIF を保持する設計（表示側で隠す設定は可能）です。食品系クラス（例：Food, Dish, Pizza, Sushi など）に絞って“original”サイズを落とせば、EXIF が残っていることが多いです。([Google Cloud Storage][4])

---

## △ EXIF が「残っている場合もある」／版により欠落しやすい

* **UEC‑FOOD‑100 / 256（日本食中心／バウンディングボックス付）**
  オリジナル画像を配布しており、入手経路次第で EXIF が残る可能性はあります。ただし Web 由来画像が含まれ、再エンコード・再配布版は EXIF が落ちがち。必要なら配布元の zip から直接展開した JPEG を確認してください。([FoodCam][5])

* **FoodSeg103 / 154（食材セグメンテーション）**
  Recipe1M 由来の画像を人手で精緻アノテーションしたベンチマーク。配布形態によってはリサイズ・再保存されており、EXIF が入っていないことがあります（要確認）。([Hugging Face][6])

* **Food‑101（分類）**
  \*\*“全画像を最大辺 512 px にリスケールして再保存”\*\*と公式に明記されており、この工程で EXIF が失われていることが多いです。EXIF 前提の用途には不向き。([Data Vision][7])

> ※「パッケージ画像」で良ければ **Open Food Facts Images（AWS Open Data）** に数百万枚規模の食品パッケージ写真があります。これは料理（調理後の皿）ではなく製品画像ですが、原本に近い JPEG が多く、EXIF を持つものもあります。([Open Data Registry][8])

---

## すぐにできる **EXIF の有無チェック**（落ちていないか検証）

* **exiftool（推奨）**

  ```bash
  exiftool -a -G -s path/to/image.jpg | head
  # 代表タグ: ExifIFD:FocalLength / ExifIFD:ExposureTime / EXIF:DateTimeOriginal / EXIF:Orientation / MakerNotes:...
  ```

  複数ファイルで「焦点距離が空 or 0」の割合を見ると、EXIF 欠落率の当たりが取れます。

* **Python（Pillow）**

  ```python
  from PIL import Image, ExifTags
  im = Image.open("image.jpg")
  exif = { ExifTags.TAGS.get(k,k): v for k,v in (im.getexif() or {}).items() }
  print(exif.get("FocalLength"), exif.get("DateTimeOriginal"), exif.get("Orientation"))
  ```

  *（コマンドは一般情報で、出典は省略）*

---

## 実務メモ（栄養推定・体積推定で EXIF を使うなら）

* **焦点距離・センサー情報は残っても、GPS 等は端末設定で記録されないことが多い**（特に学術データはプライバシー配慮で位置情報が無効）。GPS を必須にせず、\*\*既知寸法の基準物（コイン・カード等）\*\*や **マルチビュー＋幾何**も併用すると堅牢です（ECUSTFD は基準物あり）。([arXiv][2])
* **Open Images を使う場合**は “original” サイズのダウンロード（サムネイルは EXIF 不保持のことがある）。Flickr 由来である点は公式が明記。([Google Cloud Storage][4])
* **Food‑101 のような再保存系**は原理的に EXIF なし前提で扱う。([Data Vision][7])

---

## まとめ（おすすめの選び方）

1. **EXIF 必須（焦点距離・Orientation 等）** → *UNIMIB2016*, *ECUSTFD（オリジナル版）*, *NutritionVerse‑Real*。
2. **大量データが必要 & EXIF も活かしたい** → *Open Images* から *Food/Dish* 系クラスの **original** を抽出。
3. **細粒度な食材セグメ** → *FoodSeg103/154*（EXIF は期待しないで別手段を併用）。

必要なら、目的（例：**体積推定に使えるタグ（FocalLength, FocalPlaneX/YResolution, Orientation など）を保持する画像のみを自動選別**）まで含めて、ダウンロード＆チェック用のスクリプト例もお作りします。

[1]: https://www.researchgate.net/publication/311501638_Food_Recognition_A_New_Dataset_Experiments_and_Results "(PDF) Food Recognition: A New Dataset, Experiments, and Results"
[2]: https://arxiv.org/pdf/1705.07632?utm_source=chatgpt.com "arXiv:1705.07632v3 [cs.CV] 24 May 2017"
[3]: https://arxiv.org/html/2401.08598?utm_source=chatgpt.com "NutritionVerse-Real: An Open Access Manually Collected ..."
[4]: https://storage.googleapis.com/openimages/web/2018-05-17-rotation-information.html?utm_source=chatgpt.com "Rotated images in Open Images - Googleapis.com"
[5]: https://foodcam.mobi/dataset100.html?utm_source=chatgpt.com "\"UEC FOOD 100\": 100-kind food dataset (release 1.0)"
[6]: https://huggingface.co/datasets/EduardoPacheco/FoodSeg103?utm_source=chatgpt.com "EduardoPacheco/FoodSeg103 · Datasets at Hugging Face"
[7]: https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/?utm_source=chatgpt.com "Food-101 -- Mining Discriminative Components with ..."
[8]: https://registry.opendata.aws/openfoodfacts-images/?utm_source=chatgpt.com "Open Food Facts Images - Registry of Open Data on AWS"
