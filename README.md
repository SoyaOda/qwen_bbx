# Qwen BBX Inference

Qwen2.5-VL を使用した食品・食材のバウンディングボックス検出システム

## 概要

このプロジェクトは、Qwen2.5-VL-72B モデルを使用して画像内の食品や食材を検出し、バウンディングボックスとラベルを生成するシステムです。検出結果はJSON形式で保存され、可視化画像も生成されます。

## 主な機能

- 🍱 食品・食材の自動検出
- 📦 正規化されたバウンディングボックス座標 (xyxy形式)
- 🏷️ 英語ラベルによる分類
- 📊 信頼度スコア付き検出結果
- 🎨 検出結果の可視化
- 💾 JSON形式での結果保存

## セットアップ

### 1. 環境構築

```bash
# リポジトリのクローン
git clone <repository-url>
cd qwen_bbx

# Python仮想環境の作成
python3 -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt
```

### 2. API キーの設定

Alibaba Cloud DashScope のAPIキーを環境変数に設定：

```bash
export DASHSCOPE_API_KEY="your-api-key-here"
```

### 3. 設定ファイル

`config.yaml` で以下の設定をカスタマイズできます：

```yaml
provider:
  base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
  model: "qwen2.5-vl-72b-instruct"
  api_key_env: "DASHSCOPE_API_KEY"
  request_timeout_s: 120
  max_retries: 2
  temperature: 0.0
  top_p: 0.1

inference:
  max_items: 50              # 処理する画像の最大数
  conf_threshold: 0.20       # 信頼度の閾値
  iou_merge_threshold: 0.0   # IoU結合閾値

dataset:
  input_dir: "test_images"     # 入力画像ディレクトリ
  out_json_dir: "outputs/json" # JSON出力ディレクトリ
  out_viz_dir: "outputs/viz"   # 可視化画像出力ディレクトリ
```

## 使用方法

### 基本的な実行

```bash
# プロジェクトルートから実行
python src/run_infer.py

# または src ディレクトリから実行
cd src
python run_infer.py
```

### 入力画像の準備

1. `test_images/` ディレクトリに処理したい画像を配置
2. 対応形式: JPG, JPEG, PNG, BMP, GIF, TIFF, WEBP

### 出力ファイル

実行後、以下のディレクトリに結果が保存されます：

- `outputs/json/`: 検出結果のJSONファイル
- `outputs/viz/`: バウンディングボックスを描画した可視化画像

## 出力形式

### JSON構造

```json
{
  "detections": [
    {
      "label_en": "rice",
      "bbox_xyxy_norm": [0.12, 0.40, 0.58, 0.78],
      "confidence": 0.87
    },
    {
      "label_en": "curry sauce",
      "bbox_xyxy_norm": [0.20, 0.45, 0.70, 0.82],
      "confidence": 0.81
    }
  ]
}
```

- `label_en`: 検出された食品・食材の英語ラベル
- `bbox_xyxy_norm`: 正規化座標 [x_min, y_min, x_max, y_max] (0-1の範囲)
- `confidence`: 検出の信頼度スコア (0-1の範囲)

## プロジェクト構成

```
qwen_bbx/
├── src/
│   ├── run_infer.py       # メイン実行スクリプト
│   ├── qwen_client.py     # Qwen API クライアント
│   ├── prompts.py         # プロンプト生成
│   ├── dataset_utils.py   # データセット処理ユーティリティ
│   ├── visualize.py       # 可視化機能
│   └── __init__.py
├── test_images/           # 入力画像ディレクトリ
├── outputs/
│   ├── json/             # JSON出力
│   └── viz/              # 可視化画像出力
├── config.yaml           # 設定ファイル
├── requirements.txt      # Python依存パッケージ
└── README.md            # このファイル
```

## 依存パッケージ

主要な依存関係：
- OpenAI (Qwen API互換クライアント)
- OpenCV (画像処理・可視化)
- Pillow (画像読み込み)
- NumPy (数値処理)
- PyYAML (設定ファイル)
- tqdm (進捗表示)

## トラブルシューティング

### よくある問題と解決方法

1. **ModuleNotFoundError: No module named 'cv2'**
   ```bash
   pip install opencv-python
   ```

2. **API キーエラー**
   ```bash
   # 環境変数が設定されているか確認
   echo $DASHSCOPE_API_KEY
   ```

3. **タイムアウトエラー**
   - `config.yaml` の `request_timeout_s` を増やす
   - ネットワーク接続を確認

4. **メモリ不足**
   - `config.yaml` の `max_items` を減らす
   - 画像サイズを小さくする

## 注意事項

- APIの利用料金が発生する場合があります
- 大量の画像を処理する場合は、APIレート制限に注意してください
- 検出精度は画像の品質や照明条件に依存します

## ライセンス

[ライセンス情報を記載]

## 貢献

プルリクエストや Issue の報告を歓迎します。

## 連絡先

[連絡先情報を記載]