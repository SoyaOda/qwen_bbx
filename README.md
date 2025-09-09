# Qwen BBX Inference

Qwen2.5-VL を使用した食品・食材のバウンディングボックス検出システム

## 概要

このプロジェクトは、Qwen2.5-VL-72B モデルを使用して画像内の食品や食材を検出し、バウンディングボックスとラベルを生成するシステムです。検出結果はJSON形式で保存され、可視化画像も生成されます。

## 主な機能

### Qwen2.5-VL による物体検出
- 🍱 食品・食材の自動検出
- 📦 正規化されたバウンディングボックス座標 (xyxy形式)
- 🏷️ 英語ラベルによる分類
- 📊 信頼度スコア付き検出結果
- 🎨 検出結果の可視化
- 💾 JSON形式での結果保存

### SAM 2.1 統合（NEW）
- 🎯 Qwen検出結果を用いた高精度セグメンテーション
- 🔍 2つのモデル比較（base_plus vs large）
- 📈 マスク面積・IoUスコアの自動計算
- 🖼️ マスクの可視化と差分解析
- 💾 バイナリマスクPNG出力

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

# SAM2 設定（オプション）
sam2:
  repo_root: "/path/to/sam2"   # SAM2リポジトリのパス
  device: "cuda"                # 使用デバイス
  dtype: "bfloat16"            # データ型
  multimask_output: true       # 複数マスク出力
  conf_threshold: 0.20         # 信頼度閾値
```

## 使用方法

### 基本的な実行（Qwen2.5-VL のみ）

```bash
# プロジェクトルートから実行
python src/run_infer.py

# または src ディレクトリから実行
cd src
python run_infer.py
```

### SAM 2.1 統合実行（オプション）

Qwen2.5-VL の検出結果を用いて、SAM 2.1 でセグメンテーションを実行：

```bash
# 1. まず Qwen で検出を実行
python src/run_infer.py

# 2. SAM 2.1 でマスク生成（base_plus と large の両モデルで実行）
python3 src/run_sam2_v2.py
```

SAM 2.1 を使用する場合は、事前に以下のセットアップが必要です：

1. **SAM2 リポジトリのクローン**
   ```bash
   git clone https://github.com/facebookresearch/sam2.git
   cd sam2
   pip install -e .
   ```

2. **チェックポイントのダウンロード**
   ```bash
   cd checkpoints
   ./download_ckpts.sh
   # または個別にダウンロード：
   # sam2.1_hiera_base_plus.pt
   # sam2.1_hiera_large.pt
   ```

3. **config.yaml の SAM2 設定を更新**
   ```yaml
   sam2:
     repo_root: "/path/to/sam2"  # 実際のパスに変更
   ```

### 入力画像の準備

1. `test_images/` ディレクトリに処理したい画像を配置
2. 対応形式: JPG, JPEG, PNG, BMP, GIF, TIFF, WEBP

### 出力ファイル

#### Qwen2.5-VL 出力
- `outputs/json/`: 検出結果のJSONファイル
- `outputs/viz/`: バウンディングボックスを描画した可視化画像

#### SAM 2.1 出力（オプション）
- `outputs/sam2/json/`: セグメンテーション統計情報
  - 各検出物体の面積（ピクセル数）
  - 予測IoUスコア
  - base_plus vs large モデルの比較IoU
- `outputs/sam2/masks/`: バイナリマスクPNG（0/255形式）
  - 個別オブジェクトごとのマスク画像
- `outputs/sam2/viz/`: マスク可視化画像
  - `*_bplus.jpg`: base_plus モデルの結果
  - `*_large.jpg`: large モデルの結果
  - `*_panel.jpg`: 4分割比較画像（原画像｜b+｜large｜差分）

## 出力形式

### Qwen2.5-VL JSON構造

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

### SAM 2.1 JSON構造（オプション）

```json
{
  "image": "train_00000.jpg",
  "width": 512,
  "height": 384,
  "detections": [
    {
      "id": 0,
      "label_en": "rice",
      "qwen_confidence": 0.95,
      "bbox_xyxy_norm": [0.32, 0.14, 1.0, 0.89],
      "bbox_xyxy_abs": [163.84, 53.76, 512.0, 341.76],
      "sam2_bplus": {
        "area_px": 71635,
        "pred_iou": 0.965
      },
      "sam2_large": {
        "area_px": 71256,
        "pred_iou": 0.965
      },
      "bplus_vs_large_iou": 0.988
    }
  ]
}
```

追加フィールド：
- `sam2_bplus/large.area_px`: セグメントされた領域のピクセル数
- `sam2_bplus/large.pred_iou`: SAM2の予測IoUスコア
- `bplus_vs_large_iou`: 2つのモデル間のマスク一致度

## プロジェクト構成

```
qwen_bbx/
├── src/
│   ├── run_infer.py       # Qwen2.5-VL 実行スクリプト
│   ├── qwen_client.py     # Qwen API クライアント
│   ├── prompts.py         # プロンプト生成
│   ├── dataset_utils.py   # データセット処理ユーティリティ
│   ├── visualize.py       # 可視化機能
│   ├── run_sam2_v2.py     # SAM 2.1 実行スクリプト（NEW）
│   ├── sam2_runner.py     # SAM 2.1 推論ラッパー（NEW）
│   ├── viz_masks.py       # マスク可視化機能（NEW）
│   └── __init__.py
├── test_images/           # 入力画像ディレクトリ
├── outputs/
│   ├── json/             # Qwen検出結果JSON
│   ├── viz/              # Qwen可視化画像
│   └── sam2/             # SAM 2.1 出力（NEW）
│       ├── json/         # セグメンテーション統計
│       ├── masks/        # バイナリマスクPNG
│       └── viz/          # マスク可視化画像
├── config.yaml           # 設定ファイル
├── requirements.txt      # Python依存パッケージ
└── README.md            # このファイル
```

## 依存パッケージ

### 基本パッケージ
- OpenAI (Qwen API互換クライアント)
- OpenCV (画像処理・可視化)
- Pillow (画像読み込み)
- NumPy (数値処理)
- PyYAML (設定ファイル)
- tqdm (進捗表示)

### SAM 2.1 用追加パッケージ（オプション）
- PyTorch >= 2.5.1 (CUDA対応推奨)
- torchvision >= 0.20.1
- Hydra-core (設定管理)
- matplotlib (追加可視化機能)

PyTorch のインストール（CUDA 12.4 の場合）：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

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

### Qwen2.5-VL 関連
- APIの利用料金が発生する場合があります
- 大量の画像を処理する場合は、APIレート制限に注意してください
- 検出精度は画像の品質や照明条件に依存します

### SAM 2.1 関連
- GPUメモリを大量に使用します（large モデルは約8GB必要）
- 初回実行時はモデルのロードに時間がかかります
- bfloat16 対応GPUでの実行を推奨（RTX 30系以降）
- CPU実行も可能ですが、処理速度が大幅に低下します

## ライセンス

[ライセンス情報を記載]

## 貢献

プルリクエストや Issue の報告を歓迎します。

## 連絡先

[連絡先情報を記載]