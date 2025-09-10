# Qwen2.5-VL BBX + SAM 2.1 統合推論システム

Qwen2.5-VL による食品検出と SAM 2.1 による高精度セグメンテーションの統合システム

## 🌟 概要

このプロジェクトは、最先端のビジョンAIモデルを組み合わせた食品認識システムです：
1. **Qwen2.5-VL-72B**: 食品・食材の検出とバウンディングボックス生成
2. **SAM 2.1 (base_plus/large)**: 検出結果を基にした高精度インスタンスセグメンテーション

## 🚀 主な機能

### Qwen2.5-VL による物体検出
- 🍱 食品・食材の自動検出（英語/日本語ラベル対応）
- 📦 正規化バウンディングボックス座標 (xyxy形式)
- 📊 信頼度スコア付き検出結果
- 🎨 検出結果の可視化
- 💾 構造化JSONでの結果保存

### SAM 2.1 統合セグメンテーション
- 🎯 Qwen検出結果を入力とした精密なマスク生成
- 🔍 2モデル比較（base_plus: 高速版 vs large: 高精度版）
- 📈 セグメンテーション統計（面積、IoUスコア）
- 🖼️ マスクの多様な可視化（個別/比較/差分）
- 💾 バイナリマスクPNG出力

## 📋 動作確認済み環境

- **OS**: Ubuntu/WSL2
- **Python**: 3.11.13（venv環境）
- **GPU**: CUDA 12.4対応GPU（推奨）
- **メモリ**: 16GB以上（SAM 2.1 large使用時）

## 🛠️ セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd qwen_bbx
```

### 2. Python環境の構築

```bash
# Python 3.11の仮想環境を作成（重要: Python 3.11を使用）
python3.11 -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# パッケージ管理ツールの更新
pip install --upgrade pip wheel setuptools
```

### 3. 基本パッケージのインストール

```bash
# PyTorch (CUDA 12.4対応版)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Qwen推論用パッケージ
pip install openai opencv-python pillow numpy pyyaml tqdm

# SAM2依存パッケージ
pip install hydra-core iopath packaging portalocker
```

### 4. SAM 2.1 のセットアップ

```bash
# SAM2リポジトリのクローン
cd ..  # qwen_bbxの親ディレクトリへ
git clone https://github.com/facebookresearch/sam2.git
cd sam2

# SAM2のインストール（editable install）
pip install -e .

# チェックポイントのダウンロード
cd checkpoints
./download_ckpts.sh
# または個別にダウンロード：
# wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
# wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# qwen_bbxディレクトリに戻る
cd ../../qwen_bbx
```

### 5. API キーの設定

```bash
# 環境変数として設定（一時的）
export DASHSCOPE_API_KEY="your-api-key-here"

# または ~/.bashrc に追加（永続的）
echo 'export DASHSCOPE_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 6. 設定ファイルの編集

`config.yaml` を環境に合わせて編集：

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
  conf_threshold: 0.20       # 検出の信頼度閾値
  iou_merge_threshold: 0.0   # IoU結合閾値（0=結合なし）

dataset:
  input_dir: "test_images"     # 入力画像ディレクトリ
  out_json_dir: "outputs/json" # Qwen出力JSON保存先
  out_viz_dir: "outputs/viz"   # Qwen可視化画像保存先

# SAM2 設定
sam2:
  repo_root: "/home/soya/sam2_1_food_finetuning/external/sam2"  # SAM2パス（絶対パス推奨）
  cfg_base_plus: "sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"
  cfg_large: "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
  ckpt_base_plus: "/home/soya/sam2_1_food_finetuning/external/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
  ckpt_large: "/home/soya/sam2_1_food_finetuning/external/sam2/checkpoints/sam2.1_hiera_large.pt"
  device: "cuda"                # "cuda" or "cpu"
  dtype: "bfloat16"            # "bfloat16"(GPU) / "float32"(CPU)
  multimask_output: true       # 複数マスク仮説から最良を選択
  conf_threshold: 0.20         # Qwen検出の信頼度閾値

# SAM2処理用パス
paths:
  qwen_json_dir: "outputs/json"    # Qwen出力JSONの読み込み元
  input_dir: "test_images"         # 入力画像ディレクトリ
  out_root: "outputs/sam2"         # SAM2出力のルートディレクトリ
```

## 🎯 使用方法

### 重要: Python 3.11 を使用

このプロジェクトは **Python 3.11** で動作確認済みです。venv環境には複数のPythonバージョンが存在する場合があるため、必ず以下のように実行してください：

```bash
# 環境のアクティベート
source venv/bin/activate

# APIキーの設定（初回のみ）
export DASHSCOPE_API_KEY="your-api-key-here"

# 正しいPythonバージョンの確認
venv/bin/python3.11 --version  # Python 3.11.13 と表示されるはず
```

### ステップ1: Qwen2.5-VL による食品検出

```bash
# 画像から食品を検出してバウンディングボックスを生成
venv/bin/python3.11 src/run_infer.py

# 出力:
# - outputs/json/*.json  : 検出結果のJSON
# - outputs/viz/*.jpg   : バウンディングボックス付き画像
```

### ステップ2: SAM 2.1 によるセグメンテーション

```bash
# Qwenの検出結果を基にセグメンテーションマスクを生成
venv/bin/python3.11 src/run_sam2_v2.py

# 出力:
# - outputs/sam2/json/*.json     : セグメンテーション統計
# - outputs/sam2/masks/*.png     : バイナリマスク画像
# - outputs/sam2/viz/*_bplus.jpg : base_plusモデルの結果
# - outputs/sam2/viz/*_large.jpg : largeモデルの結果
# - outputs/sam2/viz/*_panel.jpg : 4分割比較画像
```

### 一括実行スクリプト

```bash
# 両方のステップを順番に実行
venv/bin/python3.11 src/run_infer.py && venv/bin/python3.11 src/run_sam2_v2.py
```

## 📁 入出力仕様

### 入力画像の準備

```bash
test_images/
├── image1.jpg
├── image2.png
└── ...
```

- **対応形式**: JPG, JPEG, PNG, BMP, GIF, TIFF, WEBP
- **推奨サイズ**: 512x512 〜 1920x1080
- **配置場所**: `test_images/` ディレクトリ

### 出力ファイル構成

```
outputs/
├── json/                    # Qwen2.5-VL 検出結果
│   ├── image1.json         # 検出物体のBBox + ラベル
│   └── ...
├── viz/                     # Qwen2.5-VL 可視化
│   ├── image1.jpg          # BBox描画済み画像
│   └── ...
└── sam2/                    # SAM 2.1 セグメンテーション結果
    ├── json/               # 統計情報
    │   ├── image1.sam2.json
    │   └── ...
    ├── masks/              # バイナリマスク（0/255 PNG）
    │   ├── image1_det00_rice_bplus.png
    │   ├── image1_det00_rice_large.png
    │   └── ...
    └── viz/                # マスク可視化
        ├── image1_bplus.jpg     # base_plusモデル結果
        ├── image1_large.jpg     # largeモデル結果
        └── image1_panel.jpg     # 4分割比較画像

## 📊 出力データ形式

### Qwen2.5-VL JSON構造

```json
{
  "detections": [
    {
      "label_en": "rice",
      "label_ja": "ご飯",  // 日本語ラベル（オプション）
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

**フィールド説明:**
- `label_en`: 英語の食品名
- `label_ja`: 日本語の食品名（設定により出力）
- `bbox_xyxy_norm`: 正規化座標 [x_min, y_min, x_max, y_max] (0-1)
- `confidence`: 検出信頼度 (0-1)

### SAM 2.1 JSON構造

```json
{
  "image": "train_00000.jpg",
  "width": 512,
  "height": 384,
  "detections": [
    {
      "id": 0,
      "label_en": "rice",
      "label_ja": "ご飯",
      "qwen_confidence": 0.95,
      "bbox_xyxy_norm": [0.32, 0.14, 1.0, 0.89],
      "bbox_xyxy_abs": [163.84, 53.76, 512.0, 341.76],
      "sam2_bplus": {
        "area_px": 71635,      // セグメント面積（ピクセル）
        "pred_iou": 0.965      // モデル予測IoU
      },
      "sam2_large": {
        "area_px": 71256,
        "pred_iou": 0.965
      },
      "bplus_vs_large_iou": 0.988  // 2モデル間の一致度
    }
  ]
}
```

**追加フィールド説明:**
- `bbox_xyxy_abs`: 絶対座標（ピクセル単位）
- `area_px`: マスク領域の面積
- `pred_iou`: SAM2モデルの自己評価スコア
- `bplus_vs_large_iou`: base_plusとlargeモデルのマスク一致度

## 🗂️ プロジェクト構成

```
qwen_bbx/
├── src/                      # ソースコード
│   ├── run_infer.py         # Qwen2.5-VL メインスクリプト
│   ├── qwen_client.py       # Qwen API クライアント
│   ├── prompts.py           # プロンプト生成
│   ├── dataset_utils.py     # データセット処理
│   ├── visualize.py         # BBox可視化
│   ├── run_sam2_v2.py       # SAM 2.1 メインスクリプト
│   ├── sam2_runner.py       # SAM 2.1 推論ラッパー
│   ├── viz_masks.py         # マスク可視化
│   └── __init__.py
├── test_images/              # 入力画像
├── outputs/                  # 出力結果
│   ├── json/                # Qwen検出JSON
│   ├── viz/                 # Qwen可視化
│   └── sam2/                # SAM2結果
│       ├── json/           # 統計情報
│       ├── masks/          # マスクPNG
│       └── viz/            # マスク可視化
├── venv/                     # Python仮想環境
├── config.yaml              # 設定ファイル
├── requirements.txt         # 依存パッケージ
├── README.md               # このファイル
└── CLAUDE.md               # AI開発ガイド

## ⚙️ 技術仕様

### モデル構成
| モデル | バージョン | 用途 | 特徴 |
|--------|-----------|------|------|
| Qwen2.5-VL | 72B-Instruct | 物体検出 | 高精度な食品認識、マルチ言語対応 |
| SAM 2.1 Base+ | 80.8M params | セグメンテーション | 高速処理（~64 FPS） |
| SAM 2.1 Large | 224.4M params | セグメンテーション | 高精度（最高品質） |

### パフォーマンス指標
- **Qwen推論**: 約10秒/画像（API経由）
- **SAM2 Base+**: 約0.5秒/画像（GPU）
- **SAM2 Large**: 約1.0秒/画像（GPU）
- **総処理時間**: 約12秒/画像（全パイプライン）

## 🔧 トラブルシューティング

### Python バージョンの問題

```bash
# 間違った例（Python 3.12が使われる）
python src/run_infer.py  # ❌ ModuleNotFoundError

# 正しい例（Python 3.11を明示的に指定）
venv/bin/python3.11 src/run_infer.py  # ✅
```

### CUDA/GPU関連

```bash
# CUDAが利用可能か確認
venv/bin/python3.11 -c "import torch; print(torch.cuda.is_available())"

# CPU実行に切り替える場合
# config.yaml を編集:
# device: "cpu"
# dtype: "float32"
```

### メモリエラー対策

```yaml
# config.yaml で調整
inference:
  max_items: 10  # 処理画像数を減らす
sam2:
  dtype: "float16"  # メモリ使用量を削減（精度は若干低下）
```

### API接続エラー

```bash
# APIキーの確認
echo $DASHSCOPE_API_KEY

# プロキシ環境の場合
export https_proxy=http://your-proxy:port
export http_proxy=http://your-proxy:port
```

## 🚀 パフォーマンス最適化

### バッチ処理の活用
```bash
# 複数画像を一括処理
find test_images -name "*.jpg" | head -100 | xargs -I {} cp {} test_images/batch/
venv/bin/python3.11 src/run_infer.py
```

### GPU最適化
```python
# config.yaml
sam2:
  dtype: "bfloat16"  # RTX 30系以降で高速化
  device: "cuda:0"   # 特定GPUを指定
```

### メモリ効率化
- 画像を事前にリサイズ（1024x1024以下推奨）
- バッチサイズを調整（GPUメモリに応じて）

## 📚 関連リソース

- [Qwen2.5-VL 公式ドキュメント](https://github.com/QwenLM/Qwen2.5-VL)
- [SAM 2 公式リポジトリ](https://github.com/facebookresearch/sam2)
- [DashScope API ガイド](https://www.alibabacloud.com/help/en/model-studio/)

## 🤝 貢献ガイドライン

1. Issue を作成して問題を報告
2. Fork してブランチを作成
3. 変更をコミット（コミットメッセージは明確に）
4. Pull Request を送信

## 📄 ライセンス

このプロジェクトは研究・教育目的で提供されています。
商用利用の場合は、各モデルのライセンスを確認してください。

## ⚠️ 注意事項

### セキュリティ
- APIキーを公開リポジトリにコミットしない
- 機密画像の処理時は適切なセキュリティ対策を実施

### 制限事項
- Qwen API: レート制限あり（約100リクエスト/分）
- SAM2: GPUメモリ8GB以上推奨
- 大量画像処理時はバッチ処理を推奨

### データプライバシー
- 処理画像はローカル保存のみ
- API経由でQwenに送信される画像に注意

---

*最終更新: 2025年1月*