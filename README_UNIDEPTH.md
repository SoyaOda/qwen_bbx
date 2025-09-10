# UniDepth v2 統合版 - 食品体積推定システム

Qwen2.5-VL → SAM 2.1 → UniDepth v2 を組み合わせた食品検出・セグメンテーション・体積推定の統合システム

## 🎯 システム概要

本システムは3つの最先端AIモデルを統合した食品分析パイプラインです：

1. **Qwen2.5-VL-72B**: 食品の検出とバウンディングボックス生成
2. **SAM 2.1**: 高精度なインスタンスセグメンテーション
3. **UniDepth v2**: メトリック深度推定と体積計算

### 主な機能
- 📦 食品の自動検出とラベリング（日英対応）
- 🎯 ピクセル単位の精密なマスク生成
- 📏 メトリック深度推定（メートル単位）
- 🏔️ RANSAC平面フィッティングによる皿/卓面推定
- 📊 各食品の体積・高さの自動計算
- 🎨 深度・高さマップの可視化

## 🛠️ セットアップ

### 1. 環境要件

- **OS**: Ubuntu 20.04+ / WSL2
- **Python**: 3.11.13（厳密にこのバージョンを使用）
- **GPU**: CUDA 12.4対応GPU（8GB VRAM以上推奨）
- **メモリ**: 16GB以上

### 2. リポジトリのクローン

```bash
git clone <repository-url>
cd qwen_bbx
```

### 3. Python環境の構築

```bash
# Python 3.11の仮想環境を作成（重要）
python3.11 -m venv venv
source venv/bin/activate  # Windowsの場合: venv\Scripts\activate

# パッケージ管理ツールの更新
pip install --upgrade pip wheel setuptools
```

### 4. 基本パッケージのインストール

```bash
# PyTorch (CUDA 12.4対応版) - 既にインストール済みの場合はスキップ
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 基本パッケージ
pip install openai opencv-python pillow numpy pyyaml tqdm
pip install hydra-core iopath packaging portalocker
pip install matplotlib scipy pandas
```

### 5. UniDepth v2 のセットアップ

```bash
# UniDepthリポジトリのクローン（qwen_bbxの親ディレクトリに配置）
cd ..
git clone https://github.com/lpiccinelli-eth/UniDepth.git
cd UniDepth

# UniDepthのインストール（editable install）
/home/soya/qwen_bbx/venv/bin/pip install -e . --extra-index-url https://download.pytorch.org/whl/cu118

# qwen_bbxディレクトリに戻る
cd ../qwen_bbx
```

### 6. SAM 2.1 のセットアップ

```bash
# SAM2リポジトリのクローン（既存の場合はスキップ）
cd ..
git clone https://github.com/facebookresearch/sam2.git sam2_1_food_finetuning/external/sam2
cd sam2_1_food_finetuning/external/sam2

# チェックポイントのダウンロード
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# qwen_bbxに戻る
cd /home/soya/qwen_bbx
```

### 7. API キーの設定

```bash
# DashScope APIキーを環境変数に設定
export DASHSCOPE_API_KEY="your-api-key-here"

# 永続化する場合
echo 'export DASHSCOPE_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 8. 設定ファイルの確認

`config.yaml`を環境に合わせて編集（特にパスの設定）：

```yaml
# SAM2のパス（絶対パスで指定）
sam2:
  repo_root: "/home/soya/sam2_1_food_finetuning/external/sam2"
  ckpt_base_plus: "/home/soya/sam2_1_food_finetuning/external/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
  ckpt_large: "/home/soya/sam2_1_food_finetuning/external/sam2/checkpoints/sam2.1_hiera_large.pt"

# UniDepth設定
unidepth:
  model_repo: "lpiccinelli/unidepth-v2-vitl14"
  device: "cuda"  # GPUがない場合は "cpu"
```

## 🚀 使用方法

### 重要: Python 3.11を明示的に使用

```bash
# 環境のアクティベート
source venv/bin/activate

# Pythonバージョンの確認
venv/bin/python3.11 --version  # Python 3.11.13 と表示されるはず
```

### 完全パイプラインの実行

#### ステップ1: Qwen2.5-VLによる食品検出

```bash
venv/bin/python3.11 src/run_infer.py
```

**出力:**
- `outputs/json/*.json` - 検出結果（バウンディングボックス、ラベル、信頼度）
- `outputs/viz/*.jpg` - バウンディングボックス付き画像

#### ステップ2: SAM 2.1によるセグメンテーション

```bash
venv/bin/python3.11 src/run_sam2_v2.py
```

**出力:**
- `outputs/sam2/json/*.sam2.json` - セグメンテーション統計
- `outputs/sam2/masks/*.png` - バイナリマスク画像
- `outputs/sam2/viz/*_panel.jpg` - 4分割比較画像

#### ステップ3: UniDepth v2による深度推定と体積計算

```bash
venv/bin/python3.11 src/run_unidepth.py
```

**出力:**
- `outputs/unidepth/depth/*.npy` - 深度マップ（メートル単位）
- `outputs/unidepth/height/*.npy` - 高さマップ
- `outputs/unidepth/viz/*_panel.jpg` - 3分割パネル（原画像｜深度｜高さ）
- `outputs/unidepth/json/*.unidepth.json` - 体積・平面・統計情報

### 一括実行

```bash
# 3つのステップを順番に実行
venv/bin/python3.11 src/run_infer.py && \
venv/bin/python3.11 src/run_sam2_v2.py && \
venv/bin/python3.11 src/run_unidepth.py
```

## 📊 出力データ形式

### UniDepth JSON構造

```json
{
  "image": "train_00000.jpg",
  "width": 512,
  "height": 384,
  "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "plane": {
    "normal": [nx, ny, nz],  // 平面の法線ベクトル
    "d": -1.279              // 平面方程式の定数項
  },
  "detections": [
    {
      "id": 0,
      "label": "rice",
      "pixels": 39643,
      "volume_mL": 60291.9,        // 体積（ミリリットル）
      "height_mean_mm": 152.0,     // 平均高さ（ミリメートル）
      "height_max_mm": 244.4       // 最大高さ（ミリメートル）
    }
  ]
}
```

## 🔧 トラブルシューティング

### Python バージョンエラー

```bash
# 問題: ModuleNotFoundError
# 解決: Python 3.11を明示的に使用
venv/bin/python3.11 src/run_unidepth.py  # ✅ 正しい
python src/run_unidepth.py               # ❌ 間違い
```

### UniDepthインポートエラー

```bash
# エラー: ImportError: cannot import name 'UniDepthV2'
# 解決: UniDepthを再インストール
cd ../UniDepth
/home/soya/qwen_bbx/venv/bin/pip install -e .
```

### CUDA/GPUエラー

```bash
# GPUが利用可能か確認
venv/bin/python3.11 -c "import torch; print(torch.cuda.is_available())"

# CPUモードに切り替える場合
# config.yaml を編集:
unidepth:
  device: "cpu"
```

### メモリ不足エラー

```yaml
# config.yaml で調整
inference:
  max_items: 5  # 処理画像数を減らす
```

## 📁 プロジェクト構成

```
qwen_bbx/
├── src/
│   ├── run_infer.py         # Qwen2.5-VL メイン
│   ├── run_sam2_v2.py       # SAM 2.1 メイン
│   ├── run_unidepth.py      # UniDepth v2 メイン ★
│   ├── unidepth_runner.py   # UniDepth推論エンジン ★
│   ├── plane_fit.py         # RANSAC平面フィッティング ★
│   ├── volume_estimator.py  # 体積推定 ★
│   ├── vis_depth.py         # 深度可視化（改良版） ★
│   └── ...
├── test_images/             # 入力画像
├── outputs/
│   ├── json/               # Qwen検出結果
│   ├── viz/                # Qwen可視化
│   ├── sam2/               # SAM2結果
│   └── unidepth/           # UniDepth結果 ★
│       ├── depth/         # 深度マップ
│       ├── height/        # 高さマップ
│       ├── viz/           # 可視化
│       └── json/          # 体積・統計
├── config.yaml             # 設定ファイル
├── README.md              # 基本README
└── README_UNIDEPTH.md     # このファイル ★
```

## 🧪 テストスクリプト

### UniDepth単体テスト

```bash
# 基本的な深度推定テスト
venv/bin/python3.11 src/test_unidepth_simple.py

# depth_modelプロジェクトと同じビジュアライゼーション
venv/bin/python3.11 src/test_unidepth_depth_model_style.py
```

### 正規化の比較テスト

```bash
# ImageNet正規化あり/なしの比較
venv/bin/python3.11 src/test_unidepth_normalized.py
```

## ⚙️ 技術仕様

### モデル詳細

| モデル | バージョン | 用途 | 特徴 |
|--------|-----------|------|------|
| Qwen2.5-VL | 72B-Instruct | 物体検出 | 高精度な食品認識 |
| SAM 2.1 | Large | セグメンテーション | ピクセル単位のマスク |
| UniDepth v2 | ViT-L14 | 深度推定 | メトリック深度・内部パラメータ推定 |

### 重要な実装詳細

1. **ImageNet正規化**: UniDepth v2の推論時に必須
   ```python
   normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
   )
   ```

2. **深度の可視化**: 逆数変換で「近い=明るい」表示
   ```python
   display_depth = 1.0 / (np.maximum(depth, 1e-6))
   ```

3. **体積計算式**: 
   ```
   V = Σ h(x,y) × (z²)/(fx×fy)
   ```
   ここで h(x,y) は平面からの高さ、z は深度、fx,fy は焦点距離

## 📈 パフォーマンス

- **Qwen推論**: 約10秒/画像（API経由）
- **SAM2推論**: 約1秒/画像（GPU）
- **UniDepth推論**: 約0.5秒/画像（GPU）
- **総処理時間**: 約12秒/画像（全パイプライン）

## ⚠️ 注意事項

### 体積推定の精度について
現在の実装では体積が実際より大きく推定される傾向があります。これは以下の要因が考えられます：
- カメラキャリブレーションの精度
- 平面推定の誤差
- マスク境界の不正確さ

実用化には追加のキャリブレーションが必要です。

### ライセンス
- UniDepth: CC BY-NC 4.0（非商用）
- SAM 2: Apache 2.0
- Qwen: 商用利用にはライセンス確認が必要

## 🤝 貢献・改善点

- [ ] 体積推定の精度向上（キャリブレーション）
- [ ] リアルタイム処理の最適化
- [ ] バッチ処理の並列化
- [ ] WebUIの追加

---

## 重要な技術的修正

### カメラパラメータのスケーリング問題と解決策

**問題**: UniDepth v2が推定するカメラ内部パラメータは小さすぎることがあり、体積が異常に大きくなる
- 例：ご飯一杯が60リットル（正常値の約200倍）

**原因**: 推定されるfx,fyが約550-700（正常値は3000-4000）

**解決策**: `config.yaml`で`K_scale_factor: 6.0`を設定

**調整方法**:
- 体積が大きすぎる → K_scale_factorを増やす
- 体積が小さすぎる → K_scale_factorを減らす
- 目安：食品一人前は100-500mL

詳細は`SOLUTION_VOLUME_ISSUE.md`を参照。

*最終更新: 2025年1月*
*作成: Claude Code (claude.ai/code)*