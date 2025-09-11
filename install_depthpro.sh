#!/bin/bash
# Depth Proのインストールスクリプト

echo "================================"
echo "Depth Proインストールスクリプト"
echo "================================"

# 1. Depth Proリポジトリのクローン
if [ ! -d "ml-depth-pro" ]; then
    echo "Depth Proリポジトリをクローン中..."
    git clone https://github.com/apple/ml-depth-pro.git
else
    echo "Depth Proリポジトリは既に存在します"
fi

# 2. Depth Proのインストール
echo "Depth Proパッケージをインストール中..."
cd ml-depth-pro
pip install -e .
cd ..

# 3. チェックポイントのダウンロード（Hugging Face）
echo "モデルチェックポイントをダウンロード中..."
if [ ! -d "checkpoints" ]; then
    mkdir -p checkpoints
fi

# Hugging Face CLIを使用してダウンロード
echo "Hugging Face Hubからモデルをダウンロード..."
huggingface-cli download --local-dir checkpoints apple/DepthPro

echo ""
echo "インストール完了！"
echo "テストコマンド: python test_depthpro_implementation.py"