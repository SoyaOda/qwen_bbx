#!/bin/bash

# Nutrition5k データセットダウンロード用 gcloud CLI インストールスクリプト
# WSL2/Ubuntu環境用

echo "=========================================="
echo "gcloud CLI インストールスクリプト"
echo "=========================================="
echo ""

# OSの検出
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✓ Linux環境を検出しました"
    
    # WSL2の検出
    if grep -qEi "(microsoft|WSL)" /proc/version; then
        echo "✓ WSL2環境を検出しました"
    fi
    
    # パッケージ管理ツールの確認
    if command -v apt-get &> /dev/null; then
        echo "✓ apt-getが利用可能です"
        INSTALLER="apt"
    else
        echo "⚠️  このスクリプトはDebian/Ubuntu系のみ対応しています"
        exit 1
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "✓ macOS環境を検出しました"
    
    # Homebrewの確認
    if command -v brew &> /dev/null; then
        echo "✓ Homebrewが利用可能です"
        echo ""
        echo "macOSの場合は以下のコマンドでインストールしてください:"
        echo "  brew install --cask google-cloud-sdk"
        exit 0
    else
        echo "⚠️  Homebrewがインストールされていません"
        echo "まず Homebrew をインストールしてください:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
else
    echo "⚠️  サポートされていないOS: $OSTYPE"
    exit 1
fi

echo ""
echo "📦 gcloud CLI をインストールします..."
echo ""

# 必要なパッケージのインストール
echo "1. 必要なパッケージをインストール中..."
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates gnupg curl

# Google Cloud の公開鍵を追加
echo ""
echo "2. Google Cloud の公開鍵を追加中..."
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

# gcloud リポジトリを追加
echo ""
echo "3. gcloud リポジトリを追加中..."
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# gcloud CLI をインストール
echo ""
echo "4. gcloud CLI をインストール中..."
sudo apt-get update
sudo apt-get install -y google-cloud-cli

# インストールの確認
echo ""
echo "5. インストールを確認中..."
if command -v gcloud &> /dev/null; then
    echo "✅ gcloud CLI のインストールが完了しました！"
    echo ""
    gcloud --version
    echo ""
    
    # gsutil の確認
    if command -v gsutil &> /dev/null; then
        echo "✅ gsutil も利用可能です"
        gsutil version -l | head -n 1
    fi
    
    echo ""
    echo "=========================================="
    echo "インストール完了！"
    echo "=========================================="
    echo ""
    echo "次のステップ:"
    echo "1. gcloud の初期設定（オプション）:"
    echo "   gcloud init"
    echo ""
    echo "2. 認証なしで公開バケットにアクセス可能ですが、"
    echo "   初期設定をしておくと安定します:"
    echo "   gcloud auth login"
    echo ""
    echo "3. Nutrition5k データセットのダウンロード:"
    echo "   cd nutrition5k"
    echo "   python download_nutrition5k.py --subset metadata  # テスト用"
    echo "   python download_nutrition5k.py  # フルダウンロード"
    echo ""
else
    echo "❌ インストールに失敗しました"
    echo "手動でインストールしてください:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi