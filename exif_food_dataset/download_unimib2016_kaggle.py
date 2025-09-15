#!/usr/bin/env python3
"""
UNIMIB2016データセットをKaggleからダウンロードするスクリプト
"""
import os
import json
import zipfile
from pathlib import Path
import subprocess
import sys

def check_kaggle_installed():
    """Kaggle APIがインストールされているかチェック"""
    try:
        # kaggleインポート時の認証を避ける
        import sys
        import importlib.util
        spec = importlib.util.find_spec("kaggle")
        return spec is not None
    except ImportError:
        return False

def install_kaggle_api():
    """Kaggle APIをインストール"""
    print("Installing Kaggle API...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

def check_kaggle_credentials():
    """Kaggle認証情報があるかチェック"""
    kaggle_json_1 = Path.home() / ".kaggle" / "kaggle.json"
    kaggle_json_2 = Path.home() / ".config" / "kaggle" / "kaggle.json"
    
    if not kaggle_json_1.exists() and not kaggle_json_2.exists():
        print("=" * 60)
        print("Kaggle API認証設定が必要です")
        print("=" * 60)
        print("""
1. Kaggleにログイン: https://www.kaggle.com
2. アカウント設定ページ: https://www.kaggle.com/settings
3. 'API' セクションで 'Create New API Token' をクリック
4. kaggle.json がダウンロードされます
5. 以下のコマンドを実行:
   mkdir -p ~/.config/kaggle
   cp ~/Downloads/kaggle.json ~/.config/kaggle/
   chmod 600 ~/.config/kaggle/kaggle.json
   
   または
   
   mkdir -p ~/.kaggle
   cp ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
""")
        return False
    return True

def download_unimib2016_from_kaggle():
    """KaggleからUNIMIB2016データセットをダウンロード"""
    
    # Kaggle APIのインストール確認
    if not check_kaggle_installed():
        install_kaggle_api()
    
    # 認証情報の確認
    if not check_kaggle_credentials():
        return False
    
    # データディレクトリの作成
    data_dir = Path("data/unimib2016_kaggle")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("UNIMIB2016データセットをKaggleからダウンロード")
    print("=" * 60)
    print("Dataset: dangvanthuc0209/unimib2016")
    print(f"Download path: {data_dir}")
    print("-" * 60)
    
    # Kaggle APIを使用してダウンロード
    import kaggle
    
    try:
        # データセットのダウンロード
        kaggle.api.dataset_download_files(
            dataset="dangvanthuc0209/unimib2016",
            path=str(data_dir),
            unzip=True
        )
        print("✅ ダウンロード完了")
        
        # ダウンロードされたファイルの確認
        print("\n📁 ダウンロードされたファイル:")
        for item in data_dir.rglob("*"):
            if item.is_file():
                size_mb = item.stat().st_size / (1024 * 1024)
                print(f"  - {item.relative_to(data_dir)}: {size_mb:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        print("\n代替手段:")
        print("1. ブラウザで直接ダウンロード:")
        print("   https://www.kaggle.com/datasets/dangvanthuc0209/unimib2016")
        print("2. Kaggle CLIでダウンロード:")
        print("   kaggle datasets download -d dangvanthuc0209/unimib2016")
        return False

def main():
    """メイン処理"""
    success = download_unimib2016_from_kaggle()
    
    if success:
        print("\n✅ セットアップ完了！")
        print("次のステップ: EXIF情報の確認スクリプトを実行してください")
    else:
        print("\n⚠️ 手動でのダウンロードが必要です")

if __name__ == "__main__":
    main()