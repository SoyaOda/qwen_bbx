#!/usr/bin/env python3
"""
UNIMIB2016データセットのダウンロードとセットアップスクリプト
"""
import os
import zipfile
import requests
from pathlib import Path
import shutil
from tqdm import tqdm

def download_file(url, dest_path):
    """URLからファイルをダウンロード"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))

def setup_unimib2016():
    """UNIMIB2016データセットのセットアップ"""
    
    # データセットディレクトリの作成
    data_dir = Path("data/unimib2016")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("UNIMIB2016 Food Dataset Setup")
    print("=" * 50)
    
    # ResearchGateのダウンロードリンク（公式）
    # 注：実際のダウンロードにはResearchGateアカウントが必要な場合があります
    dataset_info = """
    UNIMIB2016データセットの入手方法:
    
    1. ResearchGateページにアクセス:
       https://www.researchgate.net/publication/311501638_Food_Recognition_A_New_Dataset_Experiments_and_Results
    
    2. "Dataset" セクションから以下のファイルをダウンロード:
       - UNIMIB2016.zip (画像データ)
       - UNIMIB2016_annotations.zip (セグメンテーションアノテーション)
    
    3. ダウンロードしたファイルを以下のパスに配置:
       - data/unimib2016/UNIMIB2016.zip
       - data/unimib2016/UNIMIB2016_annotations.zip
    
    注意事項:
    - Samsung Galaxy S3で撮影された1,027枚のトレイ画像
    - オリジナルJPEGにはEXIF情報が含まれている可能性が高い
    - 学術利用目的でのみ使用可能
    """
    
    print(dataset_info)
    
    # ローカルにzipファイルがあるか確認
    zip_path = data_dir / "UNIMIB2016.zip"
    annotations_zip_path = data_dir / "UNIMIB2016_annotations.zip"
    
    if zip_path.exists():
        print(f"\n✓ Found: {zip_path}")
        print("Extracting dataset...")
        
        # 解凍
        extract_dir = data_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print(f"Dataset extracted to: {extract_dir}")
        
        # アノテーションファイルも存在すれば解凍
        if annotations_zip_path.exists():
            print(f"\n✓ Found: {annotations_zip_path}")
            print("Extracting annotations...")
            with zipfile.ZipFile(annotations_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            print("Annotations extracted")
    else:
        print(f"\n✗ Not found: {zip_path}")
        print("Please download the dataset manually using the instructions above.")
        return False
    
    return True

if __name__ == "__main__":
    success = setup_unimib2016()
    if success:
        print("\n✅ Setup completed successfully!")
    else:
        print("\n⚠️ Manual download required. Please follow the instructions above.")