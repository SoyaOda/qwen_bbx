#!/usr/bin/env python3
"""
GitHub リポジトリからUNIMIB2016のサンプル画像を取得
"""
import os
import requests
from pathlib import Path
import json

def download_github_samples():
    """GitHubからサンプル画像を取得"""
    
    # データディレクトリの作成
    data_dir = Path("data/github_samples")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("GitHubリポジトリからサンプル画像を取得")
    print("=" * 60)
    
    # ivanDonadello/Food-Categories-Classification リポジトリの情報
    repo_info = {
        "owner": "ivanDonadello",
        "repo": "Food-Categories-Classification",
        "branch": "master"
    }
    
    # GitHubのAPIでリポジトリ構造を確認
    api_url = f"https://api.github.com/repos/{repo_info['owner']}/{repo_info['repo']}/contents"
    
    try:
        print(f"リポジトリ: {repo_info['owner']}/{repo_info['repo']}")
        print("構造を確認中...")
        
        response = requests.get(api_url)
        if response.status_code == 200:
            contents = response.json()
            
            # リポジトリの内容を表示
            print("\n📁 リポジトリ構造:")
            for item in contents:
                print(f"  - {item['name']} ({item['type']})")
                
                # datasetsフォルダを探す
                if item['name'] == 'datasets' and item['type'] == 'dir':
                    # datasetsの中身を確認
                    datasets_url = item['url']
                    datasets_response = requests.get(datasets_url)
                    if datasets_response.status_code == 200:
                        datasets_contents = datasets_response.json()
                        print(f"\n    📁 datasets/ の内容:")
                        for dataset_item in datasets_contents:
                            print(f"      - {dataset_item['name']}")
            
            # READMEから情報を取得
            readme_url = f"https://raw.githubusercontent.com/{repo_info['owner']}/{repo_info['repo']}/{repo_info['branch']}/README.md"
            readme_response = requests.get(readme_url)
            if readme_response.status_code == 200:
                readme_content = readme_response.text
                # READMEをローカルに保存
                readme_path = data_dir / "README_github.md"
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(readme_content)
                print(f"\n✅ README保存: {readme_path}")
                
                # READMEの最初の部分を表示
                lines = readme_content.split('\n')
                print("\n📄 READMEの概要:")
                for line in lines[:20]:
                    if line.strip():
                        print(f"  {line[:100]}")
        else:
            print(f"❌ リポジトリアクセスエラー: {response.status_code}")
            
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    print("\n" + "=" * 60)
    print("UNIMIB2016データセットのダウンロード手順:")
    print("=" * 60)
    print("""
手動ダウンロード方法:

1. Kaggleから直接ダウンロード:
   https://www.kaggle.com/datasets/dangvanthuc0209/unimib2016
   
2. ResearchGateから学術版をダウンロード:
   https://www.researchgate.net/publication/311501638_Food_Recognition_A_New_Dataset_Experiments_and_Results

3. ダウンロード後、以下の場所に配置:
   exif_food_dataset/data/unimib2016/
   
データセット情報:
- 1,027枚のトレイ画像
- 73種類の食品カテゴリ
- Samsung Galaxy S3で撮影
- セグメンテーションアノテーション付き
""")

if __name__ == "__main__":
    download_github_samples()