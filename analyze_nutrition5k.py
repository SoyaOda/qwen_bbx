#!/usr/bin/env python3
"""
Nutrition5kデータセットの内容を分析
"""
import os
import pandas as pd
from pathlib import Path
import json

def analyze_nutrition5k():
    """Nutrition5kデータセットの詳細分析"""
    
    base_path = Path("nutrition5k/nutrition5k_dataset")
    
    print("=" * 80)
    print("Nutrition5k データセット分析")
    print("=" * 80)
    
    # 1. ディレクトリ構造
    print("\n📁 ディレクトリ構造:")
    dirs = {
        "dish_ids": "料理ID情報",
        "metadata": "メタデータ（栄養情報等）",
        "imagery": "画像データ",
        "scripts": "処理スクリプト"
    }
    for dir_name, desc in dirs.items():
        dir_path = base_path / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name:15s} - {desc}")
    
    # 2. 画像データの種類
    print("\n🖼️ 画像データの種類:")
    imagery_path = base_path / "imagery"
    if imagery_path.exists():
        for subdir in imagery_path.iterdir():
            if subdir.is_dir():
                count = len(list(subdir.iterdir()))
                print(f"  - {subdir.name}: {count}件")
                
                # サンプルディレクトリの中身を確認
                if count > 0:
                    sample_dir = list(subdir.iterdir())[0]
                    if sample_dir.is_dir():
                        files = list(sample_dir.iterdir())
                        print(f"    サンプル ({sample_dir.name}):")
                        for f in files[:5]:
                            print(f"      - {f.name}")
    
    # 3. メタデータの分析
    print("\n📊 メタデータファイル:")
    metadata_path = base_path / "metadata"
    if metadata_path.exists():
        for file in metadata_path.iterdir():
            if file.suffix == '.csv':
                print(f"\n  📄 {file.name}:")
                try:
                    # CSVの最初の行を読んで構造を理解
                    with open(file, 'r') as f:
                        first_line = f.readline().strip()
                        
                    # 栄養情報の場合
                    if 'dish_metadata' in file.name:
                        # 複雑な構造のため、特殊な処理
                        print("    形式: 料理ごとの栄養情報と材料リスト")
                        print("    内容例:")
                        print("      - dish_id")
                        print("      - total_calories")
                        print("      - total_mass (g)")
                        print("      - total_fat (g)")
                        print("      - total_carb (g)")
                        print("      - total_protein (g)")
                        print("      - 各材料の詳細情報")
                        
                        # データ件数を数える
                        with open(file, 'r') as f:
                            lines = f.readlines()
                        print(f"    データ件数: {len(lines)}件")
                        
                    elif 'ingredients' in file.name:
                        df = pd.read_csv(file)
                        print(f"    列: {df.columns.tolist()}")
                        print(f"    データ件数: {len(df)}件")
                        
                except Exception as e:
                    print(f"    読み込みエラー: {e}")
    
    # 4. dish_idsの確認
    print("\n🍽️ Dish IDファイル:")
    dish_ids_path = base_path / "dish_ids"
    if dish_ids_path.exists():
        for file in dish_ids_path.iterdir():
            if file.suffix == '.txt':
                with open(file, 'r') as f:
                    ids = f.readlines()
                print(f"  - {file.name}: {len(ids)}件")
        
        # splitsディレクトリ
        splits_path = dish_ids_path / "splits"
        if splits_path.exists():
            print("\n  📂 データ分割 (splits):")
            for file in splits_path.iterdir():
                if file.suffix == '.txt':
                    with open(file, 'r') as f:
                        ids = f.readlines()
                    print(f"    - {file.name}: {len(ids)}件")
    
    # 5. 深度データの確認
    print("\n🔍 深度データの確認:")
    sample_dishes = list((imagery_path / "realsense_overhead").iterdir())[:3]
    for dish_dir in sample_dishes:
        if dish_dir.is_dir():
            print(f"  {dish_dir.name}:")
            for file in dish_dir.iterdir():
                print(f"    - {file.name}")
                if file.name == "depth_raw.png":
                    print("      ✓ 生の深度データあり")
                elif file.name == "depth_color.png":
                    print("      ✓ カラー深度マップあり")
                elif file.name == "rgb.png":
                    print("      ✓ RGBカメラ画像あり")
    
    # 6. マスクデータの確認
    print("\n🎭 マスク/セグメンテーションデータ:")
    mask_found = False
    for root, dirs, files in os.walk(str(base_path)):
        for file in files:
            if 'mask' in file.lower() or 'segment' in file.lower() or 'annotation' in file.lower():
                print(f"  ✓ 発見: {os.path.relpath(os.path.join(root, file), base_path)}")
                mask_found = True
    if not mask_found:
        print("  ❌ マスク/セグメンテーションデータは見つかりませんでした")
    
    # 7. 体積データの確認
    print("\n📏 体積データの確認:")
    print("  メタデータファイルを分析中...")
    
    # dish_metadataから体積関連情報を探す
    for cafe in ['cafe1', 'cafe2']:
        metadata_file = metadata_path / f"dish_metadata_{cafe}.csv"
        if metadata_file.exists():
            print(f"\n  {metadata_file.name}:")
            with open(metadata_file, 'r') as f:
                first_line = f.readline()
            
            # 最初の行から情報を抽出
            parts = first_line.split(',')
            if len(parts) > 6:
                print(f"    - dish_id: {parts[0]}")
                print(f"    - total_calories: {parts[1]}")
                print(f"    - total_mass (g): {parts[2]} ← 重量データあり")
                print(f"    - total_fat (g): {parts[3]}")
                print(f"    - total_carb (g): {parts[4]}")
                print(f"    - total_protein (g): {parts[5]}")
                print("    ※ 直接の体積(mL)データはなし、質量(g)のみ")

if __name__ == "__main__":
    analyze_nutrition5k()