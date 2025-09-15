#!/usr/bin/env python3
"""
画像ファイルのEXIF情報を確認するスクリプト
"""
import os
import json
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import sys
from datetime import datetime

def get_exif_data(image_path):
    """画像からEXIF情報を取得"""
    try:
        image = Image.open(image_path)
        exifdata = image.getexif()
        
        if not exifdata:
            return None
        
        # EXIF情報を読みやすい形式に変換
        exif_dict = {}
        for tag_id, value in exifdata.items():
            tag = TAGS.get(tag_id, tag_id)
            exif_dict[tag] = value
            
        return exif_dict
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def analyze_exif_for_volume_estimation(exif_data):
    """体積推定に有用なEXIF情報を抽出"""
    if not exif_data:
        return None
    
    useful_tags = {
        'Make': 'カメラメーカー',
        'Model': 'カメラモデル',
        'FocalLength': '焦点距離',
        'FocalLengthIn35mmFilm': '35mm換算焦点距離',
        'FNumber': 'F値',
        'ExposureTime': '露出時間',
        'ISOSpeedRatings': 'ISO感度',
        'DateTimeOriginal': '撮影日時',
        'Orientation': '画像方向',
        'XResolution': 'X解像度',
        'YResolution': 'Y解像度',
        'ResolutionUnit': '解像度単位',
        'PixelXDimension': 'ピクセル幅',
        'PixelYDimension': 'ピクセル高さ',
        'ImageWidth': '画像幅',
        'ImageLength': '画像高さ',
        'GPSInfo': 'GPS情報'
    }
    
    result = {}
    for tag, description in useful_tags.items():
        if tag in exif_data:
            value = exif_data[tag]
            # 特殊な値の処理
            if tag == 'FocalLength' and hasattr(value, 'real'):
                value = float(value.real)
            elif tag == 'DateTimeOriginal':
                try:
                    value = str(value)
                except:
                    pass
            result[description] = value
    
    return result

def check_dataset_exif(data_dir):
    """データセットディレクトリのEXIF情報をチェック"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ ディレクトリが存在しません: {data_dir}")
        print("\n先にデータセットをダウンロードしてください:")
        print("1. Kaggle: https://www.kaggle.com/datasets/dangvanthuc0209/unimib2016")
        print("2. ResearchGate: 論文ページからダウンロード")
        return
    
    # 画像ファイルを検索
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(data_path.rglob(f'*{ext}'))
        image_files.extend(data_path.rglob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ 画像ファイルが見つかりません: {data_dir}")
        return
    
    print(f"📁 検査対象: {data_dir}")
    print(f"🖼️ 画像ファイル数: {len(image_files)}")
    print("=" * 60)
    
    # EXIF情報の統計
    exif_count = 0
    focal_length_count = 0
    camera_model_count = 0
    orientation_count = 0
    gps_count = 0
    
    # サンプル表示用
    sample_shown = False
    
    for i, img_path in enumerate(image_files[:100]):  # 最初の100枚をチェック
        exif_data = get_exif_data(img_path)
        
        if exif_data:
            exif_count += 1
            useful_exif = analyze_exif_for_volume_estimation(exif_data)
            
            if useful_exif:
                if '焦点距離' in useful_exif:
                    focal_length_count += 1
                if 'カメラモデル' in useful_exif:
                    camera_model_count += 1
                if '画像方向' in useful_exif:
                    orientation_count += 1
                if 'GPS情報' in useful_exif:
                    gps_count += 1
                
                # 最初のEXIF付き画像の詳細を表示
                if not sample_shown:
                    print(f"\n📷 サンプル画像のEXIF情報:")
                    print(f"   ファイル: {img_path.name}")
                    print(f"   パス: {img_path}")
                    print("\n   体積推定に有用な情報:")
                    for key, value in useful_exif.items():
                        print(f"     {key}: {value}")
                    print("\n   全EXIF情報:")
                    for key, value in list(exif_data.items())[:10]:
                        print(f"     {key}: {value}")
                    print("=" * 60)
                    sample_shown = True
    
    # 統計結果
    checked_count = min(100, len(image_files))
    print(f"\n📊 EXIF情報の統計 (検査画像数: {checked_count})")
    print(f"   EXIF情報あり: {exif_count}/{checked_count} ({exif_count/checked_count*100:.1f}%)")
    print(f"   焦点距離情報: {focal_length_count}/{checked_count} ({focal_length_count/checked_count*100:.1f}%)")
    print(f"   カメラモデル: {camera_model_count}/{checked_count} ({camera_model_count/checked_count*100:.1f}%)")
    print(f"   画像方向: {orientation_count}/{checked_count} ({orientation_count/checked_count*100:.1f}%)")
    print(f"   GPS情報: {gps_count}/{checked_count} ({gps_count/checked_count*100:.1f}%)")
    
    # 結果の評価
    print("\n🔍 評価:")
    if exif_count > checked_count * 0.8:
        print("   ✅ このデータセットはEXIF情報が豊富です")
        if focal_length_count > checked_count * 0.5:
            print("   ✅ 焦点距離情報があるため、体積推定に有用です")
    elif exif_count > checked_count * 0.5:
        print("   ⚠️ EXIF情報は部分的に存在します")
    else:
        print("   ❌ EXIF情報がほとんどありません")
        print("   💡 オリジナル画像の入手を検討してください")

def main():
    """メイン処理"""
    print("=" * 60)
    print("EXIF情報チェックツール")
    print("=" * 60)
    
    # チェック対象ディレクトリ
    check_dirs = [
        "data/unimib2016",
        "data/unimib2016_kaggle",
        "data/github_samples",
    ]
    
    # 引数でディレクトリ指定も可能
    if len(sys.argv) > 1:
        check_dirs = [sys.argv[1]]
    
    found = False
    for dir_path in check_dirs:
        if Path(dir_path).exists():
            check_dataset_exif(dir_path)
            found = True
            break
    
    if not found:
        print("❌ データセットが見つかりません")
        print("\n使用方法:")
        print("1. まずデータセットをダウンロード")
        print("2. python check_exif.py [データディレクトリ]")
        print("\nデフォルトチェック対象:")
        for d in check_dirs:
            print(f"  - {d}")

if __name__ == "__main__":
    main()