#!/usr/bin/env python3
"""
EXIF情報を持つ30枚の画像でテスト用データセットを作成
"""
import os
import json
import shutil
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import csv

def has_exif(image_path):
    """画像がEXIF情報を持つか確認"""
    try:
        image = Image.open(image_path)
        exifdata = image.getexif()
        return exifdata is not None and len(exifdata) > 0
    except:
        return False

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
            # バイナリデータは短縮表示
            if isinstance(value, bytes):
                value = f"<binary data {len(value)} bytes>"
            exif_dict[tag] = str(value)
            
        return exif_dict
    except Exception as e:
        return None

def create_exif_test_dataset():
    """EXIF付き画像のテストデータセットを作成"""
    
    # ソースディレクトリ
    source_dir = Path("data/unimib2016/pre8/pre8")
    
    # テスト用データセットディレクトリ
    test_dataset_dir = Path("data/exif_test_dataset")
    test_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EXIF付きテストデータセット作成")
    print("=" * 60)
    
    # EXIF情報を持つ画像を検索
    print("🔍 EXIF情報を持つ画像を検索中...")
    
    exif_images = []
    all_images = list(source_dir.rglob('*.jpg')) + list(source_dir.rglob('*.JPG'))
    
    for img_path in all_images:
        if has_exif(img_path):
            exif_data = get_exif_data(img_path)
            if exif_data:
                exif_images.append({
                    'source_path': img_path,
                    'exif': exif_data
                })
                if len(exif_images) >= 30:
                    break
    
    print(f"✅ {len(exif_images)}枚のEXIF付き画像を発見")
    
    # 画像をコピー
    print("\n📁 テストデータセットを作成中...")
    
    # メタデータ用リスト
    metadata = []
    
    for i, img_info in enumerate(exif_images, 1):
        source_path = img_info['source_path']
        
        # 新しいファイル名（連番）
        new_filename = f"exif_image_{i:03d}.jpg"
        dest_path = test_dataset_dir / new_filename
        
        # 画像をコピー
        shutil.copy2(source_path, dest_path)
        
        # メタデータを保存
        metadata.append({
            'id': i,
            'filename': new_filename,
            'original_path': str(source_path.relative_to(source_dir)),
            'exif': img_info['exif']
        })
        
        print(f"  [{i:2d}/30] {source_path.name} → {new_filename}")
    
    # メタデータをJSON形式で保存
    metadata_path = test_dataset_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\n💾 メタデータ保存: {metadata_path}")
    
    # EXIF情報サマリーをCSVで保存
    csv_path = test_dataset_dir / "exif_summary.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'ファイル名', 'オリジナルパス', 'EXIF タグ数', '主要なEXIFタグ'])
        
        for item in metadata:
            main_tags = []
            for tag in ['Orientation', 'Make', 'Model', 'FocalLength', 'DateTimeOriginal']:
                if tag in item['exif']:
                    main_tags.append(f"{tag}={item['exif'][tag]}")
            
            writer.writerow([
                item['id'],
                item['filename'],
                item['original_path'],
                len(item['exif']),
                '; '.join(main_tags) if main_tags else 'N/A'
            ])
    
    print(f"💾 CSVサマリー保存: {csv_path}")
    
    # データセット情報を表示
    print("\n" + "=" * 60)
    print("📊 作成されたテストデータセット:")
    print(f"  場所: {test_dataset_dir}")
    print(f"  画像数: {len(metadata)}枚")
    print(f"  すべての画像にEXIF情報付き")
    
    # EXIF情報の統計
    all_tags = set()
    for item in metadata:
        all_tags.update(item['exif'].keys())
    
    print(f"\n📷 EXIF情報の概要:")
    print(f"  検出されたEXIFタグ種類: {len(all_tags)}")
    print(f"  主なタグ: {', '.join([str(tag) for tag in list(all_tags)[:10]])}")
    
    return test_dataset_dir

def verify_test_dataset(dataset_dir):
    """作成したテストデータセットを検証"""
    print("\n" + "=" * 60)
    print("🔍 テストデータセットの検証")
    print("=" * 60)
    
    dataset_path = Path(dataset_dir)
    images = list(dataset_path.glob('*.jpg'))
    
    exif_count = 0
    for img_path in images:
        if has_exif(img_path):
            exif_count += 1
    
    print(f"  総画像数: {len(images)}")
    print(f"  EXIF付き: {exif_count}/{len(images)} ({exif_count/len(images)*100:.1f}%)")
    
    if exif_count == len(images):
        print("  ✅ すべての画像にEXIF情報が保持されています")
    else:
        print("  ⚠️ 一部の画像でEXIF情報が失われています")

def main():
    """メイン処理"""
    # テストデータセット作成
    test_dataset_dir = create_exif_test_dataset()
    
    # 検証
    verify_test_dataset(test_dataset_dir)
    
    print("\n✅ テストデータセット作成完了！")
    print(f"   使用方法: python check_exif.py {test_dataset_dir}")

if __name__ == "__main__":
    main()