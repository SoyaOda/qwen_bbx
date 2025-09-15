#!/usr/bin/env python3
"""
UNIMIB2016データセットの全画像のEXIF情報を詳細分析
"""
import os
import json
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
from collections import defaultdict
import csv

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
        return None

def analyze_all_images(data_dir):
    """全画像のEXIF情報を分析"""
    data_path = Path(data_dir)
    
    # 画像ファイルを検索
    image_files = list(data_path.rglob('*.jpg')) + list(data_path.rglob('*.JPG'))
    
    print(f"📁 分析対象: {data_dir}")
    print(f"🖼️ 総画像数: {len(image_files)}")
    print("=" * 60)
    
    # 統計情報
    stats = {
        'total_images': len(image_files),
        'images_with_exif': 0,
        'exif_tags_count': defaultdict(int),
        'orientation_distribution': defaultdict(int),
        'sample_images': []
    }
    
    # 全画像を分析
    for img_path in image_files:
        exif_data = get_exif_data(img_path)
        
        if exif_data:
            stats['images_with_exif'] += 1
            
            # 各タグの出現回数をカウント
            for tag in exif_data:
                stats['exif_tags_count'][tag] += 1
            
            # Orientationの分布
            if 'Orientation' in exif_data:
                stats['orientation_distribution'][exif_data['Orientation']] += 1
            
            # サンプルを保存（最初の5枚）
            if len(stats['sample_images']) < 5:
                stats['sample_images'].append({
                    'path': str(img_path.relative_to(data_path)),
                    'exif': {k: str(v)[:100] for k, v in exif_data.items()}
                })
    
    # 結果の表示
    print("\n📊 分析結果サマリー")
    print("=" * 60)
    print(f"総画像数: {stats['total_images']}")
    print(f"EXIF情報あり: {stats['images_with_exif']} ({stats['images_with_exif']/stats['total_images']*100:.2f}%)")
    
    print("\n📷 EXIFタグの出現頻度（上位10）:")
    sorted_tags = sorted(stats['exif_tags_count'].items(), key=lambda x: x[1], reverse=True)[:10]
    for tag, count in sorted_tags:
        percentage = count / stats['total_images'] * 100
        print(f"  {str(tag):25s}: {count:4d} ({percentage:6.2f}%)")
    
    print("\n🔄 Orientation（画像向き）の分布:")
    orientation_names = {
        1: "通常",
        3: "180度回転",
        6: "90度時計回り",
        8: "90度反時計回り"
    }
    for orientation, count in stats['orientation_distribution'].items():
        name = orientation_names.get(orientation, f"Unknown({orientation})")
        percentage = count / stats['total_images'] * 100
        print(f"  {name:20s}: {count:4d} ({percentage:6.2f}%)")
    
    # サンプル画像の詳細
    if stats['sample_images']:
        print("\n📝 サンプル画像のEXIF詳細（最初の3枚）:")
        for i, sample in enumerate(stats['sample_images'][:3], 1):
            print(f"\n  画像 {i}: {sample['path']}")
            for tag, value in sample['exif'].items():
                print(f"    {str(tag):20s}: {value}")
    
    # JSONレポートを保存
    report_path = Path(data_dir).parent / "exif_analysis_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n💾 詳細レポート保存: {report_path}")
    
    # CSVレポートも作成
    csv_path = Path(data_dir).parent / "exif_summary.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['メトリック', '値', 'パーセンテージ'])
        writer.writerow(['総画像数', stats['total_images'], '100.00%'])
        writer.writerow(['EXIF情報あり', stats['images_with_exif'], 
                        f"{stats['images_with_exif']/stats['total_images']*100:.2f}%"])
        writer.writerow([])
        writer.writerow(['EXIFタグ', '出現回数', 'パーセンテージ'])
        for tag, count in sorted_tags:
            writer.writerow([tag, count, f"{count/stats['total_images']*100:.2f}%"])
    print(f"💾 CSV サマリー保存: {csv_path}")
    
    return stats

def main():
    """メイン処理"""
    print("=" * 60)
    print("UNIMIB2016 EXIF詳細分析")
    print("=" * 60)
    
    # 解凍したデータのパス
    data_dir = "data/unimib2016/pre8/pre8"
    
    if not Path(data_dir).exists():
        print(f"❌ ディレクトリが存在しません: {data_dir}")
        return
    
    stats = analyze_all_images(data_dir)
    
    # 結論
    print("\n" + "=" * 60)
    print("🔍 結論:")
    if stats['images_with_exif'] / stats['total_images'] < 0.1:
        print("❌ このデータセットにはEXIF情報がほとんど含まれていません。")
        print("   Kaggle版はEXIF情報が削除された可能性があります。")
        print("\n💡 推奨事項:")
        print("   1. ResearchGateから直接オリジナル版を入手")
        print("   2. 論文著者に直接連絡してオリジナル画像を要求")
        print("   3. 他のEXIF保持データセット（ECUSTFD等）を検討")
    else:
        print("✅ EXIF情報が部分的に保持されています。")
        if 'Orientation' in stats['exif_tags_count']:
            print("   - Orientation情報は利用可能")
        if any(tag in stats['exif_tags_count'] for tag in ['FocalLength', 'Make', 'Model']):
            print("   - カメラ情報も一部利用可能")

if __name__ == "__main__":
    main()