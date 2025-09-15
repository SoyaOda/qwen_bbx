#!/usr/bin/env python3
"""
体積推定に必要なEXIF情報を抽出して保存するスクリプト
iPhone 8などの実際のカメラで撮影された画像から必要な情報を取得
"""
import os
import json
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import numpy as np
from datetime import datetime

def get_all_exif(image_path):
    """画像から全EXIF情報を取得"""
    try:
        image = Image.open(image_path)
        exifdata = image.getexif()
        
        if not exifdata:
            return None
        
        # EXIF情報を読みやすい形式に変換
        exif_dict = {}
        for tag_id, value in exifdata.items():
            tag = TAGS.get(tag_id, tag_id)
            
            # GPS情報の特別処理
            if tag == 'GPSInfo':
                gps_data = {}
                for gps_tag_id in value:
                    gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                    gps_data[gps_tag] = value[gps_tag_id]
                exif_dict[tag] = gps_data
            else:
                # バイナリデータは文字列化
                if isinstance(value, bytes):
                    try:
                        value = value.decode('utf-8', errors='ignore')
                    except:
                        value = f"<binary data {len(value)} bytes>"
                exif_dict[tag] = value
                
        return exif_dict
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None

def extract_volume_estimation_data(image_path):
    """体積推定に必要なEXIF情報を抽出"""
    
    # 画像の実際のサイズを取得
    image = Image.open(image_path)
    image_width, image_height = image.size
    
    # 全EXIF情報を取得
    all_exif = get_all_exif(image_path)
    
    if not all_exif:
        return None
    
    # 体積推定に重要な情報を抽出
    volume_data = {
        'file_path': str(image_path),
        'image_dimensions': {
            'width': image_width,
            'height': image_height
        },
        'camera_info': {},
        'lens_info': {},
        'focal_info': {},
        'exposure_info': {},
        'gps_info': {},
        'intrinsics': {},
        'timestamp': None
    }
    
    # カメラ情報
    camera_fields = ['Make', 'Model', 'Software']
    for field in camera_fields:
        if field in all_exif:
            volume_data['camera_info'][field] = str(all_exif[field])
    
    # レンズ情報
    lens_fields = ['LensMake', 'LensModel', 'LensInfo', 'LensSpecification']
    for field in lens_fields:
        if field in all_exif:
            volume_data['lens_info'][field] = str(all_exif[field])
    
    # 焦点距離情報（最重要）
    focal_fields = {
        'FocalLength': 'focal_length_mm',
        'FocalLengthIn35mmFilm': 'focal_length_35mm',
        'DigitalZoomRatio': 'digital_zoom',
        'FocalPlaneXResolution': 'focal_plane_x_resolution',
        'FocalPlaneYResolution': 'focal_plane_y_resolution',
        'FocalPlaneResolutionUnit': 'focal_plane_resolution_unit'
    }
    
    for exif_field, json_field in focal_fields.items():
        if exif_field in all_exif:
            value = all_exif[exif_field]
            # Rational型の処理
            if hasattr(value, 'real'):
                value = float(value.real)
            volume_data['focal_info'][json_field] = value
    
    # 露出情報
    exposure_fields = {
        'FNumber': 'f_number',
        'ExposureTime': 'exposure_time',
        'ISOSpeedRatings': 'iso',
        'ExposureBiasValue': 'exposure_bias',
        'ApertureValue': 'aperture_value',
        'BrightnessValue': 'brightness_value',
        'Flash': 'flash',
        'WhiteBalance': 'white_balance'
    }
    
    for exif_field, json_field in exposure_fields.items():
        if exif_field in all_exif:
            value = all_exif[exif_field]
            if hasattr(value, 'real'):
                value = float(value.real)
            volume_data['exposure_info'][json_field] = value
    
    # GPS情報（オプション）
    if 'GPSInfo' in all_exif:
        gps = all_exif['GPSInfo']
        gps_extracted = {}
        
        # 緯度経度の処理
        if 'GPSLatitude' in gps and 'GPSLatitudeRef' in gps:
            lat = convert_gps_to_decimal(gps['GPSLatitude'])
            if gps['GPSLatitudeRef'] == 'S':
                lat = -lat
            gps_extracted['latitude'] = lat
            
        if 'GPSLongitude' in gps and 'GPSLongitudeRef' in gps:
            lon = convert_gps_to_decimal(gps['GPSLongitude'])
            if gps['GPSLongitudeRef'] == 'W':
                lon = -lon
            gps_extracted['longitude'] = lon
            
        if 'GPSAltitude' in gps:
            alt = float(gps['GPSAltitude'])
            if 'GPSAltitudeRef' in gps and gps['GPSAltitudeRef'] == 1:
                alt = -alt
            gps_extracted['altitude_m'] = alt
            
        volume_data['gps_info'] = gps_extracted
    
    # 画像方向
    if 'Orientation' in all_exif:
        volume_data['orientation'] = all_exif['Orientation']
    
    # 撮影日時
    date_fields = ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized']
    for field in date_fields:
        if field in all_exif:
            volume_data['timestamp'] = str(all_exif[field])
            break
    
    # 内部パラメータKの計算
    if 'focal_length_35mm' in volume_data['focal_info']:
        K = compute_intrinsics_from_35mm(
            volume_data['focal_info']['focal_length_35mm'],
            image_width,
            image_height
        )
        volume_data['intrinsics'] = K
    
    # 生のEXIF情報も保存（デバッグ用）
    volume_data['raw_exif'] = {k: str(v)[:200] for k, v in all_exif.items()}
    
    return volume_data

def convert_gps_to_decimal(gps_coord):
    """GPS座標をDMS形式から10進数に変換"""
    if len(gps_coord) == 3:
        degrees = float(gps_coord[0])
        minutes = float(gps_coord[1])
        seconds = float(gps_coord[2])
        return degrees + minutes/60 + seconds/3600
    return None

def compute_intrinsics_from_35mm(focal_35mm, image_width, image_height):
    """35mm換算焦点距離から内部パラメータKを計算"""
    
    # 35mmフルフレームのサイズ（mm）
    sensor_width_35mm = 36.0
    sensor_height_35mm = 24.0
    
    # 焦点距離をピクセル単位に変換
    fx = image_width * (focal_35mm / sensor_width_35mm)
    fy = image_height * (focal_35mm / sensor_height_35mm)
    
    # 主点（画像中心）
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # 内部パラメータ行列K
    K = {
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy,
        'matrix': [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
    }
    
    return K

def process_image(image_path, output_dir):
    """画像を処理してEXIF情報を保存"""
    
    print(f"\n処理中: {image_path}")
    
    # EXIF情報を抽出
    volume_data = extract_volume_estimation_data(image_path)
    
    if not volume_data:
        print("  ❌ EXIF情報なし")
        return None
    
    # ファイル名を生成
    image_name = Path(image_path).stem
    json_path = Path(output_dir) / f"{image_name}_exif.json"
    
    # JSON形式で保存
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(volume_data, f, indent=2, ensure_ascii=False)
    
    print(f"  ✅ 保存: {json_path}")
    
    # 重要情報のサマリー表示
    if volume_data['intrinsics']:
        K = volume_data['intrinsics']
        print(f"  📷 カメラ: {volume_data['camera_info'].get('Make', 'Unknown')} {volume_data['camera_info'].get('Model', 'Unknown')}")
        print(f"  📐 画像サイズ: {volume_data['image_dimensions']['width']}x{volume_data['image_dimensions']['height']}")
        print(f"  🎯 焦点距離(35mm): {volume_data['focal_info'].get('focal_length_35mm', 'N/A')}mm")
        print(f"  📊 内部パラメータ K:")
        print(f"     fx={K['fx']:.2f}, fy={K['fy']:.2f}")
        print(f"     cx={K['cx']:.2f}, cy={K['cy']:.2f}")
    
    return volume_data

def main():
    """メイン処理"""
    
    # 出力ディレクトリ
    output_dir = Path("exif_food_dataset/data/volume_exif")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("体積推定用EXIF情報抽出ツール")
    print("=" * 60)
    
    # 処理対象の画像パス（例）
    test_images = [
        "exif_food_dataset/data/flickr/49632490042_8393211035_o.jpg",  # iPhone 8画像
        # 他の画像パスを追加
    ]
    
    # ディレクトリ内の画像を処理する場合
    image_dirs = [
        "exif_food_dataset/data/flickr",
        "exif_food_dataset/data/test_images",
    ]
    
    all_images = []
    
    # 個別指定の画像
    for path in test_images:
        if Path(path).exists():
            all_images.append(path)
    
    # ディレクトリ内の画像
    for dir_path in image_dirs:
        if Path(dir_path).exists():
            for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
                all_images.extend(Path(dir_path).glob(ext))
    
    if not all_images:
        print("❌ 処理する画像が見つかりません")
        print("\n画像を以下の場所に配置してください:")
        print("  - exif_food_dataset/data/flickr/")
        print("  - exif_food_dataset/data/test_images/")
        return
    
    # 各画像を処理
    results = []
    for image_path in all_images:
        result = process_image(image_path, output_dir)
        if result:
            results.append(result)
    
    # サマリーレポート作成
    summary_path = output_dir / "summary.json"
    summary = {
        'processed_count': len(all_images),
        'success_count': len(results),
        'images': []
    }
    
    for result in results:
        summary['images'].append({
            'file': result['file_path'],
            'has_35mm_focal': 'focal_length_35mm' in result['focal_info'],
            'has_intrinsics': bool(result['intrinsics']),
            'camera': f"{result['camera_info'].get('Make', 'Unknown')} {result['camera_info'].get('Model', 'Unknown')}"
        })
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("処理完了")
    print(f"  処理画像数: {len(all_images)}")
    print(f"  成功: {len(results)}")
    print(f"  出力先: {output_dir}")
    print(f"  サマリー: {summary_path}")

if __name__ == "__main__":
    main()