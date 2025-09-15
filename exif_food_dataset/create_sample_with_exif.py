#!/usr/bin/env python3
"""
iPhone 8のEXIF情報を模したサンプルデータを作成
実際のEXIF情報に基づいて体積推定用のテストデータを生成
"""
import json
from pathlib import Path

def create_iphone8_sample():
    """iPhone 8の実際のEXIF情報を基にしたサンプルデータ"""
    
    # iPhone 8の典型的なEXIF情報
    sample_exif = {
        "file_path": "exif_food_dataset/data/sample/iphone8_food_sample.jpg",
        "image_dimensions": {
            "width": 4032,  # iPhone 8の標準解像度
            "height": 3024
        },
        "camera_info": {
            "Make": "Apple",
            "Model": "iPhone 8",
            "Software": "13.1.3"
        },
        "lens_info": {
            "LensModel": "iPhone 8 back camera 3.99mm f/1.8",
            "LensInfo": "3.99mm f/1.8"
        },
        "focal_info": {
            "focal_length_mm": 3.99,  # 実焦点距離
            "focal_length_35mm": 28,  # 35mm換算
            "focal_plane_x_resolution": 6.0,
            "focal_plane_y_resolution": 6.0,
            "focal_plane_resolution_unit": 3
        },
        "exposure_info": {
            "f_number": 1.8,
            "exposure_time": 0.0833,  # 1/12秒
            "iso": 100,
            "brightness_value": 0.6775,
            "flash": "Flash (auto, did not fire)",
            "white_balance": "Auto"
        },
        "gps_info": {
            "latitude": 44.641272,  # 44°38'28.58"N
            "longitude": -93.145600,  # 93°8'44.16"W
            "altitude_m": 279
        },
        "orientation": 1,  # 通常の向き
        "timestamp": "2020:03:06 17:12:19",
        "intrinsics": {}  # 後で計算
    }
    
    # 内部パラメータKの計算
    width = sample_exif["image_dimensions"]["width"]
    height = sample_exif["image_dimensions"]["height"]
    focal_35mm = sample_exif["focal_info"]["focal_length_35mm"]
    
    # 35mmフルフレームのサイズ
    sensor_width_35mm = 36.0
    sensor_height_35mm = 24.0
    
    # 内部パラメータ計算
    fx = width * (focal_35mm / sensor_width_35mm)
    fy = height * (focal_35mm / sensor_height_35mm)
    cx = width / 2.0
    cy = height / 2.0
    
    sample_exif["intrinsics"] = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "matrix": [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
    }
    
    return sample_exif

def create_various_samples():
    """異なるカメラ/条件のサンプルデータを作成"""
    
    samples = []
    
    # 1. iPhone 8 (標準)
    iphone8 = create_iphone8_sample()
    samples.append(iphone8)
    
    # 2. iPhone 12 Pro (広角)
    iphone12_wide = {
        "file_path": "exif_food_dataset/data/sample/iphone12_wide_food.jpg",
        "image_dimensions": {"width": 4032, "height": 3024},
        "camera_info": {
            "Make": "Apple",
            "Model": "iPhone 12 Pro",
            "Software": "14.4"
        },
        "focal_info": {
            "focal_length_mm": 1.54,
            "focal_length_35mm": 13,  # 超広角
        },
        "exposure_info": {
            "f_number": 2.4,
            "exposure_time": 0.025,
            "iso": 200
        },
        "intrinsics": {}
    }
    
    # 内部パラメータ計算
    fx = 4032 * (13 / 36.0)
    fy = 3024 * (13 / 24.0)
    iphone12_wide["intrinsics"] = {
        "fx": fx, "fy": fy,
        "cx": 2016, "cy": 1512,
        "matrix": [[fx, 0, 2016], [0, fy, 1512], [0, 0, 1]]
    }
    samples.append(iphone12_wide)
    
    # 3. DSLR (Canon)
    dslr = {
        "file_path": "exif_food_dataset/data/sample/canon_dslr_food.jpg",
        "image_dimensions": {"width": 6000, "height": 4000},
        "camera_info": {
            "Make": "Canon",
            "Model": "Canon EOS 5D Mark IV",
            "Software": "Firmware 1.2.1"
        },
        "focal_info": {
            "focal_length_mm": 50,
            "focal_length_35mm": 50,  # フルフレーム
        },
        "exposure_info": {
            "f_number": 2.8,
            "exposure_time": 0.01,
            "iso": 400
        },
        "intrinsics": {}
    }
    
    # 内部パラメータ計算
    fx = 6000 * (50 / 36.0)
    fy = 4000 * (50 / 24.0)
    dslr["intrinsics"] = {
        "fx": fx, "fy": fy,
        "cx": 3000, "cy": 2000,
        "matrix": [[fx, 0, 3000], [0, fy, 2000], [0, 0, 1]]
    }
    samples.append(dslr)
    
    return samples

def main():
    """メイン処理"""
    
    # 出力ディレクトリ
    output_dir = Path("data/volume_exif")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("体積推定用EXIFサンプルデータ作成")
    print("=" * 60)
    
    # サンプルデータ生成
    samples = create_various_samples()
    
    # 各サンプルを保存
    for i, sample in enumerate(samples, 1):
        # ファイル名を生成
        camera = sample['camera_info']['Model'].replace(' ', '_')
        json_path = output_dir / f"sample_{i}_{camera}_exif.json"
        
        # JSON形式で保存
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        
        print(f"\n📷 サンプル {i}: {sample['camera_info']['Model']}")
        print(f"   画像サイズ: {sample['image_dimensions']['width']}x{sample['image_dimensions']['height']}")
        print(f"   焦点距離(35mm): {sample['focal_info'].get('focal_length_35mm', 'N/A')}mm")
        K = sample['intrinsics']
        print(f"   内部パラメータ K:")
        print(f"     fx={K['fx']:.2f}, fy={K['fy']:.2f}")
        print(f"     cx={K['cx']:.2f}, cy={K['cy']:.2f}")
        print(f"   保存先: {json_path}")
    
    # バッチ処理用の統合ファイル作成
    batch_path = output_dir / "batch_samples.json"
    batch_data = {
        "description": "体積推定用EXIFサンプルデータセット",
        "created": "2024-01-14",
        "samples": samples
    }
    
    with open(batch_path, 'w', encoding='utf-8') as f:
        json.dump(batch_data, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("✅ サンプルデータ作成完了")
    print(f"   個別ファイル: {len(samples)}個")
    print(f"   統合ファイル: {batch_path}")
    print("\n使用方法:")
    print("   これらのJSON内の内部パラメータKを体積推定に使用できます")

if __name__ == "__main__":
    main()