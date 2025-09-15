#!/usr/bin/env python3
"""
EXIF.txtファイルをパースして体積推定用のJSON形式に変換
"""
import json
import re
from pathlib import Path

def parse_gps_coordinate(coord_str):
    """GPS座標をDMS形式から10進数に変換
    例: "44 deg 38' 28.58\"" -> 44.641272
    """
    match = re.match(r"(\d+)\s*deg\s*(\d+)'\s*([\d.]+)\"?", coord_str)
    if match:
        degrees = float(match.group(1))
        minutes = float(match.group(2))
        seconds = float(match.group(3))
        return degrees + minutes/60 + seconds/3600
    return None

def parse_exif_txt(file_path):
    """EXIF.txtファイルをパースして構造化データに変換"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # パース結果を格納
    exif_data = {
        "file_info": {
            "source": "Flickr",
            "file_id": "49632490042_8393211035_o.jpg"
        },
        "camera_info": {},
        "lens_info": {},
        "focal_info": {},
        "exposure_info": {},
        "gps_info": {},
        "image_info": {},
        "datetime_info": {},
        "other_info": {}
    }
    
    # 最初の行はカメラ情報
    if lines[0].strip():
        make_model = lines[0].strip().split()
        if len(make_model) >= 3:
            exif_data["camera_info"]["Make"] = make_model[0]
            exif_data["camera_info"]["Model"] = " ".join(make_model[1:])
    
    # 2行目はレンズ情報
    if len(lines) > 1 and lines[1].strip():
        exif_data["lens_info"]["LensModel"] = lines[1].strip()
    
    # 3行目は撮影パラメータ（ƒ/1.8 4.0 mm 1/12 100）
    if len(lines) > 2 and lines[2].strip():
        params = lines[2].strip()
        # f値
        f_match = re.search(r'ƒ?/?([0-9.]+)', params)
        if f_match:
            exif_data["exposure_info"]["f_number"] = float(f_match.group(1))
        # 焦点距離mm
        mm_match = re.search(r'([0-9.]+)\s*mm', params)
        if mm_match:
            exif_data["focal_info"]["focal_length_mm"] = float(mm_match.group(1))
        # シャッタースピード
        shutter_match = re.search(r'1/(\d+)', params)
        if shutter_match:
            exif_data["exposure_info"]["exposure_time"] = 1.0 / float(shutter_match.group(1))
        # ISO
        iso_match = re.search(r'\b(\d+)\b(?!.*mm)(?!.*/)(?!.*ƒ)', params)
        if iso_match:
            exif_data["exposure_info"]["iso"] = int(iso_match.group(1))
    
    # 残りの行をパース
    for line in lines[3:]:
        if ' - ' in line:
            key, value = line.split(' - ', 1)
            key = key.strip()
            value = value.strip()
            
            # カメラ関連
            if key == "Make":
                exif_data["camera_info"]["Make"] = value
            elif key == "Model":
                exif_data["camera_info"]["Model"] = value
            elif key == "Software":
                exif_data["camera_info"]["Software"] = value
            
            # レンズ関連
            elif key == "Lens Make":
                exif_data["lens_info"]["LensMake"] = value
            elif key == "Lens Model":
                exif_data["lens_info"]["LensModel"] = value
            elif key == "Lens Info":
                exif_data["lens_info"]["LensInfo"] = value
            
            # 焦点距離関連（最重要）
            elif key == "Focal Length (35mm format)":
                focal_match = re.search(r'(\d+)', value)
                if focal_match:
                    exif_data["focal_info"]["focal_length_35mm"] = int(focal_match.group(1))
            
            # 露出関連
            elif key == "ISO Speed":
                exif_data["exposure_info"]["iso"] = int(value)
            elif key == "Brightness Value":
                exif_data["exposure_info"]["brightness_value"] = float(value)
            elif key == "Exposure Bias":
                exif_data["exposure_info"]["exposure_bias"] = value
            elif key == "Metering Mode":
                exif_data["exposure_info"]["metering_mode"] = value
            elif key == "Exposure Mode":
                exif_data["exposure_info"]["exposure_mode"] = value
            elif key == "White Balance":
                exif_data["exposure_info"]["white_balance"] = value
            elif key == "Flash":
                exif_data["exposure_info"]["flash"] = value
            
            # GPS関連
            elif key == "GPS Latitude":
                lat = parse_gps_coordinate(value)
                if lat:
                    exif_data["gps_info"]["latitude"] = lat
            elif key == "GPS Latitude Ref":
                if value == "South" and "latitude" in exif_data["gps_info"]:
                    exif_data["gps_info"]["latitude"] = -exif_data["gps_info"]["latitude"]
            elif key == "GPS Longitude":
                lon = parse_gps_coordinate(value)
                if lon:
                    exif_data["gps_info"]["longitude"] = lon
            elif key == "GPS Longitude Ref":
                if value == "West" and "longitude" in exif_data["gps_info"]:
                    exif_data["gps_info"]["longitude"] = -exif_data["gps_info"]["longitude"]
            elif key == "GPS Altitude":
                alt_match = re.search(r'([\d.]+)', value)
                if alt_match:
                    exif_data["gps_info"]["altitude_m"] = float(alt_match.group(1))
            elif key == "GPS Img Direction":
                exif_data["gps_info"]["img_direction"] = float(value)
            elif key == "GPS HPositioning Error":
                err_match = re.search(r'([\d.]+)', value)
                if err_match:
                    exif_data["gps_info"]["h_positioning_error_m"] = float(err_match.group(1))
            
            # 画像情報
            elif key in ["X-Resolution", "Y-Resolution"]:
                exif_data["image_info"][key.replace("-", "_").lower()] = value
            elif key == "Color Space":
                exif_data["image_info"]["color_space"] = value
            elif key == "Scene Type":
                exif_data["image_info"]["scene_type"] = value
            elif key == "Scene Capture Type":
                exif_data["image_info"]["scene_capture_type"] = value
            
            # 日時情報
            elif "Date and Time" in key:
                datetime_key = key.replace("Date and Time", "DateTime").replace(" ", "_").replace("(", "").replace(")", "")
                exif_data["datetime_info"][datetime_key] = value
            elif "Offset Time" in key:
                exif_data["datetime_info"][key.replace(" ", "_")] = value
            
            # その他
            else:
                exif_data["other_info"][key] = value
    
    return exif_data

def calculate_intrinsics(exif_data, image_width=None, image_height=None):
    """EXIF情報から内部パラメータKを計算"""
    
    # iPhone 8のデフォルト解像度（実際の画像サイズが不明な場合）
    if image_width is None:
        image_width = 4032  # iPhone 8標準
    if image_height is None:
        image_height = 3024  # iPhone 8標準
    
    # 35mm換算焦点距離を取得
    focal_35mm = exif_data["focal_info"].get("focal_length_35mm")
    
    if not focal_35mm:
        return None
    
    # 35mmフルフレームのサイズ
    sensor_width_35mm = 36.0
    sensor_height_35mm = 24.0
    
    # 内部パラメータ計算
    fx = image_width * (focal_35mm / sensor_width_35mm)
    fy = image_height * (focal_35mm / sensor_height_35mm)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "image_width": image_width,
        "image_height": image_height,
        "focal_length_35mm": focal_35mm,
        "matrix": [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
    }

def main():
    """メイン処理"""
    
    # 入力ファイル
    input_file = Path("data/flickr/EXIF.txt")
    
    # 出力ディレクトリ
    output_dir = Path("data/flickr")
    
    print("=" * 60)
    print("EXIF情報をJSONに変換")
    print("=" * 60)
    
    # EXIF.txtをパース
    exif_data = parse_exif_txt(input_file)
    
    # 内部パラメータKを計算
    intrinsics = calculate_intrinsics(exif_data)
    if intrinsics:
        exif_data["intrinsics"] = intrinsics
    
    # 体積推定用の要約情報を作成
    volume_estimation_data = {
        "file_id": "49632490042_8393211035_o.jpg",
        "camera": f"{exif_data['camera_info'].get('Make', '')} {exif_data['camera_info'].get('Model', '')}".strip(),
        "focal_length_mm": exif_data["focal_info"].get("focal_length_mm"),
        "focal_length_35mm": exif_data["focal_info"].get("focal_length_35mm"),
        "f_number": exif_data["exposure_info"].get("f_number"),
        "iso": exif_data["exposure_info"].get("iso"),
        "exposure_time": exif_data["exposure_info"].get("exposure_time"),
        "gps": {
            "latitude": exif_data["gps_info"].get("latitude"),
            "longitude": exif_data["gps_info"].get("longitude"),
            "altitude_m": exif_data["gps_info"].get("altitude_m")
        },
        "intrinsics": intrinsics,
        "timestamp": exif_data["datetime_info"].get("DateTime_Original")
    }
    
    # 完全なEXIFデータを保存
    full_output_path = output_dir / "exif_full.json"
    with open(full_output_path, 'w', encoding='utf-8') as f:
        json.dump(exif_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 完全なEXIFデータ: {full_output_path}")
    
    # 体積推定用データを保存
    volume_output_path = output_dir / "exif_for_volume.json"
    with open(volume_output_path, 'w', encoding='utf-8') as f:
        json.dump(volume_estimation_data, f, indent=2, ensure_ascii=False)
    print(f"✅ 体積推定用データ: {volume_output_path}")
    
    # 重要情報の表示
    print("\n📷 カメラ情報:")
    print(f"  機種: {volume_estimation_data['camera']}")
    print(f"  焦点距離: {volume_estimation_data['focal_length_mm']}mm (35mm換算: {volume_estimation_data['focal_length_35mm']}mm)")
    print(f"  F値: f/{volume_estimation_data['f_number']}")
    print(f"  ISO: {volume_estimation_data['iso']}")
    if volume_estimation_data['exposure_time']:
        print(f"  シャッター: 1/{int(1/volume_estimation_data['exposure_time'])}秒")
    else:
        print(f"  シャッター: N/A")
    
    if intrinsics:
        print("\n📊 内部パラメータ K:")
        print(f"  画像サイズ: {intrinsics['image_width']}x{intrinsics['image_height']}")
        print(f"  fx = {intrinsics['fx']:.2f}")
        print(f"  fy = {intrinsics['fy']:.2f}")
        print(f"  cx = {intrinsics['cx']:.2f}")
        print(f"  cy = {intrinsics['cy']:.2f}")
    
    print("\n📍 GPS情報:")
    print(f"  緯度: {volume_estimation_data['gps']['latitude']:.6f}°")
    print(f"  経度: {volume_estimation_data['gps']['longitude']:.6f}°")
    print(f"  標高: {volume_estimation_data['gps']['altitude_m']}m")

if __name__ == "__main__":
    main()