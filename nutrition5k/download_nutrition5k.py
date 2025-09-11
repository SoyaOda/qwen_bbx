#!/usr/bin/env python
"""
Nutrition5k データセットダウンロードスクリプト
Google Cloud Storage (GCS) から Nutrition5k データセットをダウンロードします。
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path
import time
from typing import Optional, List


class Nutrition5kDownloader:
    """Nutrition5k データセットダウンロードクラス"""
    
    def __init__(self, output_dir: str = "nutrition5k_dataset", use_tar: bool = False):
        """
        Args:
            output_dir: データセットの保存先ディレクトリ
            use_tar: True の場合、tar.gz ファイルをダウンロード（181.4GB）
                     False の場合、rsync で同期（推奨）
        """
        self.output_dir = Path(output_dir)
        self.use_tar = use_tar
        self.gcs_bucket = "gs://nutrition5k_dataset/nutrition5k_dataset"
        self.gcs_tar = "gs://nutrition5k_dataset/nutrition5k_dataset.tar.gz"
        
    def check_gcloud_installed(self) -> bool:
        """gcloud CLI がインストールされているか確認"""
        try:
            result = subprocess.run(["gcloud", "--version"], 
                                  capture_output=True, text=True)
            print("✓ gcloud CLI が検出されました")
            print(f"  バージョン: {result.stdout.split()[0]}")
            return True
        except FileNotFoundError:
            print("✗ gcloud CLI が見つかりません")
            return False
            
    def check_gsutil_installed(self) -> bool:
        """gsutil がインストールされているか確認"""
        try:
            result = subprocess.run(["gsutil", "version"], 
                                  capture_output=True, text=True)
            print("✓ gsutil が検出されました")
            return True
        except FileNotFoundError:
            print("✗ gsutil が見つかりません")
            return False
            
    def setup_output_directory(self):
        """出力ディレクトリを作成"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n✓ 出力ディレクトリを準備しました: {self.output_dir.absolute()}")
        
    def check_disk_space(self) -> bool:
        """十分なディスク容量があるか確認（最低200GB）"""
        import shutil
        stat = shutil.disk_usage(self.output_dir)
        free_gb = stat.free / (1024 ** 3)
        
        print(f"\n📊 ディスク容量チェック:")
        print(f"  空き容量: {free_gb:.1f} GB")
        
        required_gb = 200 if not self.use_tar else 400  # tar.gz の場合は展開分も必要
        if free_gb < required_gb:
            print(f"  ⚠️  警告: {required_gb}GB 以上の空き容量が推奨されます")
            return False
        else:
            print(f"  ✓ 十分な空き容量があります")
            return True
            
    def download_with_rsync(self) -> bool:
        """rsync を使用してデータセットを同期（推奨方法）"""
        print("\n📥 rsync でデータセットをダウンロード中...")
        print(f"  ソース: {self.gcs_bucket}")
        print(f"  保存先: {self.output_dir}")
        print("  ※ 初回は時間がかかります（約180GB）\n")
        
        # gcloud storage rsync を使用（新推奨）
        cmd = [
            "gcloud", "storage", "rsync", "-r",
            self.gcs_bucket,
            str(self.output_dir)
        ]
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT, 
                                     text=True, bufsize=1)
            
            # リアルタイムで出力を表示
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"  {line.strip()}")
                    
            process.wait()
            
            if process.returncode == 0:
                print("\n✓ ダウンロード完了！")
                return True
            else:
                print(f"\n✗ ダウンロードエラー (code: {process.returncode})")
                return False
                
        except Exception as e:
            print(f"\n✗ エラー: {e}")
            return False
            
    def download_tar_gz(self) -> bool:
        """tar.gz ファイルをダウンロード（代替方法）"""
        print("\n📥 tar.gz ファイルをダウンロード中...")
        print(f"  ソース: {self.gcs_tar}")
        print(f"  保存先: {self.output_dir}")
        print("  ※ 181.4GB のファイルです\n")
        
        tar_path = self.output_dir / "nutrition5k_dataset.tar.gz"
        
        cmd = [
            "gcloud", "storage", "cp",
            self.gcs_tar,
            str(tar_path)
        ]
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT, 
                                     text=True, bufsize=1)
            
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(f"  {line.strip()}")
                    
            process.wait()
            
            if process.returncode == 0:
                print("\n✓ ダウンロード完了！")
                print("\n📦 tar.gz を展開中...")
                
                # 展開
                extract_cmd = ["tar", "-xzf", str(tar_path), "-C", str(self.output_dir)]
                subprocess.run(extract_cmd, check=True)
                
                print("✓ 展開完了！")
                
                # tar.gz を削除するか確認
                response = input("\ntar.gz ファイルを削除しますか？ (y/n): ")
                if response.lower() == 'y':
                    tar_path.unlink()
                    print("✓ tar.gz ファイルを削除しました")
                    
                return True
            else:
                print(f"\n✗ ダウンロードエラー (code: {process.returncode})")
                return False
                
        except Exception as e:
            print(f"\n✗ エラー: {e}")
            return False
            
    def verify_dataset(self):
        """ダウンロードしたデータセットの構成を確認"""
        print("\n🔍 データセット構成を確認中...")
        
        expected_dirs = [
            "imagery/side_angles",
            "imagery/realsense_overhead", 
            "metadata",
            "dish_ids/splits",
            "scripts"
        ]
        
        for dir_path in expected_dirs:
            full_path = self.output_dir / dir_path
            if full_path.exists():
                print(f"  ✓ {dir_path}")
            else:
                print(f"  ✗ {dir_path} (見つかりません)")
                
        # メタデータファイルのサンプル表示
        metadata_dir = self.output_dir / "metadata"
        if metadata_dir.exists():
            csv_files = list(metadata_dir.glob("*.csv"))[:3]
            if csv_files:
                print(f"\n📋 メタデータファイル例:")
                for csv_file in csv_files:
                    print(f"  - {csv_file.name}")
                    
    def download_subset(self, components: List[str]):
        """特定のコンポーネントのみダウンロード"""
        print(f"\n📥 選択されたコンポーネントをダウンロード中: {', '.join(components)}")
        
        for component in components:
            if component == "metadata":
                src = f"{self.gcs_bucket}/metadata"
                dst = self.output_dir / "metadata"
            elif component == "overhead":
                src = f"{self.gcs_bucket}/imagery/realsense_overhead"  
                dst = self.output_dir / "imagery" / "realsense_overhead"
            elif component == "side_angles":
                src = f"{self.gcs_bucket}/imagery/side_angles"
                dst = self.output_dir / "imagery" / "side_angles"
            elif component == "splits":
                src = f"{self.gcs_bucket}/dish_ids/splits"
                dst = self.output_dir / "dish_ids" / "splits"
            else:
                print(f"  ⚠️  不明なコンポーネント: {component}")
                continue
                
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            cmd = ["gcloud", "storage", "rsync", "-r", src, str(dst)]
            
            print(f"\n  📂 {component} をダウンロード中...")
            subprocess.run(cmd, check=True)
            print(f"  ✓ {component} 完了")
            
    def run(self, subset: Optional[List[str]] = None):
        """ダウンロード処理を実行"""
        print("=" * 60)
        print("Nutrition5k データセットダウンローダー")
        print("=" * 60)
        
        # gcloud CLI チェック
        if not self.check_gcloud_installed():
            print("\n⚠️  gcloud CLI をインストールしてください:")
            print("  brew install --cask gcloud-cli")
            print("  または https://cloud.google.com/sdk/docs/install")
            return False
            
        # 出力ディレクトリ準備
        self.setup_output_directory()
        
        # ディスク容量チェック
        self.check_disk_space()
        
        # ダウンロード実行
        start_time = time.time()
        
        if subset:
            # 部分ダウンロード
            self.download_subset(subset)
            success = True
        elif self.use_tar:
            # tar.gz ダウンロード
            success = self.download_tar_gz()
        else:
            # rsync ダウンロード（推奨）
            success = self.download_with_rsync()
            
        elapsed_time = time.time() - start_time
        
        if success:
            print(f"\n✅ 完了！（所要時間: {elapsed_time/60:.1f} 分）")
            self.verify_dataset()
        else:
            print(f"\n❌ ダウンロードに失敗しました")
            
        return success


def main():
    parser = argparse.ArgumentParser(
        description="Nutrition5k データセットをダウンロード",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # フルデータセットをダウンロード（推奨）
  python download_nutrition5k.py
  
  # tar.gz 版をダウンロード
  python download_nutrition5k.py --use-tar
  
  # メタデータのみダウンロード
  python download_nutrition5k.py --subset metadata
  
  # 複数コンポーネントをダウンロード
  python download_nutrition5k.py --subset metadata overhead splits
        """
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="nutrition5k_dataset",
        help="出力ディレクトリ（デフォルト: nutrition5k_dataset）"
    )
    
    parser.add_argument(
        "--use-tar",
        action="store_true",
        help="tar.gz ファイルをダウンロード（181.4GB）"
    )
    
    parser.add_argument(
        "--subset",
        nargs="+",
        choices=["metadata", "overhead", "side_angles", "splits"],
        help="特定のコンポーネントのみダウンロード"
    )
    
    args = parser.parse_args()
    
    downloader = Nutrition5kDownloader(
        output_dir=args.output_dir,
        use_tar=args.use_tar
    )
    
    success = downloader.run(subset=args.subset)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()