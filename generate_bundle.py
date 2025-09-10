#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UniDepth v2実装の主要ファイルを1つのMarkdownファイルにまとめるスクリプト
"""
import os
import sys
from datetime import datetime

# 出力ファイル名
output_file = "UNIDEPTH_IMPLEMENTATION_BUNDLE.md"

# 読み込むファイルのリスト（順序付き）
files_to_bundle = [
    ("src/unidepth_runner.py", "UniDepth v2推論エンジン（初期実装、K_scale_factor=6.0）"),
    ("src/unidepth_runner_v2.py", "改良版UniDepth推論エンジン（レビュー推奨実装、K_scale削除）"),
    ("src/unidepth_runner_final.py", "最終版UniDepth推論エンジン（適応的K_scale対応）"),
    ("src/plane_fit.py", "RANSAC平面フィッティング（初期実装）"),
    ("src/plane_fit_v2.py", "改良版RANSAC平面フィッティング（適応的閾値）"),
    ("src/volume_estimator.py", "体積推定モジュール"),
    ("src/vis_depth.py", "深度マップ可視化"),
    ("src/run_unidepth.py", "メインパイプライン（Qwen→SAM2→UniDepth→体積）"),
    ("config.yaml", "全体設定"),
    ("test_all_images.py", "汎用性検証テスト（全画像での最適K_scale探索）"),
    ("test_with_masks_only.py", "SAM2マスク画像での体積推定テスト")
]

# Markdownコンテンツを構築
content = []

# ヘッダー
content.append("# UniDepth v2 実装コード集")
content.append("")
content.append(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
content.append("")
content.append("## 概要")
content.append("")
content.append("このドキュメントは、Qwen BBXプロジェクトにおけるUniDepth v2統合の主要実装ファイルをまとめたものです。")
content.append("")
content.append("### 検証結果と重要な発見")
content.append("")
content.append("#### 1. K_scale_factorの必要性")
content.append("- **UniDepthの根本的な問題**: モデルが推定するカメラ内部パラメータ（fx, fy）が実際の値より約10倍小さい")
content.append("- **原因**: UniDepthは近接撮影（食品撮影）のデータで訓練されていない")
content.append("- **影響**: K_scale_factorなしでは体積が46Lなど非現実的な値になる")
content.append("")
content.append("#### 2. 最適なK_scale_factorの値")
content.append("- **理論的に正しい値**: 約10.5（検証により判明）")
content.append("- **実用的な値**: 画像によって1.0〜12.0まで大きく変動")
content.append("- **深度による適応的調整**:")
content.append("  - 深度 < 0.8m: K_scale = 12.0")
content.append("  - 深度 < 1.2m: K_scale = 10.5")
content.append("  - 深度 < 1.5m: K_scale = 8.0")
content.append("  - それ以上: K_scale = 6.0")
content.append("")
content.append("#### 3. 検証結果（test_all_images.py）")
content.append("- **固定K_scale=6.0**: 成功率11.1%（画像によって最適値が異なるため）")
content.append("- **固定K_scale=10.5**: 理論的に正しいが、汎用性は低い")
content.append("- **適応的K_scale**: 深度に基づく調整でも限界あり（成功率約33%）")
content.append("")
content.append("### 技術的詳細")
content.append("")
content.append("#### ピクセル面積の計算式")
content.append("```")
content.append("a_pix = Z² / (fx * fy)")
content.append("```")
content.append("- Z: 深度値[m]")
content.append("- fx, fy: カメラの焦点距離[pixels]")
content.append("- UniDepthはfx≈686を推定するが、実際にはfx≈10,500が必要")
content.append("")
content.append("#### ImageNet正規化について")
content.append("- **結論**: 正規化は不要（model.infer()が内部で処理）")
content.append("- **影響**: 正規化の有無で体積が0.8倍程度変化するのみ")
content.append("")
content.append("---")
content.append("")

# 目次
content.append("## 目次")
content.append("")
for i, (filepath, description) in enumerate(files_to_bundle, 1):
    filename = os.path.basename(filepath)
    content.append(f"{i}. [{filename}](#{filename.replace('.', '').replace('_', '-').lower()}) - {description}")
content.append("")
content.append("---")
content.append("")

# 各ファイルの内容
for filepath, description in files_to_bundle:
    filename = os.path.basename(filepath)
    
    # セクションヘッダー
    content.append(f"## {filename}")
    content.append("")
    content.append(f"**パス**: `{filepath}`")
    content.append("")
    content.append(f"**説明**: {description}")
    content.append("")
    
    # ファイル内容を読み込み
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # コードブロックとして追加
        if filepath.endswith('.yaml'):
            content.append("```yaml")
        else:
            content.append("```python")
        content.append(file_content)
        content.append("```")
        content.append("")
        
        # ファイルサイズと行数
        lines = file_content.split('\n')
        content.append(f"*ファイルサイズ: {len(file_content):,} bytes, 行数: {len(lines):,}*")
        content.append("")
        
    except FileNotFoundError:
        content.append("**エラー**: ファイルが見つかりません")
        content.append("")
    except Exception as e:
        content.append(f"**エラー**: {str(e)}")
        content.append("")
    
    content.append("---")
    content.append("")

# フッター
content.append("## 使用上の注意")
content.append("")
content.append("### 環境要件")
content.append("- Python 3.11")
content.append("- CUDA対応GPU（推奨）")
content.append("- UniDepthパッケージ（GitHub: lpiccinelli-eth/UniDepth）")
content.append("")
content.append("### 体積調整方法")
content.append("")
content.append("#### 方法1: 固定K_scale_factor（簡単だが精度低い）")
content.append("`config.yaml`で設定:")
content.append("```yaml")
content.append("unidepth:")
content.append("  K_scale_factor: 6.0  # 3.0〜12.0の範囲で調整")
content.append("```")
content.append("")
content.append("#### 方法2: 適応的K_scale（推奨）")
content.append("`unidepth_runner_final.py`の`estimate_K_scale_for_food()`メソッドを使用")
content.append("")
content.append("### 既知の問題と制限")
content.append("1. **モデルの限界**: UniDepth v2は食品撮影用に訓練されていない")
content.append("2. **汎用性の欠如**: 単一のK_scale_factorでは全画像に対応不可")
content.append("3. **GitHub Issue #105**: UniDepthのカメラパラメータ処理に既知の問題")
content.append("")
content.append("### 推奨事項")
content.append("- 食品体積推定には専用の深度推定モデルの開発または他の手法の検討を推奨")
content.append("- 暫定的な解決策として、深度に基づく適応的K_scale調整を使用")
content.append("")
content.append("---")
content.append("")
content.append("*このドキュメントは`create_unidepth_bundle.py`によって自動生成されました。*")

# ファイルに書き込み
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(content))

print(f"✅ {output_file} を作成しました")

# ファイル情報を表示
total_size = 0
total_lines = 0

print("\n含まれるファイル:")
for filepath, description in files_to_bundle:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = f.read()
            size = len(file_content)
            lines = len(file_content.split('\n'))
            total_size += size
            total_lines += lines
            print(f"  - {filepath}: {size:,} bytes, {lines:,} 行")

print(f"\n合計: {total_size:,} bytes, {total_lines:,} 行")

# 出力ファイルのサイズ
with open(output_file, 'r', encoding='utf-8') as f:
    output_size = len(f.read())
print(f"出力ファイル: {output_size:,} bytes")