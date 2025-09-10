# -*- coding: utf-8 -*-
"""
深度マップと高さマップの可視化ユーティリティ
depth_modelプロジェクトと同じビジュアライゼーション方法を実装
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from typing import Optional, Tuple, Dict

def normalize_depth_for_display(
    depth: np.ndarray,
    percentile: Tuple[float, float] = (2, 98)
) -> np.ndarray:
    """
    深度を可視化用に正規化（depth_modelと同じ）
    
    Args:
        depth: 入力深度マップ
        percentile: クリッピング用のパーセンタイル (min, max)
        
    Returns:
        normalized: [0, 1]に正規化された深度
    """
    # 無効な値を除去
    valid_depth = depth[np.isfinite(depth) & (depth > 0)]
    
    if len(valid_depth) == 0:
        return np.zeros_like(depth)
    
    # ロバストな正規化のためのパーセンタイル値を取得
    vmin = np.percentile(valid_depth, percentile[0])
    vmax = np.percentile(valid_depth, percentile[1])
    
    # クリップして正規化
    normalized = np.clip(depth, vmin, vmax)
    normalized = (normalized - vmin) / (vmax - vmin + 1e-6)
    
    # 無効な値を処理
    normalized[~np.isfinite(depth)] = 0
    
    return normalized

def apply_colormap(
    depth: np.ndarray,
    colormap: str = "viridis",
    normalize: bool = True,
    model_name: Optional[str] = None
) -> np.ndarray:
    """
    深度マップにカラーマップを適用（depth_modelと同じ実装）
    
    Args:
        depth: 入力深度マップ
        colormap: Matplotlibカラーマップ名
        normalize: 深度を最初に正規化するか
        model_name: モデル固有の処理用のモデル名（"UniDepth"など）
        
    Returns:
        colored: RGB画像 (H, W, 3), uint8
    """
    # モデル固有の処理（統一ルール: 「近い=明るい」）
    if model_name:
        if model_name == "UniDepth":
            # UniDepthはメートル単位の深度を出力 - 近い=小さい値
            # 逆数化して近い=大きい値に変換
            display_depth = 1.0 / (np.maximum(depth, 1e-6))
        else:
            display_depth = depth
    else:
        display_depth = depth
    
    if normalize:
        display_depth = normalize_depth_for_display(display_depth)
    
    # カラーマップを取得
    if hasattr(cm, 'get_cmap'):
        cmap = cm.get_cmap(colormap)
    else:
        # Matplotlib 3.11以降の新しいAPI
        cmap = plt.colormaps[colormap]
    
    # カラーマップを適用（RGBAを返す）
    colored = cmap(display_depth)
    
    # RGB uint8に変換
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)
    
    return rgb

def colorize_depth(depth: np.ndarray, clip_q: Tuple[float, float] = (0.02, 0.98)) -> np.ndarray:
    """
    深度マップをカラーマップで可視化（後方互換性のため維持）
    
    Args:
        depth: 深度マップ [m]
        clip_q: クリッピング用のパーセンタイル (low, high)
    
    Returns:
        カラーマップ適用済みのBGR画像
    """
    # depth_modelと同じViridisカラーマップを使用
    rgb = apply_colormap(depth, colormap="viridis", model_name="UniDepth")
    # OpenCVはBGRを期待するので変換
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def colorize_height(height: np.ndarray, max_h_m: float = 0.05) -> np.ndarray:
    """
    高さマップをカラーマップで可視化
    
    Args:
        height: 高さマップ [m]
        max_h_m: 表示する最大高さ [m]
    
    Returns:
        カラーマップ適用済みのBGR画像
    """
    h = np.clip(height / max_h_m, 0, 1)
    h8 = (h * 255).astype(np.uint8)
    return cv2.applyColorMap(h8, cv2.COLORMAP_MAGMA)

def create_depth_panel(
    original: np.ndarray,
    depth: np.ndarray,
    height: Optional[np.ndarray] = None,
    confidence: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    元画像、深度マップ、高さマップ、信頼度マップを並べたパネル画像を作成
    
    Args:
        original: 元画像 (BGR)
        depth: 深度マップ [m]
        height: 高さマップ [m]（オプション）
        confidence: 信頼度マップ（オプション）
    
    Returns:
        パネル画像 (BGR)
    """
    H, W = depth.shape
    panels = []
    
    # 元画像
    if original.shape[:2] != (H, W):
        original = cv2.resize(original, (W, H))
    panels.append(original)
    
    # 深度マップ
    depth_viz = colorize_depth(depth)
    panels.append(depth_viz)
    
    # 高さマップ（あれば）
    if height is not None:
        height_viz = colorize_height(height)
        panels.append(height_viz)
    
    # 信頼度マップ（あれば）
    if confidence is not None:
        conf_viz = (np.clip(confidence, 0, 1) * 255).astype(np.uint8)
        conf_viz = cv2.applyColorMap(conf_viz, cv2.COLORMAP_VIRIDIS)
        panels.append(conf_viz)
    
    # パネルを横に並べる
    panel = np.concatenate(panels, axis=1)
    
    # ラベルを追加
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    
    labels = ["Original", "Depth"]
    if height is not None:
        labels.append("Height")
    if confidence is not None:
        labels.append("Confidence")
    
    for i, label in enumerate(labels):
        x = i * W + 10
        y = 30
        cv2.putText(panel, label, (x, y), font, font_scale, color, thickness)
    
    return panel

def save_depth_as_16bit_png(
    depth: np.ndarray,
    save_path: str,
    max_depth: Optional[float] = None
) -> None:
    """
    深度を16ビットPNGとして保存（depth_modelと同じ）
    
    Args:
        depth: 入力深度マップ
        save_path: PNG保存パス
        max_depth: スケーリング用の最大深度
    """
    if max_depth is None:
        valid_depth = depth[np.isfinite(depth) & (depth > 0)]
        if len(valid_depth) > 0:
            max_depth = np.percentile(valid_depth, 99)
        else:
            max_depth = 1.0
    
    # 16ビット範囲にスケーリング
    depth_scaled = np.clip(depth / max_depth, 0, 1)
    depth_16bit = (depth_scaled * 65535).astype(np.uint16)
    
    # PNGとして保存
    Image.fromarray(depth_16bit, mode='I;16').save(save_path)

def create_comparison_panel(
    rgb_image: np.ndarray,
    depth_maps: Dict[str, np.ndarray],
    save_path: str,
    colormap: str = "viridis"
) -> None:
    """
    RGBと深度マップの比較パネルを作成（depth_modelと同じ）
    
    Args:
        rgb_image: 入力RGB画像
        depth_maps: モデル名 -> 深度マップの辞書
        save_path: パネル保存パス
        colormap: 深度可視化用のカラーマップ
    """
    n_models = len(depth_maps)
    fig, axes = plt.subplots(2, (n_models + 1) // 2 + 1, 
                            figsize=(5 * ((n_models + 1) // 2 + 1), 10))
    
    # 1次元配列に平坦化
    axes = axes.flatten()
    
    # RGB画像を表示
    axes[0].imshow(rgb_image)
    axes[0].set_title("Input RGB", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # モデル固有の処理で深度マップを表示
    for idx, (model_name, depth) in enumerate(depth_maps.items(), 1):
        colored_depth = apply_colormap(depth, colormap=colormap, model_name=model_name)
        axes[idx].imshow(colored_depth)
        axes[idx].set_title(model_name, fontsize=12)
        axes[idx].axis('off')
    
    # 未使用のサブプロットを非表示
    for idx in range(n_models + 1, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_depth_as_ply(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    output_path: str = "output.ply"
) -> None:
    """
    3D点群をPLY形式で保存
    
    Args:
        points: 3D点群 (3, H, W) または (N, 3)
        colors: 色情報 (3, H, W) または (N, 3)、0-255の範囲
        output_path: 出力ファイルパス
    """
    # 点群を (N, 3) 形式に変換
    if points.shape[0] == 3 and len(points.shape) == 3:
        # (3, H, W) -> (N, 3)
        points = points.reshape(3, -1).T
    
    # 有効な点のみを抽出
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    
    if colors is not None:
        if colors.shape[0] == 3 and len(colors.shape) == 3:
            colors = colors.reshape(3, -1).T
        colors = colors[valid_mask]
    
    # PLYヘッダー
    num_points = points.shape[0]
    header = f"""ply
format ascii 1.0
element vertex {num_points}
property float x
property float y
property float z"""
    
    if colors is not None:
        header += """
property uchar red
property uchar green
property uchar blue"""
    
    header += "\nend_header\n"
    
    # ファイルに書き込み
    with open(output_path, 'w') as f:
        f.write(header)
        for i in range(num_points):
            line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
            if colors is not None:
                line += f" {int(colors[i, 0])} {int(colors[i, 1])} {int(colors[i, 2])}"
            f.write(line + "\n")
    
    print(f"PLYファイル保存: {output_path} ({num_points} points)")