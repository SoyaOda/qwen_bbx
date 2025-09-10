#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UniDepth v2 修正版テストスクリプト
正しい逆投影式と前処理を使用して体積を推定
"""
import numpy as np
import torch
import cv2
from PIL import Image
from unidepth.models import UniDepthV2
import glob
import os

def unproject_depth_to_xyz(depth, K):
    """
    深度マップから3D点群を生成（正しい逆投影式を使用）
    
    Args:
        depth: (H,W) 深度マップ[m]
        K: (3,3) カメラ内部パラメータ
    
    Returns:
        (H,W,3) 3D点群（カメラ座標系）
    """
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    
    # 正しい逆投影式（K^{-1}を使用）
    Z = depth
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    
    return np.stack([X, Y, Z], axis=-1)

def pixel_area_from_depth(depth, K):
    """
    各ピクセルが表す実世界の面積を計算
    
    Args:
        depth: (H,W) 深度マップ[m]
        K: (3,3) カメラ内部パラメータ
    
    Returns:
        (H,W) ピクセル面積マップ[m²]
    """
    fx, fy = K[0,0], K[1,1]
    return (depth * depth) / (fx * fy)

def fit_plane_ransac(points_xyz, mask, dist_th=0.006, iters=2000):
    """
    RANSAC法で平面を当てはめ
    
    Args:
        points_xyz: (H,W,3) 3D点群
        mask: (H,W) bool 平面候補点のマスク
        dist_th: インライア閾値[m]
        iters: 最大反復回数
    
    Returns:
        (n, d): 平面パラメータ（n·X + d = 0）
    """
    P = points_xyz[mask].reshape(-1, 3)  # 候補点を抽出
    N = P.shape[0]
    
    if N < 100:
        raise RuntimeError(f"平面RANSAC: 候補点が不足 ({N} < 100)")
    
    rng = np.random.default_rng(0)
    best_inl, best = 0, None
    
    for _ in range(iters):
        # 3点をランダムに選択
        i = rng.choice(N, 3, replace=False)
        
        # 3点から平面を計算
        A = np.c_[P[i], np.ones(3)]
        try:
            # 最小二乗法で平面方程式を求める
            # ax + by + cz + d = 0
            sol = np.linalg.lstsq(A, np.zeros(3), rcond=None)[0]
            a, b, c, d = sol
        except:
            continue
        
        # 法線ベクトルを正規化
        n = np.array([a, b, c])
        nrm = np.linalg.norm(n)
        if nrm < 1e-9:
            continue
        n = n / nrm
        d = d / nrm
        
        # インライアをカウント
        dist = np.abs(P @ n + d)
        inl = (dist < dist_th).sum()
        
        if inl > best_inl:
            best_inl, best = inl, (n, d)
    
    if best is None:
        raise RuntimeError("平面を見つけられませんでした")
    
    return best

def estimate_volume_liters(image_path, mask_path=None, device="cuda"):
    """
    画像から食品の体積を推定（修正版）
    
    Args:
        image_path: 入力画像のパス
        mask_path: 食品マスクのパス（オプション）
        device: 実行デバイス
    
    Returns:
        体積[L]
    """
    # 1) デバイス設定
    device = torch.device(device if (device=="cuda" and torch.cuda.is_available()) else "cpu")
    print(f"デバイス: {device}")
    
    # 2) モデルロード
    print("UniDepth v2 モデルをロード中...")
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(device)
    model.eval()
    
    # 3) 画像読み込み（正規化はモデル内部で実施）
    print(f"画像を読み込み中: {image_path}")
    rgb_pil = Image.open(image_path).convert("RGB")
    rgb_np = np.array(rgb_pil)
    
    # RGBテンソルに変換（uint8のまま）
    rgb = torch.from_numpy(rgb_np).permute(2,0,1).unsqueeze(0).to(device)
    
    # 4) 推論実行（Kは推定させる）
    print("深度推定中...")
    with torch.inference_mode():
        pred = model.infer(rgb)  # 正規化はモデル内部で行われる
    
    # 深度とKを取得
    depth = pred["depth"].squeeze().detach().cpu().numpy()   # [m]
    K = pred["intrinsics"].squeeze().detach().cpu().numpy()  # (3,3)
    conf = pred.get("confidence", None)
    if conf is not None:
        conf = conf.squeeze().detach().cpu().numpy()
    
    print(f"深度範囲: {depth.min():.3f} - {depth.max():.3f} m")
    print(f"カメラ内部パラメータ:")
    print(f"  fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"  cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    
    # 5) 逆投影（検証付き）
    print("3D点群を生成中...")
    xyz = unproject_depth_to_xyz(depth, K)
    
    # Z成分の一致を検証
    z_error = np.mean(np.abs(xyz[...,2] - depth))
    print(f"Z成分検証: 平均誤差 = {z_error:.6f} m")
    if z_error > 1e-6:
        print("警告: Z成分が深度と一致しません！")
    
    # 6) マスク読み込み
    H, W = depth.shape
    if mask_path and os.path.exists(mask_path):
        print(f"マスクを読み込み中: {mask_path}")
        mask = cv2.imread(mask_path, 0) > 127
        food_mask = mask
        cand_mask = ~mask  # 皿/卓面候補
    else:
        print("マスクなし: 画像中央を食品、周辺を皿として仮定")
        # 画像中央を食品として仮定
        cy, cx = H // 2, W // 2
        r = min(H, W) // 6
        yy, xx = np.ogrid[:H, :W]
        food_mask = ((yy - cy)**2 + (xx - cx)**2) <= r**2
        cand_mask = ~food_mask
    
    # 7) 平面フィッティング
    print("皿/卓面を推定中...")
    try:
        plane = fit_plane_ransac(xyz, cand_mask, dist_th=0.006)
        n, d = plane
        print(f"平面パラメータ: n=[{n[0]:.3f}, {n[1]:.3f}, {n[2]:.3f}], d={d:.3f}")
    except Exception as e:
        print(f"平面推定失敗: {e}")
        # フォールバック: 最も低い点を基準
        min_z = depth[food_mask].min() if food_mask.any() else depth.min()
        print(f"フォールバック: 最小深度 {min_z:.3f}m を基準面とします")
        n = np.array([0, 0, 1])
        d = -min_z
    
    # 8) 高さマップ計算
    h = xyz @ n + d  # 平面からの符号付き距離
    
    # 法線が上向きの場合、符号を調整
    if n[2] > 0:
        h = -h
    
    h = np.clip(h, 0, None)  # 負の高さは0にクリップ
    
    # 9) ピクセル面積計算
    a_pix = pixel_area_from_depth(depth, K)
    
    # a_pixのオーダー確認
    mean_apix = a_pix[food_mask].mean() if food_mask.any() else a_pix.mean()
    print(f"平均ピクセル面積: {mean_apix:.2e} m²/px")
    
    # 10) 体積積分
    vol_m3 = np.sum(h[food_mask] * a_pix[food_mask])
    vol_L = vol_m3 * 1000.0  # m³ → L
    
    # 統計情報
    if food_mask.any():
        h_mean = h[food_mask].mean() * 1000  # mm
        h_max = h[food_mask].max() * 1000    # mm
        pixels = food_mask.sum()
        print(f"\n結果:")
        print(f"  ピクセル数: {pixels}")
        print(f"  平均高さ: {h_mean:.1f} mm")
        print(f"  最大高さ: {h_max:.1f} mm")
        print(f"  体積: {vol_L:.2f} L ({vol_L*1000:.1f} mL)")
    
    return vol_L

def test_with_sample_images():
    """テスト画像で体積推定をテスト"""
    # テスト画像を検索
    test_images = glob.glob("test_images/*.jpg")
    if not test_images:
        test_images = glob.glob("test_images/*.png")
    
    if not test_images:
        print("テスト画像が見つかりません")
        return
    
    print(f"\n{len(test_images)}枚のテスト画像を処理します\n")
    
    for img_path in test_images[:3]:  # 最初の3枚だけテスト
        print("="*60)
        basename = os.path.basename(img_path)
        print(f"画像: {basename}")
        print("="*60)
        
        try:
            # マスクファイルを探す（あれば使用）
            stem = os.path.splitext(basename)[0]
            mask_paths = [
                f"outputs/sam2/masks/{stem}_det00_*.png",
                f"outputs/sam2/masks/{stem}_det01_*.png",
                f"outputs/sam2/masks/{stem}_det02_*.png",
            ]
            
            mask_found = None
            for pattern in mask_paths:
                matches = glob.glob(pattern)
                if matches:
                    mask_found = matches[0]
                    break
            
            # 体積推定
            vol = estimate_volume_liters(img_path, mask_found)
            
            print(f"\n最終推定体積: {vol:.2f} L")
            
            # 現実的な範囲かチェック
            if vol < 0.01:
                print("⚠️ 体積が小さすぎます（< 10mL）")
            elif vol > 2.0:
                print("⚠️ 体積が大きすぎます（> 2L）")
            else:
                print("✓ 体積は現実的な範囲です")
                
        except Exception as e:
            print(f"エラー: {e}")
            import traceback
            traceback.print_exc()
        
        print()

if __name__ == "__main__":
    print("UniDepth v2 修正版テスト")
    print("="*60)
    test_with_sample_images()