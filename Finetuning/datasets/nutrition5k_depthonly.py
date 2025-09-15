import os
import random
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class Nutrition5KDepthOnly(Dataset):
    """
    Nutrition5K overhead RGB-D を読み込み、RGB(PIL)・depth[m]・valid(>0) を返す。
    - 教師は深度のみ（マスク・体積は一切扱わない）
    - depth_raw.png は 16bit で 1 unit = 1e-4 [m]（引数で可変）
    """
    def __init__(self, root_dir: str, ids: List[str],
                 depth_scale: float = 1e-4, max_depth_m: float = 10.0):
        self.root = root_dir
        self.ids = ids
        self.depth_scale = float(depth_scale)
        self.max_depth = float(max_depth_m)

    def __len__(self): 
        return len(self.ids)

    def _paths(self, dish_id: str) -> Tuple[str, str]:
        # dish_idはすでに"dish_"で始まっているので、そのまま使用
        base = os.path.join(
            self.root, "imagery", "realsense_overhead", dish_id)
        return os.path.join(base, "rgb.png"), os.path.join(base, "depth_raw.png")

    @staticmethod
    def _read_rgb(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _read_depth_m(self, path: str) -> np.ndarray:
        try:
            # PILで開く前にファイルが正常か確認
            with open(path, 'rb') as f:
                # PNGヘッダーの確認（最初の8バイト）
                header = f.read(8)
                if len(header) < 8:
                    raise ValueError(f"Invalid PNG file: {path}")
            
            arr = np.array(Image.open(path))  # uint16 想定
            if arr.dtype != np.uint16: 
                arr = arr.astype(np.uint16)
            depth_m = arr.astype(np.float32) * self.depth_scale
            # 無効値処理
            depth_m[(depth_m <= 0) | (depth_m > self.max_depth)] = 0.0
            return depth_m  # [H,W] in meters
        except Exception as e:
            raise IOError(f"Failed to read depth image {path}: {e}")

    def __getitem__(self, idx):
        did = self.ids[idx]
        rgb_path, dep_path = self._paths(did)
        
        try:
            img = self._read_rgb(rgb_path)
            depth_m = self._read_depth_m(dep_path)
            depth_t = torch.from_numpy(depth_m).unsqueeze(0)  # [1,H,W]
            valid = (depth_t > 0).float()
            return img, depth_t, valid
        except Exception as e:
            # エラーが発生した場合は次のサンプルを試す
            print(f"Warning: Failed to load {did}: {e}")
            # 次のインデックスを試す（ループを避けるため最大で全体の10%まで）
            max_attempts = min(10, len(self.ids) // 10)
            for attempt in range(1, max_attempts):
                next_idx = (idx + attempt) % len(self.ids)
                next_did = self.ids[next_idx]
                next_rgb_path, next_dep_path = self._paths(next_did)
                try:
                    img = self._read_rgb(next_rgb_path)
                    depth_m = self._read_depth_m(next_dep_path)
                    depth_t = torch.from_numpy(depth_m).unsqueeze(0)
                    valid = (depth_t > 0).float()
                    return img, depth_t, valid
                except:
                    continue
            
            # すべて失敗した場合はダミーデータを返す
            print(f"Error: Could not load any valid sample after {max_attempts} attempts")
            dummy_img = Image.new('RGB', (640, 480), color='black')
            dummy_depth = torch.zeros(1, 480, 640)
            dummy_valid = torch.zeros(1, 480, 640)
            return dummy_img, dummy_depth, dummy_valid

def load_split_ids(n5k_root: str, val_ratio: float = 0.1, seed: int = 42, use_clean: bool = True, use_temporal: bool = True):
    """ dish_ids/splits/depth_train_ids.txt, depth_test_ids.txt を読む """
    
    # 時系列考慮型の新しい分割を優先的に使用
    if use_temporal:
        temporal_dir = os.path.join(os.path.dirname(__file__), "..", "temporal_aware_splits")
        if os.path.exists(temporal_dir):
            temporal_train = os.path.join(temporal_dir, "temporal_train_ids.txt")
            temporal_val = os.path.join(temporal_dir, "temporal_val_ids.txt")
            temporal_test = os.path.join(temporal_dir, "temporal_test_ids.txt")
            
            if all(os.path.exists(f) for f in [temporal_train, temporal_val, temporal_test]):
                def _read(p):
                    with open(p, "r") as f:
                        return [ln.strip() for ln in f if ln.strip()]
                
                print(f"Using temporal-aware split files (no data leak)")
                return _read(temporal_train), _read(temporal_val), _read(temporal_test)
    
    # クリーンなIDファイルが存在する場合はそれを使用
    if use_clean:
        # まず最新版のクリーンファイルを探す
        clean_dirs = [
            os.path.join(os.path.dirname(__file__), "..", "cleaned_splits_v2"),
            os.path.join(os.path.dirname(__file__), "..", "cleaned_splits"),
        ]
        
        for clean_dir in clean_dirs:
            if os.path.exists(clean_dir):
                clean_train = os.path.join(clean_dir, "clean_train_ids.txt")
                clean_val = os.path.join(clean_dir, "clean_val_ids.txt")
                clean_test = os.path.join(clean_dir, "clean_test_ids.txt")
                
                if all(os.path.exists(f) for f in [clean_train, clean_val, clean_test]):
                    def _read(p):
                        with open(p, "r") as f:
                            return [ln.strip() for ln in f if ln.strip()]
                    
                    print(f"Using cleaned split files from {os.path.basename(clean_dir)}")
                    return _read(clean_train), _read(clean_val), _read(clean_test)
    
    # オリジナルのsplitファイルを読み込み
    split_dir = os.path.join(n5k_root, "dish_ids", "splits")
    
    def _read(p):
        with open(p, "r") as f:
            return [ln.strip() for ln in f if ln.strip()]
    
    train_all = _read(os.path.join(split_dir, "depth_train_ids.txt"))
    test_ids  = _read(os.path.join(split_dir, "depth_test_ids.txt"))
    
    # 存在しないIDを除外（安全のため）
    def filter_existing(ids):
        existing = []
        for dish_id in ids:
            base_path = os.path.join(n5k_root, "imagery", "realsense_overhead", dish_id)
            rgb_path = os.path.join(base_path, "rgb.png")
            depth_path = os.path.join(base_path, "depth_raw.png")
            # 両方のファイルが存在し、サイズが0でないことを確認
            if (os.path.exists(rgb_path) and os.path.getsize(rgb_path) > 0 and
                os.path.exists(depth_path) and os.path.getsize(depth_path) > 0):
                existing.append(dish_id)
        return existing
    
    train_all = filter_existing(train_all)
    test_ids = filter_existing(test_ids)
    
    rnd = random.Random(seed)
    rnd.shuffle(train_all)
    n_val = max(1, int(len(train_all) * val_ratio))
    val_ids = train_all[:n_val]
    train_ids = train_all[n_val:]
    
    return train_ids, val_ids, test_ids