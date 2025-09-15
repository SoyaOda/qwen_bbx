#!/usr/bin/env python3
"""
時系列的に近いdish_idをグループ化して適切なデータ分割を作成
同じ料理セッション（5分以内）のデータは同じセットに配置
"""

import os
from collections import defaultdict
from datetime import datetime
import random

def create_temporal_aware_splits():
    """時系列を考慮した新しいデータ分割を作成"""
    
    # オリジナルのIDを読み込み
    split_dir = "nutrition5k/nutrition5k_dataset/dish_ids/splits"
    
    def read_ids(filepath):
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    # depth_idsのみを使用（depthデータがあるもののみ）
    all_train_ids = read_ids(os.path.join(split_dir, "depth_train_ids.txt"))
    all_test_ids = read_ids(os.path.join(split_dir, "depth_test_ids.txt"))
    
    all_ids = all_train_ids + all_test_ids
    print(f"Total dishes with depth data: {len(all_ids)}")
    
    # IDからタイムスタンプを抽出
    def extract_timestamp(dish_id):
        try:
            return int(dish_id.split('_')[1])
        except:
            return None
    
    # dish_idをタイムスタンプでソート
    ids_with_timestamps = [(id, extract_timestamp(id)) for id in all_ids if extract_timestamp(id)]
    ids_with_timestamps.sort(key=lambda x: x[1])
    
    # セッション（5分以内の連続した撮影）をグループ化
    sessions = []
    current_session = [ids_with_timestamps[0]]
    
    for i in range(1, len(ids_with_timestamps)):
        current_id, current_ts = ids_with_timestamps[i]
        prev_id, prev_ts = ids_with_timestamps[i-1]
        
        # 5分（300秒）以内なら同じセッション
        if current_ts - prev_ts <= 300:
            current_session.append((current_id, current_ts))
        else:
            sessions.append(current_session)
            current_session = [(current_id, current_ts)]
    
    if current_session:
        sessions.append(current_session)
    
    print(f"\nDetected {len(sessions)} recording sessions")
    print(f"Average dishes per session: {sum(len(s) for s in sessions) / len(sessions):.1f}")
    
    # セッション統計
    session_sizes = [len(s) for s in sessions]
    print(f"Session size distribution:")
    print(f"  Min: {min(session_sizes)}, Max: {max(session_sizes)}")
    print(f"  Sessions with 1 dish: {sum(1 for s in session_sizes if s == 1)}")
    print(f"  Sessions with 2-5 dishes: {sum(1 for s in session_sizes if 2 <= s <= 5)}")
    print(f"  Sessions with >5 dishes: {sum(1 for s in session_sizes if s > 5)}")
    
    # セッションを単位として分割（シャッフルしてランダムに分割）
    random.seed(42)
    random.shuffle(sessions)
    
    # 70% train, 15% val, 15% test の比率で分割
    n_sessions = len(sessions)
    n_train = int(n_sessions * 0.7)
    n_val = int(n_sessions * 0.15)
    
    train_sessions = sessions[:n_train]
    val_sessions = sessions[n_train:n_train + n_val]
    test_sessions = sessions[n_train + n_val:]
    
    # 各セットのIDリストを作成
    train_ids = [id for session in train_sessions for id, _ in session]
    val_ids = [id for session in val_sessions for id, _ in session]
    test_ids = [id for session in test_sessions for id, _ in session]
    
    print(f"\nNew split statistics:")
    print(f"  Train: {len(train_ids)} dishes from {len(train_sessions)} sessions")
    print(f"  Val: {len(val_ids)} dishes from {len(val_sessions)} sessions")
    print(f"  Test: {len(test_ids)} dishes from {len(test_sessions)} sessions")
    
    # 時系列の重複チェック
    def check_temporal_overlap(set1_ids, set2_ids, set1_name, set2_name):
        set1_timestamps = [extract_timestamp(id) for id in set1_ids]
        set2_timestamps = [extract_timestamp(id) for id in set2_ids]
        
        nearby_count = 0
        for ts1 in set1_timestamps:
            for ts2 in set2_timestamps:
                if abs(ts1 - ts2) <= 300:  # 5分以内
                    nearby_count += 1
                    break
        
        return nearby_count
    
    print(f"\nTemporal overlap check (dishes within 5 minutes):")
    print(f"  Train-Val: {check_temporal_overlap(train_ids, val_ids, 'Train', 'Val')}")
    print(f"  Train-Test: {check_temporal_overlap(train_ids, test_ids, 'Train', 'Test')}")
    print(f"  Val-Test: {check_temporal_overlap(val_ids, test_ids, 'Val', 'Test')}")
    
    # 新しい分割を保存
    output_dir = "Finetuning/temporal_aware_splits"
    os.makedirs(output_dir, exist_ok=True)
    
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        filepath = os.path.join(output_dir, f"temporal_{name}_ids.txt")
        with open(filepath, 'w') as f:
            for id in sorted(ids):
                f.write(f"{id}\n")
        print(f"Saved {filepath}")
    
    # セッション情報も保存
    session_info_path = os.path.join(output_dir, "session_info.txt")
    with open(session_info_path, 'w') as f:
        f.write(f"Total sessions: {len(sessions)}\n")
        f.write(f"Train sessions: {len(train_sessions)}\n")
        f.write(f"Val sessions: {len(val_sessions)}\n")
        f.write(f"Test sessions: {len(test_sessions)}\n\n")
        
        f.write("Sample sessions:\n")
        for i, session in enumerate(sessions[:5]):
            f.write(f"Session {i+1}: {len(session)} dishes\n")
            for id, ts in session[:3]:  # 最初の3つのみ
                f.write(f"  {id} (timestamp: {ts})\n")
            if len(session) > 3:
                f.write(f"  ... and {len(session) - 3} more\n")
    
    print(f"\n✅ Created temporal-aware data splits in {output_dir}")
    print("These splits ensure that dishes from the same recording session")
    print("(likely the same or similar meals) stay together in the same set.")
    
    return train_ids, val_ids, test_ids

if __name__ == "__main__":
    create_temporal_aware_splits()