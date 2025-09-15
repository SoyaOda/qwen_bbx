#!/usr/bin/env python3
"""
Dish IDの時系列相関とデータリークの可能性を分析
"""

import os
from collections import defaultdict
from datetime import datetime

def analyze_dish_ids():
    """dish_idの時系列パターンを分析"""
    
    # クリーンなIDファイルを読み込み
    base_dir = "Finetuning/cleaned_splits_v2"
    
    def read_ids(filepath):
        with open(filepath, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    train_ids = read_ids(os.path.join(base_dir, "clean_train_ids.txt"))
    val_ids = read_ids(os.path.join(base_dir, "clean_val_ids.txt"))
    test_ids = read_ids(os.path.join(base_dir, "clean_test_ids.txt"))
    
    print("="*60)
    print("Dish ID Temporal Analysis")
    print("="*60)
    
    # IDからタイムスタンプを抽出（dish_XXXXXXXXXXの形式）
    def extract_timestamp(dish_id):
        try:
            return int(dish_id.split('_')[1])
        except:
            return None
    
    # 各セットのタイムスタンプ範囲を分析
    for name, ids in [("Train", train_ids), ("Val", val_ids), ("Test", test_ids)]:
        timestamps = [extract_timestamp(id) for id in ids if extract_timestamp(id)]
        if timestamps:
            timestamps.sort()
            
            print(f"\n{name} Set:")
            print(f"  Total dishes: {len(ids)}")
            print(f"  Timestamp range: {timestamps[0]} - {timestamps[-1]}")
            
            # Unix timestampをdatetimeに変換
            try:
                start_date = datetime.fromtimestamp(timestamps[0])
                end_date = datetime.fromtimestamp(timestamps[-1])
                print(f"  Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            except:
                pass
            
            # 連続性の分析（60秒以内の間隔を同一セッションとみなす）
            sessions = []
            current_session = [timestamps[0]]
            
            for i in range(1, len(timestamps)):
                if timestamps[i] - timestamps[i-1] <= 60:  # 60秒以内
                    current_session.append(timestamps[i])
                else:
                    if len(current_session) > 1:
                        sessions.append(current_session)
                    current_session = [timestamps[i]]
            
            if len(current_session) > 1:
                sessions.append(current_session)
            
            print(f"  Detected sessions (dishes within 60s): {len(sessions)}")
            if sessions:
                print(f"  Average dishes per session: {sum(len(s) for s in sessions) / len(sessions):.1f}")
                
                # 最初の数セッションを表示
                for i, session in enumerate(sessions[:3]):
                    if len(session) > 2:
                        print(f"    Session {i+1}: {len(session)} dishes in {session[-1] - session[0]} seconds")
    
    # セット間での時系列の重複をチェック
    print("\n" + "="*60)
    print("Temporal Overlap Analysis")
    print("="*60)
    
    train_timestamps = set([extract_timestamp(id) for id in train_ids if extract_timestamp(id)])
    test_timestamps = set([extract_timestamp(id) for id in test_ids if extract_timestamp(id)])
    val_timestamps = set([extract_timestamp(id) for id in val_ids if extract_timestamp(id)])
    
    # 近い時間のデータが異なるセットに分かれているかチェック
    def find_nearby_timestamps(set1, set2, threshold=300):  # 5分以内
        nearby = []
        for ts1 in set1:
            for ts2 in set2:
                if abs(ts1 - ts2) <= threshold:
                    nearby.append((ts1, ts2, abs(ts1 - ts2)))
        return nearby
    
    train_test_nearby = find_nearby_timestamps(train_timestamps, test_timestamps)
    train_val_nearby = find_nearby_timestamps(train_timestamps, val_timestamps)
    val_test_nearby = find_nearby_timestamps(val_timestamps, test_timestamps)
    
    print(f"Train-Test dishes within 5 minutes: {len(train_test_nearby)}")
    print(f"Train-Val dishes within 5 minutes: {len(train_val_nearby)}")
    print(f"Val-Test dishes within 5 minutes: {len(val_test_nearby)}")
    
    # 連続したdish_idが異なるセットに分かれているケースを探す
    print("\n" + "="*60)
    print("Sequential ID Split Analysis")
    print("="*60)
    
    all_ids_with_split = []
    for id in train_ids:
        all_ids_with_split.append((id, 'train'))
    for id in val_ids:
        all_ids_with_split.append((id, 'val'))
    for id in test_ids:
        all_ids_with_split.append((id, 'test'))
    
    # タイムスタンプでソート
    all_ids_with_split.sort(key=lambda x: extract_timestamp(x[0]) or 0)
    
    # 連続した異なるセットのdishを探す
    consecutive_different = []
    for i in range(1, min(len(all_ids_with_split), 1000)):  # 最初の1000個のみチェック
        id1, split1 = all_ids_with_split[i-1]
        id2, split2 = all_ids_with_split[i]
        
        ts1 = extract_timestamp(id1)
        ts2 = extract_timestamp(id2)
        
        if ts1 and ts2 and abs(ts2 - ts1) <= 60 and split1 != split2:
            consecutive_different.append((id1, split1, id2, split2, ts2 - ts1))
    
    print(f"Found {len(consecutive_different)} cases of consecutive dishes in different splits")
    if consecutive_different:
        print("\nFirst 10 examples:")
        for i, (id1, split1, id2, split2, diff) in enumerate(consecutive_different[:10]):
            print(f"  {i+1}. {id1} ({split1}) -> {id2} ({split2}), diff: {diff}s")
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("="*60)
    print("⚠️  IMPORTANT: Dishes recorded within minutes of each other")
    print("likely represent the SAME or SIMILAR meals from different angles")
    print("or incremental additions. This creates implicit data leakage")
    print("when they are split across train/val/test sets!")
    
    return consecutive_different

if __name__ == "__main__":
    analyze_dish_ids()