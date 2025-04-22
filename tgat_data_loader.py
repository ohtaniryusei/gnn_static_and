# utils/tgat_data_loader.py

import pandas as pd
import torch
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
from torch_geometric.data import Data
import numpy as np 

def load_pyg_temporal_dataset_from_csv(trans_path, static_path, in_dim, num_snapshots=10):
    """
    CSV から PyG Temporal 用データを作成する
    """
    # ------------------------
    # Step 1: 動的データ（User-Merchant取引）
    # ------------------------
    df = pd.read_csv(trans_path)

    # 時間順に並べて等分割（例：10スナップショット）
    df = df.sort_values('timestamp')
    snapshots = []
    snapshot_size = len(df) // num_snapshots
    for i in range(num_snapshots):
        df_snap = df.iloc[i * snapshot_size : (i + 1) * snapshot_size]

        # ノード数を揃える（例：ユーザー100人）
        users = df_snap['user_id'].unique()
        merchants = df_snap['merchant_id'].unique()
        num_users = max(users) + 1
        num_merchants = max(merchants) + 1
        num_nodes = max(num_users, num_merchants + 100)  # 商人もノード扱いにして大きめに

        # edge_index
        edge_index = torch.tensor(df_snap[['user_id', 'merchant_id']].values.T, dtype=torch.long)
        edge_index[1] += num_users  # 商人IDをノード空間の後半に押し込む

        # edge_attr（今はなし）
        edge_weight = None

        # ノード特徴（ここではランダム生成）
        x = torch.randn(num_nodes, in_dim)


        # ラベル（ユーザーごと）
        y = torch.zeros(num_nodes, 1)
        y_np = y.numpy()
        for _, row in df_snap.iterrows():
             uid = int(row['user_id'])
             label = float(row['label'])  # 明示的にfloatにしてもOK
             y[uid] = label  # デフォルトラベルはユーザーに対応

        snapshots.append((x, edge_index, edge_weight, y_np))

    features = [snap[0] for snap in snapshots]
    edge_indices = [snap[1] for snap in snapshots]
    edge_weights = [snap[2] for snap in snapshots]
    targets = [snap[3] for snap in snapshots]

    dynamic_dataset = DynamicGraphTemporalSignal(
        edge_indices=edge_indices,
        edge_weights=edge_weights,
        features=features,
        targets=targets
    )

    # ------------------------
    # Step 2: 静的データ（User-User送金）
    # ------------------------
    df_static = pd.read_csv(static_path)
    edge_index = torch.tensor(df_static[['source', 'target']].values.T, dtype=torch.long)
    num_static_nodes = dynamic_dataset.features[0].shape[0]
    x = torch.randn(num_static_nodes, in_dim)
    static_graph = Data(x=x, edge_index=edge_index)

    return dynamic_dataset, static_graph
