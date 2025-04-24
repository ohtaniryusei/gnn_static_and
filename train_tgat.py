# train_tgat.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from full_model import FullModel
from utils.tgat_data_loader import load_pyg_temporal_dataset_from_csv
from static_gnn import StaticGraphEncoder

def train():
    # モデル定義
    in_dim = 128
    model = FullModel(in_dim=in_dim, out_dim=128)
    static_encoder = StaticGraphEncoder(in_channels=in_dim, hidden_channels=128, out_channels=128)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # データ読み込み（PyG Temporal）
    dataset, static_graph = load_pyg_temporal_dataset_from_csv(
        trans_path='/workspaces/gnn_static_and/data/transactions.csv',
        static_path='/workspaces/gnn_static_and/data/transfers.csv',
        in_dim=in_dim
    )

    model.train()
    auc_scores = []

    for epoch in range(20):
        total_loss = 0.0
        epoch_aucs = []

        for snapshot in dataset:
            x, edge_index, edge_attr, y = snapshot.x, snapshot.edge_index, snapshot.edge_attr, snapshot.y

            # 静的GNNによるユーザー埋め込み（送金グラフ）
            h_static = static_encoder(static_graph.x, static_graph.edge_index)

            # --- 以下を追加(仮) ---
            B = snapshot.x.shape[0]
            N = 10  # 仮の近傍数（論文もこれくらい）

            # 各ユーザーの取引時刻（例：ランダム整数で代用）
            t = torch.randint(0, 1000, (B,))  # [B]

            # 各ユーザーが持つ近傍との取引時刻（仮に10個）
            t_prime = torch.randint(0, 1000, (B, N))  # [B, N]

            # semantic_feat: 時間的意味情報（祝日、週末など）をダミー化（例：8次元）
            semantic_feat = torch.randn(B, N, 8)  # [B, N, d_se]

            # TGAT + GraphSAGE + Fusion + MLP
            y_pred = model(snapshot, h_static, t, t_prime, semantic_feat).squeeze()

            loss = criterion(y_pred, y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # AUC評価
            auc = roc_auc_score(y.squeeze().cpu().numpy(), y_pred.detach().cpu().numpy())
            epoch_aucs.append(auc)
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | AUC: {sum(epoch_aucs)/len(epoch_aucs):.4f}")
        auc_scores.append(sum(epoch_aucs) / len(epoch_aucs))

    print(f"\n✅ 最終平均AUC: {sum(auc_scores)/len(auc_scores):.4f}")

if __name__ == '__main__':
    train()
