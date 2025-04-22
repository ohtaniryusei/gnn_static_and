# train_tgat.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from full_model import FullModel_TGCN
from utils.tgat_data_loader import load_pyg_temporal_dataset_from_csv
from static_gnn import StaticGraphEncoder

def train():
    # モデル定義
    in_dim = 128
    model = FullModel_TGCN(in_dim=in_dim, out_dim=128)
    static_encoder = StaticGraphEncoder(in_channels=in_dim, hidden_channels=128, out_channels=128)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # データ読み込み（PyG Temporal）
    dataset, static_graph = load_pyg_temporal_dataset_from_csv(
        trans_path='/workspaces/gnn_static_and/gnn_research/data/transactions.csv',
        static_path='/workspaces/gnn_static_and/gnn_research/data/transfers.csv',
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

            # TGAT + GraphSAGE + Fusion + MLP
            y_pred = model(snapshot, h_static).squeeze()

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
