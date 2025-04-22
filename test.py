# test.py

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from full_model import FullModel_TGCN
from utils.tgat_data_loader import load_pyg_temporal_dataset_from_csv
from static_gnn import StaticGraphEncoder

def evaluate():
    # 設定
    in_dim = 128
    model = FullModel_TGCN(in_dim=in_dim, out_dim=128)
    static_encoder = StaticGraphEncoder(in_channels=in_dim, hidden_channels=128, out_channels=128)

    # データ読み込み
    dataset, static_graph = load_pyg_temporal_dataset_from_csv(
        trans_path='/workspaces/gnn_static_and/gnn_research/data/transactions.csv',
        static_path='/workspaces/gnn_static_and/gnn_research/data/transfers.csv',
        in_dim=in_dim
    )

    model.eval()
    aucs = []

    with torch.no_grad():
        for i, snapshot in enumerate(dataset):
            x, edge_index, edge_attr, y = snapshot.x, snapshot.edge_index, snapshot.edge_attr, snapshot.y

            # 静的GNN出力
            h_static = static_encoder(static_graph.x, static_graph.edge_index)

            # モデル出力（TGAT + GraphSAGE + Fusion + Classifier）
            y_pred = model(snapshot, h_static).squeeze()

            # AUC
            auc = roc_auc_score(y.squeeze().cpu().numpy(), y_pred.cpu().numpy())
            aucs.append(auc)

            print(f"[Snapshot {i+1}] AUC: {auc:.4f}")

    print(f"\n✅ 平均AUC（全スナップショット）: {sum(aucs)/len(aucs):.4f}")

if __name__ == '__main__':
    evaluate()
