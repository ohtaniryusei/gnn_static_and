# test_wiki.py

import torch
from sklearn.metrics import roc_auc_score
from full_model import FullModel
from static_gnn import StaticGraphEncoder
from utils.tgat_data_loader_wikipedia import load_wikipedia_dataset


def evaluate():
    in_dim = 8
    out_dim = 128
    time_config = (16, 16, 8, 32)

    model = FullModel(in_dim=in_dim, out_dim=out_dim, time_config=time_config)
    model.load_state_dict(torch.load("checkpoints/wiki_model.pt"))
    model.eval()

    static_encoder = StaticGraphEncoder(in_channels=in_dim, hidden_channels=128, out_channels=out_dim)

    dataset, static_graph = load_wikipedia_dataset()

    aucs = []
    with torch.no_grad():
        for i, snapshot in enumerate(dataset):
            x, edge_index, y = snapshot.x, snapshot.edge_index, snapshot.y

            h_static = static_encoder(static_graph.x, static_graph.edge_index)

            B = x.shape[0]
            N = 10
            t = torch.randint(0, 1000, (B,))
            t_prime = torch.randint(0, 1000, (B, N))
            semantic_feat = torch.randn(B, N, 8)

            snapshot.edge_attr = None
            y_pred = model(snapshot, h_static, t, t_prime, semantic_feat).squeeze()

            y_true = y.squeeze().cpu().numpy().astype(int)
            y_score = y_pred.cpu().numpy()

            mask = (y_true == 0) | (y_true == 1)
            if mask.sum() == 0:
                continue
            auc = roc_auc_score(y_true[mask], y_score[mask])
            aucs.append(auc)

            print(f"[Snapshot {i+1}] AUC: {auc:.4f}")

    if aucs:
        print(f"\n✅ 平均AUC（WikiMaths）: {sum(aucs)/len(aucs):.4f}")
    else:
        print("⚠️ 有効なスナップショットがありませんでした")


if __name__ == '__main__':
    evaluate()
