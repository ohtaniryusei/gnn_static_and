# train_wiki.py

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from full_model import FullModel
from static_gnn import StaticGraphEncoder
from utils.tgat_data_loader_wikipedia import load_wikipedia_dataset


def train():
    in_dim = 8  # WikiMaths のノード特徴数
    out_dim = 128
    time_config = (16, 16, 8, 32)

    model = FullModel(in_dim=in_dim, out_dim=out_dim, time_config=time_config)
    static_encoder = StaticGraphEncoder(in_channels=in_dim, hidden_channels=128, out_channels=out_dim)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset, static_graph = load_wikipedia_dataset()
    

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        aucs = []

        for snapshot in dataset:
            print(torch.unique(snapshot.y))  # snapshot = dataset[0]
            x, edge_index, y = snapshot.x, snapshot.edge_index, snapshot.y

            h_static = static_encoder(static_graph.x, static_graph.edge_index)

            B = x.shape[0]
            N = 10  # 近傍数（仮）
            t = torch.randint(0, 1000, (B,))
            t_prime = torch.randint(0, 1000, (B, N))
            semantic_feat = torch.randn(B, N, 8)

            snapshot.edge_attr = None
            y_pred = model(snapshot, h_static, t, t_prime, semantic_feat).squeeze()
            y = y.squeeze()

            mask = (y == 0) | (y == 1)
            if mask.sum() == 0:
                continue
            loss = criterion(y_pred[mask], y[mask].float())
            auc = roc_auc_score(y[mask].cpu().numpy().astype(int), y_pred[mask].detach().cpu().numpy())
            aucs.append(auc)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if aucs:
            print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | AUC: {sum(aucs)/len(aucs):.4f}")
        else:
            print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | AUC: N/A (no valid labels)")
            torch.save(model.state_dict(), "checkpoints/wiki_model.pt")
            print("\n✅ モデルを checkpoints/wiki_model.pt に保存しました")


if __name__ == '__main__':
    train()