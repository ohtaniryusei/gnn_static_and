# utils/mooc_data_loader.py

from torch_geometric_temporal.dataset import MOOC
from torch_geometric.data import Data

def load_mooc_dataset():
    dataset = MOOC(root="data/mooc")

    # 1スナップショット目の構造を静的グラフとして使用
    first = dataset[0]
    static_graph = Data(x=first.x, edge_index=first.edge_index)

    return dataset, static_graph
