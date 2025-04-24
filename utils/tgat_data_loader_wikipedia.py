# utils/tgat_data_loader_wikipedia.py

from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
from torch_geometric.data import Data

def load_wikipedia_dataset():
    loader = WikiMathsDatasetLoader()
    dataset = loader.get_dataset()
    static_graph = dataset[0]  # snapshotから代表構造を取る
    return dataset, static_graph

    # 静的グラフを構築（すべてのsnapshotの最初の構造で代表化）
    first = dataset[0]
    static_graph = Data(x=first.x, edge_index=first.edge_index)

    return dataset, static_graph
