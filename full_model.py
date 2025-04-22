# full_model_tgat.py
import torch
import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import TGCN
from static_gnn import StaticGraphEncoder
from fusion import FusionLayer
from classifier import CreditRiskClassifier

class FullModel_TGCN(nn.Module):
    def __init__(self, in_dim=128, out_dim=128):
        super().__init__()
        self.tgat = TGCN(in_channels=in_dim, out_channels=out_dim)
        self.static_encoder = StaticGraphEncoder(in_channels=in_dim, hidden_channels=128, out_channels=out_dim)
        self.fusion = FusionLayer(out_dim)
        self.classifier = CreditRiskClassifier(out_dim)

    def forward(self, data_tgcn, h_static):
        """
        data_tgat: PyG Temporal data object（動的グラフの時系列スナップショット）
        h_static: Tensor [B, D] - 静的GNNの出力（バッチ単位）
        """
        h_dynamic = self.tgat(data_tgcn.x, data_tgcn.edge_index, data_tgcn.edge_attr)  # [B, D]
        h_fused = self.fusion(h_static, h_dynamic)
        y_pred = self.classifier(h_fused)
        return y_pred
