import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv

from utils import *

class ColorGNN(nn.Module):
    def __init__(self, feature_size):
        super(ColorGNN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(feature_size, 512)
        self.activation1 = nn.ReLU()
        self.conv2 = GCNConv(512, 256)
        self.activation2 = nn.ReLU()
        self.conv3 = GCNConv(256, 64)
        self.activation3 = nn.ReLU()
        self.color_picker = Linear(64, 3)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.activation1(h)
        h = self.conv2(h, edge_index)
        h = self.activation2(h)
        h = self.conv3(h, edge_index)
        h = self.activation3(h)  # Final GNN embedding space.
        
        out = self.color_picker(h)

        return out