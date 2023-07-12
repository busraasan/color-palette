import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv

'''
    EMBEDDING SIZE IS TOO BIG
    try without embeddings
    small untrained CNN for embeddings
    may be adding a few layers to resnet and finetune

    MORE THAN ONE COLOR PER NODE
    zero padding to missing colors
    use sequence to learn 3d output (seq-to-seq encoding)

    Convert rgb to cielab before training, do not convert it while calculating the loss
'''
from utils import *

class ColorGNNSmall(nn.Module):
    def __init__(self, feature_size):
        super(ColorGNNSmall, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(feature_size, 16)
        self.activation1 = nn.ReLU()
        self.conv2 = GCNConv(16, 32)
        self.activation2 = nn.ReLU()
        self.conv3 = GCNConv(32, 16)
        self.activation3 = nn.ReLU()
        self.color_picker = Linear(16, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.activation1(h)
        h = self.conv2(h, edge_index)
        h = self.activation2(h)
        h = self.conv3(h, edge_index)
        h = self.activation3(h)  # Final GNN embedding space.
        
        out = self.color_picker(h)

        return out

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.activation1(h)
        h = self.conv2(h, edge_index)
        h = self.activation2(h)
        h = self.conv3(h, edge_index)
        h = self.activation3(h)  # Final GNN embedding space.
        
        out = self.color_picker(h)

        return out

class ColorGNNBigger(nn.Module):
    def __init__(self, feature_size):
        super(ColorGNNBigger, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(feature_size, 2048)
        self.activation1 = nn.ReLU()
        self.conv2 = GCNConv(2048, 1024)
        self.activation2 = nn.ReLU()
        self.conv3 = GCNConv(1024, 512)
        self.activation3 = nn.ReLU()
        self.conv4 = GCNConv(512, 128)
        self.activation4 = nn.ReLU()
        self.conv5 = GCNConv(128, 64)
        self.activation5 = nn.ReLU()
        self.color_picker = Linear(64, 3)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = self.activation1(h)
        h = self.conv2(h, edge_index)
        h = self.activation2(h)
        h = self.conv3(h, edge_index)
        h = self.activation3(h)  # Final GNN embedding space.
        h = self.conv4(h, edge_index)
        h = self.activation4(h)  # Final GNN embedding space.
        h = self.conv5(h, edge_index)
        h = self.activation5(h)  # Final GNN embedding space.

        
        out = self.color_picker(h)

        return out