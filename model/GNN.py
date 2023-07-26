import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, BatchNorm
from utils import *
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

class ColorGNNSmall(nn.Module):
    def __init__(self, feature_size):
        super(ColorGNNSmall, self).__init__()
        torch.manual_seed(42)
        # gcn(1700, 128)
        self.conv1 = GCNConv(feature_size, 128)
        self.activation1 = nn.LeakyReLU()
        self.conv2 = GCNConv(128, 32)
        self.activation2 = nn.LeakyReLU()
        self.conv3 = GCNConv(64, 32)
        self.activation3 = nn.ReLU()
        self.color_picker = Linear(32, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = self.activation1(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = self.activation2(h)
        # h = self.conv3(h, edge_index, edge_attr)
        # h = self.activation3(h)  # Final GNN embedding space.
        
        out = self.color_picker(h)

        return out
    
class ColorGNNSmallEmbedding(nn.Module):
    def __init__(self, feature_size):
        super(ColorGNNSmallEmbedding, self).__init__()
        torch.manual_seed(42)
        # gcn(1700, 128)
        self.conv1 = GCNConv(feature_size-5+(250*2)+(85*3), 128)
        self.activation1 = nn.ReLU()
        self.conv2 = GCNConv(128, 32)
        self.activation2 = nn.ReLU()
        self.conv3 = GCNConv(64, 32)
        self.activation3 = nn.ReLU()
        self.color_picker = Linear(32, 3)

        self.layer_embedding = nn.Embedding(3, 250)
        self.color_embedding = nn.Embedding(256, 85)
        self.relative_size_embedding = nn.Embedding(11, 250) # 1 0.4 0.3


    def forward(self, x, edge_index, edge_attr):
        layer_embed = self.layer_embedding(x[:, 0].long())
        resnet_embedding = x[:, 1:1001]
        relative_size_embedding = self.relative_size_embedding(torch.round(x[:, 1001]*10).long())
        color_embedding =  self.color_embedding(x[:, -3:].long()).reshape(x.shape[0], -1)
        x = torch.hstack((layer_embed, resnet_embedding, relative_size_embedding, color_embedding))
        h = self.conv1(x, edge_index, edge_attr)
        h = self.activation1(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = self.activation2(h)
        # h = self.conv3(h, edge_index, edge_attr)
        # h = self.activation3(h)  # Final GNN embedding space.
        
        out = self.color_picker(h)

        return out

class ColorGNN(nn.Module):
    def __init__(self, feature_size):
        super(ColorGNN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(feature_size, 512)
        self.activation1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=0.4)
        self.conv2 = GCNConv(512, 256)
        self.activation2 = nn.LeakyReLU()
        self.conv3 = GCNConv(256, 64)
        self.activation3 = nn.LeakyReLU()
        self.dropout2= nn.Dropout(p=0.4)
        self.color_picker = Linear(64, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = self.activation1(h)
        h = self.dropout1(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = self.activation2(h)
        h = self.conv3(h, edge_index, edge_attr)
        h = self.activation3(h)  # Final GNN embedding space.
        h = self.dropout2(h)
        out = self.color_picker(h)

        return out

class ColorGNNEmbedding(nn.Module):
    def __init__(self, feature_size):
        super(ColorGNNEmbedding, self).__init__()
        torch.manual_seed(42)
        # 1750
        self.conv1 = GCNConv(feature_size-5+(250*2)+(85*3), 512)
        self.batchnorm1 = BatchNorm(512)
        self.activation1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=0.4)
        self.conv2 = GCNConv(512, 256)
        self.batchnorm2 = BatchNorm(256)
        self.activation2 = nn.LeakyReLU()
        self.conv3 = GCNConv(256, 64)
        self.batchnorm3 = BatchNorm(64)
        self.activation3 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(p=0.4)
        self.color_picker = Linear(64, 3)

        self.layer_embedding = nn.Embedding(3, 250)
        self.color_embedding = nn.Embedding(256, 85)
        self.relative_size_embedding = nn.Embedding(11, 250)

    def forward(self, x, edge_index, edge_attr):
        x[:, 0] = x[:, 0]
        layer_embed = self.layer_embedding(x[:, 0].long())
        resnet_embedding = x[:, 1:1001]
        relative_size_embedding = self.relative_size_embedding(torch.round(x[:, 1001]*10).long())
        color_embedding =  self.color_embedding(x[:, -3:].long()).reshape(x.shape[0], -1)
        x = torch.hstack((layer_embed, resnet_embedding, relative_size_embedding, color_embedding))
        h = self.conv1(x, edge_index, edge_attr)
        h = self.batchnorm1(h)
        h = self.activation1(h)
        h = self.dropout1(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = self.batchnorm2(h)
        h = self.activation2(h)
        h = self.dropout2(h)
        h = self.conv3(h, edge_index, edge_attr)
        h = self.batchnorm3(h)
        h = self.activation3(h)
        out = self.color_picker(h)
        return out
    
class ColorGNNBiggerEmbedding(nn.Module):
    def __init__(self, feature_size):
        super(ColorGNNEmbedding, self).__init__()
        torch.manual_seed(42)
        # 1750
        self.conv1 = GCNConv(feature_size-5+(250*2)+(85*3), 2048)
        self.batchnorm1 = BatchNorm(2048)
        self.activation1 = nn.LeakyReLU()
        self.conv2 = GCNConv(2048, 1024)
        self.batchnorm2 = BatchNorm(1024)
        self.activation2 = nn.LeakyReLU()
        self.conv3 = GCNConv(1024, 256)
        self.batchnorm3 = BatchNorm(256)
        self.activation3 = nn.LeakyReLU()
        self.conv4 = GCNConv(256, 64)
        self.batchnorm4 = BatchNorm(64)
        self.activation4 = nn.LeakyReLU()
        self.color_picker = Linear(64, 3)

        self.layer_embedding = nn.Embedding(2, 250)
        self.color_embedding = nn.Embedding(256, 85)
        self.relative_size_embedding = nn.Embedding(11, 250)

    def forward(self, x, edge_index, edge_attr):
        layer_embed = self.layer_embedding(x[:, 0].long())
        # if torch.any(torch.isnan(layer_embed)):
        #     print("nan embed")
        resnet_embedding = x[:, 1:1001]
        relative_size_embedding = self.relative_size_embedding(torch.round(x[:, 1001]*10).long())
        # if torch.any(torch.isnan(relative_size_embedding)):
        #     print("nan relative size embed")
        color_embedding =  self.color_embedding(x[:, -3:].long()).reshape(x.shape[0], -1)
        # if torch.any(torch.isnan(color_embedding)):
        #     print("nan color embed")
        #     exit()
        x = torch.hstack((layer_embed, resnet_embedding, relative_size_embedding, color_embedding))
        h = self.conv1(x, edge_index, edge_attr)
        h = self.batchnorm1(h)
        h = self.activation1(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = self.batchnorm2(h)
        h = self.activation2(h)
        h = self.conv3(h, edge_index, edge_attr)
        h = self.batchnorm3(h)
        h = self.activation3(h)
        h = self.conv4(h, edge_index, edge_attr)
        h = self.batchnorm4(h)
        h = self.activation4(h)

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

    def forward(self, x, edge_index, edge_attr):
        h = self.conv1(x, edge_index, edge_attr)
        h = self.activation1(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = self.activation2(h)
        h = self.conv3(h, edge_index, edge_attr)
        h = self.activation3(h)  # Final GNN embedding space.
        h = self.conv4(h, edge_index, edge_attr)
        h = self.activation4(h)  # Final GNN embedding space.
        h = self.conv5(h, edge_index, edge_attr)
        h = self.activation5(h)  # Final GNN embedding space.

        
        out = self.color_picker(h)

        return out