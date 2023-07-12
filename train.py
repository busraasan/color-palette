import torch
import torch.nn as nn
from torch_geometric.data import DataLoader


from model.GNN import ColorGNN
from dataset import GraphDestijlDataset
from utils import *

num_epoch = 100
batch_size = 16
model = ColorGNN(feature_size=1005)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

train_dataset = GraphDestijlDataset(root='../destijl_dataset/')
test_dataset = GraphDestijlDataset(root='../destijl_dataset/', test=True)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def train(data, target_color, node_to_mask):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = CIELab_distance(out.detach().numpy(), target_color)  # Compute the loss solely based on the training nodes.
    torch.tensor([loss], requires_grad=True).backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, out

def test(data, target_color, node_to_mask):
    model.eval()
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = CIELab_distance(out.detach().numpy(), target_color)  # Compute the loss solely based on the training nodes.
    return loss, out

for epoch in range(num_epoch):

    total_train_loss = 0
    total_val_loss = 0

    for input_data, target_color, node_to_mask in train_loader:
        loss, out = train(input_data, target_color, node_to_mask)
        total_train_loss += loss.item()

    for input_data, target_color, node_to_mask in test_loader:
        loss, out = test(input_data, target_color, node_to_mask)
        total_val_loss += loss.item()

    print("------- Epoch ", epoch, " -------")
    print("Train loss: ", total_train_loss/len(train_dataset))
    print("Test loss: ", total_val_loss/len(test_dataset))
