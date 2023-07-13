import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt

from model.GNN import *
from dataset import GraphDestijlDataset

from utils import *
import logging
'''
    Take small test, very small, Overfit, See model is corrext
    nn embedding to 1000 thousand or close to that
    Grey might have Minimum distance to evetutjing else, Penalize grey based on the saturation
    changing the loss function, Assumption that distance is perceptually uniform, We can change it to examples in wikipedia
'''

num_epoch = 1000
batch_size = 16
model = ColorGNN(feature_size=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

train_dataset = GraphDestijlDataset(root='../destijl_dataset/')
test_dataset = GraphDestijlDataset(root='../destijl_dataset/', test=True)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def train(data, target_color, node_to_mask):
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index.long(), data.edge_attr) 
     # Perform a single forward pass.
    #loss = criterion(out[node_to_mask, :]*255, target_color*255)
    loss = CIELab_distance2(out[node_to_mask, :][0], target_color[0])  # Compute the loss solely based on the training nodes.
    # loss = torch.tensor([loss], requires_grad=True)  # Derive gradients.
    # count=0
    # for param in model.parameters():
    #     print(count)
    #     count+=1
    #     print(param)
    loss.backward()
    optimizer.step()  # Update parameters based on gradients.
    return loss, out

def test(data, target_color, node_to_mask):
    model.eval()
    out = model(data.x, data.edge_index.long(), data.edge_attr)  # Perform a single forward pass.
    loss = CIELab_distance2(out.detach().numpy()[node_to_mask, :], target_color[0])  # Compute the loss solely based on the training nodes.
    #loss = criterion(out[node_to_mask, :]*255, target_color*255)
    return loss, out

for epoch in range(num_epoch):

    total_train_loss = 0
    total_val_loss = 0
    count=0
    best_loss = 1000
    for input_data, target_color, node_to_mask in train_loader:
        loss, out = train(input_data, target_color, node_to_mask)
        total_train_loss += loss.item()

    for input_data, target_color, node_to_mask in test_loader:
        loss, out = test(input_data, target_color, node_to_mask)
        prediction = out[node_to_mask, :]*255
        other_colors = input_data.y.clone()
        other_colors = torch.cat([other_colors[0:node_to_mask, :], other_colors[node_to_mask+1:, :]])
        current_palette = sns.color_palette(np.clip(torch.cat([other_colors, prediction, target_color*255]).type(torch.int32).numpy()/255, 0, 1))
        sns.palplot(current_palette)
        plt.savefig("palettes-n/color_"+str(count)+".jpg")
        plt.close()
        count +=1

        total_val_loss += loss.item()

    print("------- Epoch ", epoch, " -------")
    print("Train loss: ", total_train_loss/len(train_dataset))
    print("Test loss: ", total_val_loss/len(test_dataset))

    lr_scheduler.step()
