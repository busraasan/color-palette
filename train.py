import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import color, io
from colormath.color_objects import sRGBColor, HSVColor, LabColor, LCHuvColor, XYZColor, LCHabColor, AdobeRGBColor

from model.GNN import *
from dataset import GraphDestijlDataset

from utils import *
import logging
import yaml
import argparse

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

'''
    Take small test, very small, Overfit, See model is corrext
    nn embedding to 1000 thousand or close to that
    Grey might have Minimum distance to evetutjing else, Penalize grey based on the saturation
'''

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="config/conf.yaml", help="Path to the config file.")
args = parser.parse_args()
config_file = args.config_file

with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

data_type = config["data_type"]
model_name = config["model_name"]
device = config["device"]

save_path = "../models/" + model_name + "/weights"
loss_path =  "../models/" + model_name + "/losses"
log_path = "../models/"+model_name

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(loss_path):
    os.makedirs(loss_path)

if not os.path.exists(log_path):
    os.makedirs(log_path)

logging.basicConfig(
    filename=log_path+"/log.log",
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M-%S', level=logging.DEBUG, filemode='w')

logger = logging.getLogger()
logging.getLogger('matplotlib.font_manager').disabled = True
seed = 42
logger.info("#" * 100)

num_epoch = config["num_epoch"]
batch_size = config["batch_size"]
feature_size = config["feature_size"]
lr =  config["lr"]
weight_decay = config["weight_decay"]

model = ColorGNNEmbedding(feature_size=feature_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

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
    l, a, b, loss = CIELab_distance2(out[node_to_mask, :][0], target_color[0])
    loss2 = colormath_CIE2000(out[node_to_mask, :][0], target_color[0])
    # loss = torch.tensor([loss], requires_grad=True)  # Derive gradients.
    # count=0
    # for param in model.parameters():
    #     print(count)
    #     count+=1
    #     print(param)
    loss2.backward()
    optimizer.step()  # Update parameters based on gradients.
    return  l, a, b, loss2, out, out[node_to_mask, :][0], target_color[0]

def test(data, target_color, node_to_mask):
    model.eval()
    out = model(data.x, data.edge_index.long(), data.edge_attr)  # Perform a single forward pass.
    l, a, b, loss = CIELab_distance2(out[node_to_mask, :][0], target_color[0])  # Compute the loss solely based on the training nodes.
    loss2 = colormath_CIE2000(out[node_to_mask, :][0], target_color[0])
    #loss = criterion(out[node_to_mask, :]*255, target_color*255)
    return l, a, b, loss2, out

train_losses = []
val_losses = []
for epoch in range(num_epoch):

    total_train_loss = 0
    total_val_loss = 0
    count = 0
    best_loss = 1000
    
    for input_data, target_color, node_to_mask in train_loader:
        l, a, b, loss, out, pred, target = train(input_data.to(device), target_color.to(device), node_to_mask)
        # if count == 0:
        #     print("IMAGE LOSS")
        #     print(loss.item())
        #     print("l: ", l.item(), "a: ", a.item(), "b: ", b.item())
        #     print("Pred and target")
        #     print(pred)
        #     print(target)
        # prediction = out[node_to_mask, :]
        # other_colors = input_data.y.clone()
        # other_colors = torch.cat([other_colors[0:node_to_mask, :], other_colors[node_to_mask+1:, :]])
        # # prediction in cielab
        # current_palette = torch.cat([other_colors, prediction, target_color]).type(torch.float32).detach().numpy()
        # sns.palplot(CIELab2RGB(current_palette))
        # path = "../"+model_name+"_palettes"
        # if not os.path.exists(path):
        #     os.mkdir(path)
        # plt.savefig(path+"/color_"+str(count)+".jpg")
        # plt.close()
        # count +=1

        total_train_loss += loss.item()

    for input_data, target_color, node_to_mask in test_loader:
        l, a, b, loss, out = test(input_data.to(device), target_color.to(device), node_to_mask)
        total_val_loss += loss.item()

    val_loss = total_val_loss/len(test_dataset)
    train_loss = total_train_loss/len(train_dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    logger.info("------- Epoch " + str(epoch) + " -------")
    logger.info("Train loss: " + str(train_loss)) 
    logger.info("Test loss: " + str(val_loss))

    print("------- Epoch ", epoch, " -------")
    print("Train loss: ", train_loss)
    print("Test loss: ", val_loss)

    save_plot(train_losses=train_losses, val_losses=val_losses, loss_type="CIE2000", loss_path=loss_path)
    if val_loss < best_loss:
        best_loss = val_loss
        model = model.cpu()
        best_model = model.state_dict()
        model = model.to(device)
        save_model(state_dict=best_model, train_losses=train_losses, val_losses=val_losses, epoch=epoch, save_path=save_path)

    lr_scheduler.step()
