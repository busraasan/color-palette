import torch
import torch.nn as nn
from torch import autograd
from torch_geometric.loader import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
from skimage import color, io
from colormath.color_objects import sRGBColor, HSVColor, LabColor, LCHuvColor, XYZColor, LCHabColor, AdobeRGBColor

from dataset import GraphDestijlDataset

from utils import *
import logging
import yaml
import argparse

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

'''
    Grey might have Minimum distance to everything else, Penalize grey based on the saturation
    Dataset size is small -- DONE
    smaller network --- DONE
    permute colors for more data -- DONE
    training CNN to map resnet embeddings to a smaller size (joint or separate)

    train resnet50 last layer maybe

    ######### Some DL things to try #########
    normalization & standardization (only edge, otherwise does not work with embeddings)
    adding batchnorm -- does not work
    adding dropout -- does not work

    * Try having more than one colors in the image. current colors are not representative and
    not enough for the images.

    * Try the network without image.

    In some experiments I was using cielab loss on rgb. fixed it. retrain experiments.
    Why converted colors have very big numbers at the output. I think that is hard to learn.
    Check about the correct network to see other cielab values.

    remove the elements with greys/blacks data completely

    heterogenous graph with design, decoration, image and color nodes.
    Or simpler: design and color nodes?
    Forget about the distance. A color is connected to text, or image.
    Text and image is connected to design
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
dataset_root = config["dataset_root"]
loss_function = config["loss_function"]

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

#model = ColorGNN(feature_size=feature_size).to(device)
model = model_switch(model_name, feature_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = nn.MSELoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

train_dataset = GraphDestijlDataset(root=dataset_root)
test_dataset = GraphDestijlDataset(root=dataset_root, test=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def train(data, target_color, node_to_mask):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index.long(), data.edge_attr)
    loss2 = 0
    mini_batch_size = node_to_mask.shape[0]

    
    
    for i in range(mini_batch_size):
        if loss_function == "MSE":
            loss2 += criterion(out[node_to_mask, :][i], target_color[i])
        elif loss_function == "CIELab2000":
            loss2 += colormath_CIE2000(out[node_to_mask, :][i], target_color[i])
        elif loss_function == "CIELabCMC":
            loss2 += colormath_CIECMC(out[node_to_mask, :][i], target_color[i])
    loss2 /= mini_batch_size
    loss2.backward()
    optimizer.step()
    return  loss2, out

def test(data, target_color, node_to_mask):
    model.eval()
    out = model(data.x, data.edge_index.long(), data.edge_attr)
    if loss_function == "MSE":
        loss2 = criterion(out[node_to_mask, :][0], target_color[0])
    elif loss_function == "CIELab2000":
        loss2 = colormath_CIE2000(out[node_to_mask, :][0], target_color[0])
    elif loss_function == "CIELabCMC":
        loss2 = colormath_CIECMC(out[node_to_mask, :][0], target_color[0])
    return loss2, out

train_losses = []
val_losses = []

best_loss = 100000
for epoch in range(num_epoch):

    total_train_loss = 0
    total_val_loss = 0
    count = 0

    num_batches = len(train_loader)
    
    for input_data, target_color, node_to_mask in train_loader:
        loss, out = train(input_data.to(device), target_color.to(device), node_to_mask)
        total_train_loss += loss.item()

    for input_data, target_color, node_to_mask in test_loader:
        loss, out = test(input_data.to(device), target_color.to(device), node_to_mask)
        total_val_loss += loss.item()

    val_loss = total_val_loss/len(test_dataset)
    train_loss = total_train_loss/num_batches
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
