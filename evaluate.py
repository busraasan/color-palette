import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from dataset import GraphDestijlDataset
import yaml
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
from utils import *

######################## Set Parameters ########################

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="config/conf.yaml", help="Path to the config file.")
args = parser.parse_args()
config_file = args.config_file

with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

data_type = config["data_type"]
model_name = config["model_name"]
device = config["device"]
feature_size = config["feature_size"]

model_weight_path = "../models/" + model_name + "/weights/best.pth"

######################## Model ########################

test_dataset = GraphDestijlDataset(root='../destijl_dataset/', test=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
num_of_plots = len(test_loader)

model = model_switch(model_name, feature_size).to(device)
model.load_state_dict(torch.load(model_weight_path)["state_dict"])

######################## Helper Functions ########################

def test(data, target_color, node_to_mask):
    model.eval()
    out = model(data.x, data.edge_index.long(), data.edge_attr)
    loss = colormath_CIE2000(out[node_to_mask, :][0], target_color[0])
    return loss, out

def my_palplot(pal, size=1, ax=None):
    """Plot the values in a color palette as a horizontal array.
    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by seaborn.color_palette()
    size :
        scaling factor for size of plot
    ax :
        an existing axes to use
    """

    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    n = len(pal)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(np.arange(n).reshape(1, n),
              cmap=mpl.colors.ListedColormap(list(pal)),
              interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(n) - .5)
    ax.set_yticks([-.5, .5])
    # Ensure nice border between colors
    ax.set_xticklabels(["" for _ in range(n)])
    # The proper way to set no ticks
    ax.yaxis.set_major_locator(ticker.NullLocator())

rows = num_of_plots//3 + 1
cols = 3

fig, ax_array = plt.subplots(rows, cols, figsize=(60, 60), dpi=80, squeeze=False)
fig.suptitle(model_name+" Test Palettes", fontsize=100)

plot_count = 0
val_losses = []
for i, (input_data, target_color, node_to_mask) in enumerate(test_loader):
    loss, out = test(input_data.to(device), target_color.to(device), node_to_mask)
    val_losses.append(loss.item())

    # Get predicton and other colors in the palette
    ax = plt.subplot(rows, cols, plot_count+1)

    prediction = out[node_to_mask, :]
    
    other_colors = input_data.y.clone()
    other_colors = torch.cat([other_colors[0:node_to_mask, :], other_colors[node_to_mask+1:, :]])
    
    other_colors = other_colors.type(torch.float32).detach().cpu().numpy()
    other_colors /= 255
    palette = np.clip(np.concatenate([other_colors, CIELab2RGB(prediction), CIELab2RGB(target_color[0])]), a_min=0, a_max=1)

    # if "embedding" in model_name.lower():
    #     other_colors = other_colors.type(torch.float32).detach().cpu().numpy()
    #     other_colors /= 255
    #     palette = np.clip(np.concatenate([other_colors, CIELab2RGB(prediction), CIELab2RGB(target_color[0])]), a_min=0, a_max=1)
    # else:
    #     current_palette = torch.cat([other_colors, prediction, target_color.to(device)]).type(torch.float32).detach().cpu().numpy()
    #     palette = CIELab2RGB(current_palette)

    my_palplot(palette, ax=ax)

    plot_count+=1

    if i == num_of_plots-1:
        path = "../models/"+model_name
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(path+"/palettes.jpg")
        plt.close()