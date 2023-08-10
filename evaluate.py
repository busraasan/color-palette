import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from dataset import GraphDestijlDataset
import yaml
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from config import *

######################## Set Parameters ########################

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# you can specify the config file you want to provide
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
loss_function = config["loss_function"]

model_weight_path = "../models/" + model_name + "/weights/best.pth"

dataset_root = config.dataset
######################## Model ########################

# Prepare dataset
# Set test=True for testing on the test set. Otherwise it tests on the train set.
test_dataset = GraphDestijlDataset(root=dataset_root, test=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
num_of_plots = len(test_loader)

# Model is selected according to name you provide
model = model_switch(model_name, feature_size).to(device)
model.load_state_dict(torch.load(model_weight_path)["state_dict"])
criterion = nn.MSELoss()
######################## Helper Functions ########################

# Skip black flag was used to eliminate examples that includes black colors.
# Always set to False for default experiments

skip_black_flag=False
def test(data, target_color, node_to_mask):

    model.eval()
    out = model(data.x, data.edge_index.long(), data.edge_attr)

    if skip_black_flag:
        for color in data.y:
            a, b, c = color
            if 0 <= a <= 30 and 0 <= b <= 30 and 0 <= c <= 30:
                return None, None
            else:
                loss = colormath_CIE2000(out[node_to_mask, :][0], target_color[0])
    else:
        # Loss is only in MSE now.
        loss = criterion(out[node_to_mask, :][0], target_color[0]/255)
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

# Config for plot
rows = num_of_plots//3 + 1
cols = 3

fig, ax_array = plt.subplots(rows, cols, figsize=(60, 60), dpi=80, squeeze=False)

column_titles = ["                                                                     Prediction | Target" for i in range(cols)]
for ax, col in zip(ax_array[0], column_titles):
    ax.set_title(col, fontdict={'fontsize': 30, 'fontweight': 'medium'})

fig.suptitle(model_name+" Test Palettes", fontsize=100)

# Code for evaluation loop
plot_count = 0
val_losses = []
palettes = []
for i, (input_data, target_color, node_to_mask) in enumerate(test_loader):
    loss, out = test(input_data.to(device), target_color.to(device), node_to_mask)
    if loss != None:
        val_losses.append(loss.item())

        # Get predicton and other colors in the palette
        ax = plt.subplot(rows, cols, plot_count+1)

        # Get prediction for a masked node
        prediction = out[node_to_mask, :]
        
        # Concat unmasked colors with prediction and ground truth
        other_colors = input_data.y.clone()
        other_colors = torch.cat([other_colors[0:node_to_mask, :], other_colors[node_to_mask+1:, :]])
        other_colors = other_colors.type(torch.float32).detach().cpu().numpy()
        # Normalize since they are in (0, 255) range.
        other_colors /= 255

        if loss_function == "CIELab":
            palette = np.clip(np.concatenate([other_colors, CIELab2RGB(prediction), CIELab2RGB(target_color[0])]), a_min=0, a_max=1)
        else:
            # Concat palettes. All of them are between (0, 1)
            palette = np.clip(np.concatenate([other_colors, prediction.detach().cpu().numpy(), target_color.detach().cpu().numpy()/255]), a_min=0, a_max=1)

        # I commented out codes related to calculating results in CIELab

        # if "embedding" in model_name.lower():
        #     other_colors = other_colors.type(torch.float32).detach().cpu().numpy()
        #     other_colors /= 255
        #     palette = np.clip(np.concatenate([other_colors, CIELab2RGB(prediction), CIELab2RGB(target_color[0])]), a_min=0, a_max=1)
        # else:
        #     current_palette = torch.cat([other_colors, prediction, target_color.to(device)]).type(torch.float32).detach().cpu().numpy()
        #     palette = CIELab2RGB(current_palette)

        # Save all the palettes to use it for distribution histograms.
        palettes.append(prediction.detach().tolist()[0])
        my_palplot(palette, ax=ax)
    else:
        print("none")

    plot_count+=1

    if i == num_of_plots-1:
        path = "../models/"+model_name
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(path+"palettes.jpg")
        plt.close()

# This is for checking prediction distribution
# It is saved as a histogram.
check_distributions(palettes)