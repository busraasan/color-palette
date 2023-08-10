import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml
import argparse
from torchvision import transforms

import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from model.CNN import *
from cnn_dataset import *

import pandas as pd

######################## Set Parameters ########################

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = "cuda:1"

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="ColorCNN", help="Give the name of the model.")
args = parser.parse_args()
model_name = args.model_name

config_file = "../CNN_models/"+model_name+"/conf.yaml"

with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

## Basic Training Parameters ##
model_name = config["model_name"]
device = config["device"]

## Neural Network Parameters ##
loss_function = config["loss_function"]
out_features = config["out_features"]
color_space = config["color_space"]
input_color_space = config["input_color_space"]
is_classification = config["is_classification"]
input_size = config["input_size"]
normalize_rgb = config["normalize_rgb"]
normalize_cielab = config["normalize_cielab"]

model_weight_path = "../CNN_models/" + model_name + "/weights/best.pth"

if out_features == 1:
    out_type = "Lightness"
else:
    out_type = "Color"

print("Evaluating for the model: ", model_name, "\n",
      "Loss function: ", loss_function, "\n",
      "Output Color Space: ", color_space, "\n",
      "Color or Lightness?: ", out_type, "\n",
      "Device: ", device, "\n")
######################## Model ########################

transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    #transforms.Normalize((255,), (255,))
    ])

test_dataset = PreviewDataset(transform=transform, 
                               color_space=color_space, 
                               input_color_space=input_color_space,
                               normalize_rgb=normalize_rgb,
                               normalize_cielab=normalize_cielab,
                               test=True)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
num_of_plots = len(test_loader)

model = model_switch_CNN(model_name, out_features).to(device)
model.load_state_dict(torch.load(model_weight_path)["state_dict"])

if loss_function == "MSE":
    criterion = nn.MSELoss()
elif loss_function == "Cross-Entropy":
    criterion = nn.CrossEntropyLoss()
elif loss_function == "MAE":
    criterion = nn.L1Loss()

######################## Helper Functions ########################
def test(data, color):
    model.eval()
    out = model(data)

    if out_features == 1:
        if loss_function != "CIELab":
            loss = criterion(out[0][0], color[0][0])
        else:
            loss = colormath_CIE2000(out[0][0], color[0][0])
    else:
        if loss_function != "CIELab":
            loss = criterion(out, color)
        else:
            loss = colormath_CIE2000(out, color)
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
column_titles = ["Prediction                   Target" for i in range(cols)]
for ax, col in zip(ax_array[0], column_titles):
    ax.set_title(col, fontdict={'fontsize': 45, 'fontweight': 'medium'})

fig.suptitle(model_name+" Test Palettes", fontsize=100)

plot_count = 0
val_losses = []

outputs = []
target_colors = []

for i, (input_data, target_color) in enumerate(test_loader):
    loss, out = test(input_data.to(device), target_color.to(device))
    val_losses.append(loss.item())
    # Get predicton and other colors in the palette
    ax = plt.subplot(rows, cols, plot_count+1)

    if color_space == "CIELab":
        out = out.detach().cpu().numpy()
        out = np.append(out, [[30.0, 30.0]], axis=1)
        target_color = np.array([[target_color.detach().cpu().numpy()[0][0], 30.0, 30.0]])
        palette = np.clip(np.concatenate([CIELab2RGB(out), CIELab2RGB(target_color)]), a_min=0, a_max=1)
    else:
        palette = np.clip(np.concatenate([out.detach().cpu().numpy(), target_color/255]), a_min=0, a_max=1)
    outputs.append(out)
    target_colors.append(target_color)

    my_palplot(palette, ax=ax)

    plot_count+=1

    if i == num_of_plots-1:
        path = "../CNN_models/"+model_name
        if not os.path.exists(path):
            os.mkdir(path)
        plt.savefig(path+"/palettes.jpg")
        plt.close()

        cielab_dict = {'Output': outputs, 'Targets': target_colors}
        df = pd.DataFrame(data=cielab_dict)

#df.to_csv("trainset_predictions.csv")
