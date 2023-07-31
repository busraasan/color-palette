import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
import yaml
import argparse
from torchvision import transforms

import seaborn as sns
import matplotlib.pyplot as plt
from utils import *
from model.CNN import *

import pandas as pd

######################## Set Parameters ########################

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

model_name = "CNN_FinetuneResNet18_map_output"

device = "cuda:1"

model_weight_path = "../CNN_models/" + model_name + "/weights/best.pth"

######################## Model ########################

transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
    ])

class PreviewDataset(Dataset):
    def __init__(self, root="../destijl_dataset/rgba_dataset/", transform=None, test=False):

        self.test = test
        self.sample_filenames = os.listdir(root+"00_preview_cropped")
        self.transform = transform
        self.img_dir = root

        self.train_filenames, self.test_filenames = train_test_split(self.sample_filenames,
                                                                     test_size=0.2, 
                                                                     random_state=42) 
    def __len__(self):
        if self.test:
            return len(self.test_filenames)
        else:
            return len(self.train_filenames)
        
    def __getitem__(self, idx):

        path_idx = "{:04d}".format(idx)
        img_path = os.path.join(self.img_dir, "00_preview_cropped/" + self.sample_filenames[idx])
        image = Image.open(img_path).convert("RGB")

        bg_path = os.path.join("../destijl_dataset/01_background/" + self.sample_filenames[idx])
        color = self.kmeans_for_bg(bg_path)[0]
        color_cielab = torch.tensor(RGB2CIELab(color.astype(np.int32)))

        if self.transform:
            image = self.transform(image)
        return image, torch.squeeze(color_cielab)
    
    def kmeans_for_bg(self, bg_path):
        image = cv2.imread(bg_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        n_colors = 1

        # Apply KMeans to the text area
        pixels = np.float32(image.reshape(-1, 3))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        palette = np.asarray(palette, dtype=np.int64) # RGB

        return palette

test_dataset = PreviewDataset(transform=transform, test=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
num_of_plots = len(test_loader)

model = FinetuneResNet18(map_outputs=True).to(device)
model.load_state_dict(torch.load(model_weight_path)["state_dict"])


######################## Helper Functions ########################
def test(data, color):
    model.eval()
    out = model(data)
    loss = colormath_CIE2000(out[0], color[0])
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

outputs = []
target_colors = []

for i, (input_data, target_color) in enumerate(test_loader):
    loss, out = test(input_data.to(device), target_color.to(device))
    print("CIELAB")
    print(out)
    val_losses.append(loss.item())
    print("RGB")
    print(CIELab2RGB(out))
    # Get predicton and other colors in the palette
    ax = plt.subplot(rows, cols, plot_count+1)
    palette = np.clip(np.concatenate([CIELab2RGB(out), CIELab2RGB(target_color)]), a_min=0, a_max=1)
    outputs.append(out.detach().cpu().numpy())
    target_colors.append(target_color.detach().cpu().numpy())

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
