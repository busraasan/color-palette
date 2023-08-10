
import torch
import torch.nn as nn
from torch_geometric.data import Data
from skimage.color import rgb2lab, lab2rgb

import numpy as np
from utils import *
from PIL import Image
import math
import cv2
import yaml
import argparse
from config import *

from torchvision import transforms

"""
    All of the processed files are corrupted unfortunately.
"""
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256,256)),
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))
    ])

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="config/conf.yaml", help="Path to the config file.")
args = parser.parse_args()
config_file = args.config_file

config = DataConfig()
dataset_root = config.dataset

with open(config_file, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

data_type = config["data_type"]
model_name = config["model_name"]
threshold = config["threshold_for_neighbours"]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class DesignGraph():
    def __init__(self, pretrained_model, all_images, all_bboxes, layers, preview_path, idx):
        '''
            Node types: Image, Background, Text
            Node features: embedding, color palette, relative size, class
            Edge features: distance between elements
        '''
        self.pretrained_model = pretrained_model
        self.layers = layers   
        self.all_bboxes = all_bboxes
        self.all_images = all_images
        self.preview_path = preview_path
        self.idx = idx
        self.all_colors = []

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        self.flags = cv2.KMEANS_RANDOM_CENTERS

        self.num_nodes = 0
        self.num_nodes_per_class = {
            'image':0,
            'background':0,
            'text':0,
            "decoration":0
        }
        self.layer_classes = {
            'image':0,
            'background':1,
            'text':2,
            "decoration":3,
        }
        self.node_information = []
        self.preview_img = cv2.imread(preview_path)

        for layer in layers:
            for bbox in self.all_bboxes[layer]:
                bbox_smallest_x, bbox_smallest_y = np.min(bbox, axis=0)
                bbox_biggest_x, bbox_biggest_y = np.max(bbox, axis=0)
                width = bbox_biggest_x - bbox_smallest_x
                height = bbox_biggest_y - bbox_smallest_y
                if width < 2 or height < 2:
                    self.all_bboxes[layer].remove(bbox)
            
            if layer == "background":
                self.num_nodes += 1
                self.num_nodes_per_class[layer] = 1
            else:
                self.num_nodes += len(self.all_bboxes[layer])
                self.num_nodes_per_class[layer] = len(self.all_bboxes[layer])
        
        self.distance_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self.COO_matrix = []
        self.edge_features = []
        self.construct_features()
        self.construct_graph()

    def calculate_distances(self, bbox, num_node):
        '''
            Calculates and fills distance matrix
            TODO: If two bboxes overlap, make distance equal to zero
        '''
        self_xmin, self_ymin = np.min(bbox, axis=0)
        self_xmax, self_ymax = np.max(bbox, axis=0)
        self_middle_x = self_xmax - self_xmin
        self_middle_y = self_ymax - self_ymin
        for node in self.node_information:
            if num_node == node[0]:
                self.distance_matrix[num_node, num_node] = -1
            else:
                other_bbox = node[2]
                if calculate_overlap(other_bbox, bbox) > 0.9:
                    distance = 0
                else:
                    xmin, ymin = np.min(other_bbox, axis=0)
                    xmax, ymax = np.max(other_bbox, axis=0)
                    middle_x = xmax - xmin
                    middle_y = ymax - ymin

                    img_hipotenus = math.sqrt(math.pow(self.preview_img.shape[0], 2)+math.pow(self.preview_img.shape[1],2))
                    distance = math.sqrt(math.pow(self_middle_x-middle_x, 2)+math.pow(self_middle_y-middle_y,2))
                    distance = distance / img_hipotenus
                    distance = format(distance, '.3f')
           
                self.distance_matrix[num_node, node[0]] = distance

    def construct_features(self):
        num_node = 0
        for layer in self.layers:
            for bbox in self.all_bboxes[layer]:
                x, y = int(bbox[0][0]), int(bbox[0][1])
                z, t = int(bbox[2][0]), int(bbox[2][1])

                if data_type == "processed_rgb_cnn":
                    img = Image.open(self.all_images[layer]).convert("RGB")
                    temp = self.pretrained_model.encoder(transform(img))
                    embedding = torch.flatten(temp, start_dim=1)
                else:
                    img = cv2.imread(self.all_images[layer])
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    convert_to_tensor = torch.from_numpy(img[y:t, x:z]).permute(2,0,1).float()
                    processed_img = preprocess(convert_to_tensor)
                    # cropped_image = torch.unsqueeze((processed_img), 0)
                    # embedding = self.pretrained_model(transform(cropped_image))
                    embedding = self.pretrained_model(torch.unsqueeze(transform(processed_img), 0))

                if layer == "background":
                    relative_size = 1
                else:
                    relative_size = (t*z) / (self.preview_img.shape[0]*self.preview_img.shape[1])
                if relative_size > 1:
                    relative_size = 1
                self.node_information.append([num_node, layer, bbox, embedding, relative_size])
                #self.node_information.append([num_node, layer, bbox, relative_size])
                num_node+=1

        for node in self.node_information:
            self.calculate_distances(node[2], node[0])

        # Construct COO format
        rows = self.distance_matrix.shape[0]
        cols = self.distance_matrix.shape[1]
        for row in range(rows):
            for col in range(cols):
                if row != col:
                    self.COO_matrix.append([row,col])
                    self.edge_features.append(self.distance_matrix[row, col])

    def extract_text_color_from_design(self, preview_path, bbox):
        image = cv2.imread(preview_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        n_colors = 2
        
        # Crop the text area
        x, y = int(bbox[0][0]), int(bbox[0][1])
        z, t = int(bbox[2][0]), int(bbox[2][1])
        cropped_image = image[y:t, x:z]

        # Apply KMeans to the text area
        pixels = np.float32(cropped_image.reshape(-1, 3))
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, self.criteria, 10, self.flags)
        palette = np.asarray(palette, dtype=np.int64)
        palette_w_white = []

        for i, color in enumerate(palette):
            x, y, z = color
            palette_w_white.append(color)

        _, counts = np.unique(labels, return_counts=True)
        text_color = palette_w_white[np.argmin(counts)]
        background_color = palette_w_white[np.argmax(counts)]

        # ax1 = plt.subplot(1, 1, 1)
        # color_palette = np.asarray(text_color)/255
        # self.my_palplot(color_palette, ax=ax1)
        # plt.savefig("all_conversions_text.jpg")

        return text_color

    def extract_image_color_from_design(self, preview_path, bbox, layer):
        image = cv2.imread(preview_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if layer == "background":
            n_colors = 1
        else:
            n_colors = 3
        # Crop the text area
        x, y = int(bbox[0][0]), int(bbox[0][1])
        z, t = int(bbox[2][0]), int(bbox[2][1])
        cropped_image = image[y:t, x:z]
        cv2.imwrite("cropped.jpg", cropped_image)

        # Apply KMeans to the text area
        pixels = np.float32(cropped_image.reshape(-1, 3))
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, self.criteria, 10, self.flags)
        palette = np.asarray(palette, dtype=np.int64)
        palette_w_white = []
        if layer != "background":
            for i, color in enumerate(palette):
                x, y, z = color
                # Do not add white to the palette
                if not (252 < x < 256 and 252 < y < 256 and 252 < z < 256):
                    palette_w_white.append(color)
                else:
                    labels = np.delete(labels, np.where(labels == i))
                x, y, z = color

            _, counts = np.unique(labels, return_counts=True)
            background_color = palette_w_white[np.argmax(counts)] #dominant
            # ax1 = plt.subplot(1, 1, 1)
            # color_palette = np.asarray(background_color)/255
            # self.my_palplot(color_palette, ax=ax1)
            # plt.savefig("all_conversions.jpg")

            return background_color
        else:
            return palette

    def get_all_colors_in_design(self):
        return self.all_colors
    
    def my_palplot(self, pal, size=1, ax=None):
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

    def sanity_check(self, color_palette):
        '''
            Palette in RGB
        '''
        rows = 2
        cols = 2
        fig, ax_array = plt.subplots(rows, rows, figsize=(20, 20), dpi=80, squeeze=False)
        fig.suptitle("Sanity Check Palettes", fontsize=100)
        
        new_color_palette = []
        for color in color_palette:
            color = color[0]
            if len(color.shape) == 2:
                color = color[0]
            
            new_color_palette.append(color)

        ax1 = plt.subplot(rows, cols, 1)
        color_palette = np.asarray(new_color_palette)/255
        self.my_palplot(color_palette, ax=ax1)

        ax2 = plt.subplot(rows, cols, 2)
        lab_palette = [rgb2lab(color) for color in color_palette]
        #self.my_palplot(lab_palette, ax=ax2)

        ax3 = plt.subplot(rows, cols, 3)
        rgb_palette = [lab2rgb(color) for color in lab_palette]
        self.my_palplot(rgb_palette, ax=ax3)

        ax4 =  plt.subplot(rows, cols, 4)
        rgb_palette_colormath = CIELab2RGB(lab_palette)
        self.my_palplot(rgb_palette_colormath, ax=ax4)

        plt.savefig("all_conversions.jpg")

    def construct_graph(self):
        '''
            For now, only one color from the image is used
        '''

        node_features = []
        y = []
        for i, node in enumerate(self.node_information):
            num_node, layer, bbox, embedding, relative_size = node
            #num_node, layer, bbox, relative_size = node
            if layer == 'image':
                # print("Image")
                color_palette = [self.extract_image_color_from_design(self.preview_path, bbox, layer)]
                #print(color_palette)
            elif layer == 'text':
                # print("Text")
                color_palette = [self.extract_text_color_from_design(self.preview_path, bbox)]
                # print(color_palette)
            elif layer == 'background':
                # print("Background")
                color_palette = [self.extract_image_color_from_design(self.all_images[layer], bbox, layer)]
                # print(color_palette)
            elif layer == "decoration":
                # print("Decoration")
                color_palette = [self.extract_image_color_from_design(self.preview_path, bbox, layer)]
                # print(color_palette)
            
            self.all_colors.append(color_palette)

            if "embedding" in model_name.lower():
                new_color_palette = []
                for color in color_palette:
                    if len(color.shape) == 2:
                        color = color[0]
                    x, f, z = color
                    if x == 0:
                        x += 1
                    if f == 0:
                        f += 1
                    if z == 0:
                        z += 1
                    color = [x, f, z]
                    new_color_palette.append(color)
                colors = np.asarray(new_color_palette).flatten()
            else:
                new_color_palette = []
                for color in color_palette:
                    color = color/255
                    lab_color = rgb2lab(color)
                    new_color_palette.append(lab_color)
                colors = np.asarray(new_color_palette).flatten()

            if "w_embedding" in data_type.lower():
                feature_vector = np.concatenate((np.asarray([self.layer_classes[layer]]), np.asarray([relative_size]), colors))
            else:
                feature_vector = np.concatenate((np.asarray([self.layer_classes[layer]]), embedding.detach().numpy().flatten(), [relative_size], colors))
            node_features.append(feature_vector)
            y.append(colors)
        #self.sanity_check(self.all_colors)
        path_idx = "{:04d}".format(self.idx)
        data = Data(x=torch.from_numpy(np.asarray(node_features)).type(torch.float), 
                    edge_index=torch.Tensor(self.COO_matrix).permute(1,0).type(torch.int32),
                    edge_weight=torch.Tensor(self.edge_features),
                    y=torch.from_numpy(np.asarray(y)),
                    )
        
        if not os.path.exists(dataset_root+data_type):
            os.mkdir(dataset_root+data_type)
        torch.save(data, os.path.join(dataset_root+data_type, f'data_{path_idx}.pt'))
