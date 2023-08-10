import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
import yaml
from utils import *
from skimage.color import rgb2lab, lab2rgb

with open("config/conf.yaml", 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

data_type = config["data_type"]
threshold = config["threshold_for_neighbours"]
mask_type = config["node_to_mask"]

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

class GraphDestijlDataset(Dataset):
    def __init__(self, root, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.sample_filenames = os.listdir(root + '/' + data_type +'/')
        self.processed_data_dir = root + '/' + data_type + '/'

        # If you want to use less data than the whole dataset, you can specify the range here.
        # Than it loads only samples up to that sample.
        # self.sample_filenames = ["data_{:04d}.pt".format(idx) for idx in range(0, 31)]

        self.mean_node = np.load("dataset_statistics/mean_node.npy").astype(np.float32)
        self.std_dev_node = np.load("dataset_statistics/std_dev_node.npy").astype(np.float32)
        self.mean_edge = np.load("dataset_statistics/mean_edge.npy").astype(np.float32)
        self.std_dev_edge = np.load("dataset_statistics/std_dev_edge.npy").astype(np.float32)

        # Train test filenames.
        self.train_filenames, self.test_filenames = train_test_split(self.sample_filenames, 
                                                            test_size=0.2, 
                                                            random_state=42)

        super(GraphDestijlDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        return "empty"

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""

        if self.test:
            return self.test_filenames
        else:
            return self.train_filenames

    def download(self):
        pass

    def process(self):
        pass

    def test_train_mask(self, data):

        '''
            Input: graph data

            Mask the color of one node. The ground truth color is the last 3 dimension of the feature vector.
            Data is saved as RGB. 
            If you want you can convert unnormalized RGB ground truth color to Lab.
            (Conversion is done using COLORMATH)
            Put mask on a random node's color information by setting that color to [0, 0, 0].

            Return: new_data with masked RGB colors, color_to_hide in lab, node_to_mask scalar
        '''

        # Take number of nodes
        n_nodes = len(data.x)
        if mask_type == -1:
            # Chose the color to mask randomly
            node_to_mask = random.randint(0, n_nodes-2)
        else:
            # Mask the second last node each time
            node_to_mask = n_nodes-2

        # If you chosed a folder that has processed_rgb name in it, then all the colors are stores in (0, 255) RGB.
        feature_vector = data.x
        # This is our target.
        color_to_hide = feature_vector[node_to_mask, -3:].clone()

        # Conversion to cielab. I do not use it anymore.
        #color_to_hide = torch.tensor(RGB2CIELab(color_to_hide.numpy().astype(np.int32)))
        #color_to_hide = torch.tensor(rgb2lab(color_to_hide.numpy().astype(np.int32)))

        # Set node to mask in feature vector to zero.
        feature_vector[node_to_mask, -3:] = torch.Tensor([0.0, 0.0, 0.0])
        # Assing the new feature vector to the graph.
        new_data = data.clone()
        new_data.x = feature_vector

        # This code below is used if we want to apply a threshold while adding edges.
        # It just removes the edges with a higher distance than the threshold.
        
        # new_edge_weight = []
        # new_edge_index = []
        # for k, edge in enumerate(new_data.edge_weight):
        #     if edge.item() < threshold:
        #         new_edge_weight.append(edge.item())
        #         new_edge_index.append([new_data.edge_index[0][k], new_data.edge_index[1][k]])

        # new_data.edge_index = torch.Tensor(new_edge_index).T
        # new_data.edge_weight = torch.Tensor(new_edge_weight)
        #print("New calculations")
        #print(new_data.edge_index, new_data.edge_weight)

        return new_data, color_to_hide, node_to_mask

    def len(self):
        if self.test:
            return len(self.test_filenames)
        else:
            return len(self.train_filenames)

    def get(self, idx):
        if self.test:
            data = torch.load(self.processed_data_dir+self.test_filenames[idx])
            new_data, target_color, node_to_mask = self.test_train_mask(data)
        else:
            data = torch.load(self.processed_data_dir+self.train_filenames[idx])
            new_data, target_color, node_to_mask = self.test_train_mask(data)
        return new_data, target_color, node_to_mask

if __name__ == '__main__':
    dataset_obj = GraphDestijlDataset(root='../shape_dataset/')
