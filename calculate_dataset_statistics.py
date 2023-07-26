import torch
import numpy as np

import os
import yaml

with open("config/conf.yaml", 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

data_type = config["data_type"]
root = config["dataset_root"]
path =  root + '/' + data_type + '/'

sample_filenames = os.listdir(path)

all_node_mean = []
all_node_std = []
all_edge_mean = []
all_edge_std = []

for file in sample_filenames:
    data = torch.load(path + file)

    node_mean = torch.mean(data.x, axis=0).tolist()
    node_std = torch.std(data.x, axis=0).tolist()

    edge_mean = torch.mean(data.edge_attr, axis=0).tolist()
    edge_std = torch.std(data.edge_attr, axis=0).tolist()

    all_node_mean.append(node_mean)
    all_node_std.append(node_std)
    all_edge_mean.append(edge_mean)
    all_edge_std.append(edge_std)

mean_node = np.mean(all_node_mean, axis=0)
std_dev_node = np.std(all_node_std, axis=0)

if data_type == "processed_rgb":
    mean_node[0] = 0
    std_dev_node[0] = 1
    mean_node[-3:] = [0, 0, 0]
    std_dev_node[-3:] = [1, 1, 1]

mean_edge = np.mean(all_edge_mean, axis=0)
std_dev_edge = np.std(all_edge_std, axis=0)

np.save("dataset_statistics/mean_node.npy", mean_node)
np.save("dataset_statistics/std_dev_node.npy", std_dev_node)
np.save("dataset_statistics/mean_edge.npy", mean_edge)
np.save("dataset_statistics/std_dev_edge.npy", std_dev_edge)
