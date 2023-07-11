import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np

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
        self.sample_filenames = os.listdir(root+'processed/')
        self.processed_data_dir = root + 'processed/'

        self.train_filenames, self.test_filenames = train_test_split(self.sample_filenames, 
                                                            test_size=0.30, 
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
            Mask the color of one node. The color is the last 3 dimension of the feature vector.
            Put mask on a random node's color information.
        '''
        n_nodes = len(data.x)
        node_to_mask = random.randint(0, n_nodes-1)
        feature_vector = torch.stack(data.x)
        color_to_hide = feature_vector[node_to_mask, -3:]
        feature_vector[node_to_mask, -3:] = torch.Tensor([0.0, 0.0, 0.0])
        new_data = data.clone()
        new_data.x = feature_vector
        return new_data, color_to_hide, node_to_mask

    def len(self):
        if self.test:
            return len(self.test_filenames)
        else:
            return len(self.train_filenames)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
        """
        if self.test:
            data = torch.load(self.processed_data_dir+self.test_filenames[idx])
        else:
            data = torch.load(self.processed_data_dir+self.train_filenames[idx])
            target_color, node_to_mask, new_data = self.test_train_mask(data)
        return new_data, target_color, node_to_mask

if __name__ == '__main__':
    dataset_obj = GraphDestijlDataset(root='../destijl_dataset/')
    dataset_obj.get(0)
