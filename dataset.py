import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
from sklearn.model_selection import train_test_split
import os

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
        self.processed_dir = root + 'processed/'

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

    def len(self):
        if self.test:
            return len(self.test_filenames)
        else:
            return len(self.train_filenames)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
        """
        if self.test:
            data = torch.load(self.processed_dir+self.test_filenames[idx])
        else:
            data = torch.load(self.processed_dir+self.train_filenames[idx])
        return data

if __name__ == '__main__':
    dataset_obj = GraphDestijlDataset(root='../destijl_dataset/')
