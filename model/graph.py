
import torch
import torch.nn as nn
import numpy as np
from utils import *
from PIL import Image

class DesignGraph():
    def __init__(self, pretrained_model, idx, data_path):
        '''
            Node types: Image, Background, Text
            Node features: embedding, color palette, relative size, class
            Edge features: distance between elements
        '''
        self.pretrained_model = pretrained_model
        self.embedding_list = []
        self.layers = ['image', 'background', 'text'] # Take this from config file later

        path_idx = "{:04d}".format(idx)

        self.img_path_dict = {
            'preview': data_path + '/00_preview/' + path_idx + '.png',
            'background': data_path + '/01_background/' + path_idx + '.png',
            'image': data_path + '/02_image/' + path_idx + '.png',
            'text': data_path + '/04_text/' + path_idx + '.png',
        }

        self.annotation_path_dict = {
            'preview': data_path + '/xmls/' +'/00_preview/' + path_idx + '.xml',
            'background': data_path + '/xmls/' + '/01_background/' + path_idx + '.xml',
            'image': data_path + '/xmls/' + '/02_image/' + path_idx + '.xml',
            'text': data_path + '/xmls/' + '/04_text/' + path_idx + '.xml',
        }

        for i, layer in enumerate(self.layers):
            img = np.array(Image.open(self.img_path_dict[layer]).convert('RGB'))
            bboxes = VOC2bbox(self.annotation_path_dict[layer])

    def calculate_distances(self):
        pass

    def construct_embeddings(self):
        pass

    def calculate_relative_size(self):
        pass

