
import torch
import torch.nn as nn
import numpy as np
from utils import *
from PIL import Image

class DesignGraph():
    def __init__(self, pretrained_model, all_images, all_bboxes, layers):
        '''
            Node types: Image, Background, Text
            Node features: embedding, color palette, relative size, class
            Edge features: distance between elements
        '''
        self.pretrained_model = pretrained_model
        self.embedding_dict = {}            

        for layer in layers:
            self.embedding_dict[layer] = []
            for bbox in all_bboxes[layer]:
                x, y = int(bbox[0][0]), int(bbox[0][1])
                z, t = int(bbox[2][0]), int(bbox[2][1])
                cropped_image = all_images[layer][y:t, x:z]
                self.embedding_list.append(self.pretrained_model(cropped_image))

    def calculate_distances(self):
        pass

    def construct_embeddings(self):
        pass

    def calculate_relative_size(self):
        pass

