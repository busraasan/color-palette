
import torch
import torch.nn as nn
import numpy as np
from utils import *
from PIL import Image
import math

class DesignGraph():
    def __init__(self, pretrained_model, all_images, all_bboxes, layers):
        '''
            Node types: Image, Background, Text
            Node features: embedding, color palette, relative size, class
            Edge features: distance between elements
        '''
        self.pretrained_model = pretrained_model
        self.embedding_dict = {}  
        self.distance_dict = {}   
        self.layers = layers   
        self.all_bboxes = all_bboxes
        self.all_images = all_images    

        for layer in layers:
            self.embedding_dict[layer] = []
            self.distance_dict[layer] = []
            for bbox in all_bboxes[layer]:
                x, y = int(bbox[0][0]), int(bbox[0][1])
                z, t = int(bbox[2][0]), int(bbox[2][1])
                cropped_image = all_images[layer][y:t, x:z]
                self.embedding_list.append(self.pretrained_model(cropped_image))

    def calculate_distances(self, bbox):
        '''
            TODO: If two bboxes overlap, make distance equal to zero
        '''
        self_xmin, self_ymin = np.min(bbox, axis=0)
        self_xmax, self_ymax = np.max(bbox, axis=0)
        self_middle_x = self_xmax - self_xmin
        self_middle_y = self_ymax - self_ymin

        for layer in self.layers:
            for other_box in self.all_bboxes[layer]:
                if not np.array_equal(bbox, other_box):
                    # [[smallest_x, smallest_y], [biggest_x, smallest_y], [biggest_x, biggest_y], [smallest_x, biggest_y]]
                    xmin, ymin = np.min(other_box, axis=0)
                    xmax, ymax = np.max(other_box, axis=0)
                    middle_x = xmax-xmin
                    middle_y = ymax-ymin

                    distance = math.sqrt(math.square(middle_x-self_middle_x)+np.square(middle_y-self_middle_y))
                    self.distance[layer].append(distance)
                else:
                    self.distance[layer].append(-1)

    def construct_embeddings(self):
        pass

    def calculate_relative_size(self):
        pass

