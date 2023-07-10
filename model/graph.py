
import torch
import torch.nn as nn
import numpy as np
from utils import *
from PIL import Image
import math
import cv2

class DesignGraph():
    def __init__(self, pretrained_model, all_images, all_bboxes, layers, preview_path):
        '''
            Node types: Image, Background, Text
            Node features: embedding, color palette, relative size, class
            Edge features: distance between elements
        '''
        self.pretrained_model = pretrained_model
        self.layers = layers   
        self.all_bboxes = all_bboxes
        self.all_images = all_images

        self.num_nodes = 0
        self.num_nodes_per_class = {
            'image':0,
            'background':0,
            'text':0
        }
        self.node_information = []
        self.preview_img = cv2.imread(preview_path)

        for layer in layers:
            if layer == "background":
                self.num_nodes += 1
                self.num_nodes_per_class[layer] = 1
            else:
                self.num_nodes += len(self.all_bboxes[layer])
                self.num_nodes_per_class[layer] = len(self.all_bboxes[layer])
        
        self.distance_matrix = np.zeros((self.num_nodes, self.num_nodes))
        self.construct_features()

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

                    distance = math.sqrt(math.pow(self_middle_x-middle_x, 2)+math.pow(self_middle_y-middle_y,2))
                    distance = format(distance, '.3f')
                
                self.distance_matrix[num_node, node[0]] = distance

    def construct_features(self):
        num_node = 0
        for layer in self.layers:
            for bbox in self.all_bboxes[layer]:
                x, y = int(bbox[0][0]), int(bbox[0][1])
                z, t = int(bbox[2][0]), int(bbox[2][1])
                img = cv2.imread(self.all_images[layer])
                cropped_image = torch.unsqueeze(torch.Tensor(img[y:t, x:z]).permute(2,0,1), 0)
                embedding = self.pretrained_model(cropped_image)
                relative_size = t*z / self.preview_img[0]*self.preview_img[1]
                self.node_information.append([num_node, layer, bbox, embedding, relative_size])
                num_node+=1

        for node in self.node_information:
            self.calculate_distances(node[2], node[0])

    def construct_graph(self):
        pass


