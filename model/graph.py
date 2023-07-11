
import torch
import torch.nn as nn
from torch_geometric.data import Data

import numpy as np
from utils import *
from PIL import Image
import math
import cv2

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

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        self.flags = cv2.KMEANS_RANDOM_CENTERS

        self.num_nodes = 0
        self.num_nodes_per_class = {
            'image':0,
            'background':0,
            'text':0
        }
        self.layer_classes = {
            'image':0,
            'background':1,
            'text':2
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
                relative_size = t*z / self.preview_img.shape[0]*self.preview_img.shape[1]
                self.node_information.append([num_node, layer, bbox, embedding, relative_size])
                num_node+=1

        for node in self.node_information:
            self.calculate_distances(node[2], node[0])

        # Construct COO format
        rows = self.distance_matrix.shape[0]
        cols = self.distance_matrix.shape[1]
        print(rows+cols)
        for row in range(rows):
            for col in range(cols):
                if row != col:
                    self.COO_matrix.append([row,col])
                    self.edge_features.append(self.distance_matrix[row, col])

    def extract_text_color_from_design(self, preview_path, bbox):
        image = cv2.imread(preview_path)
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
        return text_color

    def extract_image_color_from_design(self, preview_path, bbox):
        image = cv2.imread(preview_path)
        n_colors = 3
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
            # Do not add white to the palette
            if not (252 < x < 256 and 252 < y < 256 and 252 < z < 256):
                palette_w_white.append(color)
            else:
                labels = np.delete(labels, np.where(labels == i))
            x, y, z = color
            palette_w_white.append(color)

        _, counts = np.unique(labels, return_counts=True)
        text_color = palette_w_white[np.argmin(counts)]
        background_color = palette_w_white[np.argmax(counts)] #dominant color

        return background_color

    def construct_graph(self):
        '''
            For now, only one color from the image is used
        '''

        node_features = []
        y = []
        for node in self.node_information:
            num_node, layer, bbox, embedding, relative_size = node
            if layer == 'image':
                color_palette = [self.extract_image_color_from_design(self.preview_path, bbox)]
            elif layer == 'text':
                color_palette = [self.extract_text_color_from_design(self.preview_path, bbox)]
            elif layer == 'background':
                color_palette = [self.extract_image_color_from_design(self.all_images[layer], bbox)]
            
            colors = torch.flatten(torch.Tensor(color_palette))
            feature_vector = torch.cat((torch.Tensor([self.layer_classes[layer]]), torch.flatten(embedding), torch.Tensor([relative_size]), colors))
            node_features.append(feature_vector)
            y.append(colors)

        path_idx = "{:04d}".format(self.idx)
        data = Data(x=node_features, 
                    edge_index=self.COO_matrix,
                    edge_attr=self.edge_features,
                    y=y,
                    )

        torch.save(data, os.path.join('../destijl_dataset/graph_dataset/', f'data_{path_idx}.pt'))



