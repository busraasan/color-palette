import torch
import torch.nn as nn
from torch_geometric.data import Dataset

import os
from utils import *

from paddleocr import PaddleOCR, draw_ocr

from PIL import Image, ImageFont
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

from torchvision.models import resnet50, ResNet50_Weights

from model.graph import DesignGraph

class ProcessedDeStijl(Dataset):
    def __init__(self, data_path):
        self.path = data_path
        self.path_dict = {
            'preview': data_path + '/00_preview/',
            'background': data_path + '/01_background/',
            'image': data_path + '/02_image/',
            'decoration': data_path + '/03_decoration/',
            'text': data_path + '/04_text/',
            'theme': data_path + '/05_theme/'
        }

        self.data_path = data_path
        self.dataset_size = len(next(os.walk(self.path_dict['preview']))[2])
        self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        self.flags = cv2.KMEANS_RANDOM_CENTERS

        self.layers = ['image', 'background', 'text'] # Take this from config file later
        
        self.pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    def len(self):
        return self.dataset_size

    def get(self, idx):
        '''
            Return a graph object based on the information
        '''
        pass
            
    ######### RUNTIME EXTRACTION ###########


    ######### PROCESSING AND ANNOTATING THE DATASET ###########

    def extract_text_bbox(self, img_path, preview_image_path):
        '''
            Input: path to the text image
            Extract text using paddleOCR.
            Crop text from bounding box.
            Extract colors using Kmeans inside the bbox.
            Return the dominant color and the position.
            
            DONE: Try to combine very close lines as paragraph bbox. 
            If the the distance between two bbox is smaller than the bbox height and color is the same,
            we can group them as paragraphs.

            TODO: Cut images automatically from the sides by a margin.
            When constructing bounding boxes, add these margins back to the coordinates.
            Sometimes texts are extremely small that the model cannot detect.

            Return: text color palettes, dominant colors for each text and position list (as bboxes).
        '''
        # Parameters for KMeans.
        n_colors = 3

        result = self.ocr.ocr(img_path, cls=True)[0]

        image = Image.open(img_path).convert('RGB')
        boxes = [line[0] for line in result]
        texts = [line[1][0] for line in result]
        image = cv2.imread(img_path)
        preview_image = cv2.imread(preview_image_path)

        palettes = []
        dominants = []
        new_bboxes = []

        # Run KMeans for each text object
        for bbox in boxes:
            # Crop the text area
            x, y = int(bbox[0][0]), int(bbox[0][1])
            z, t = int(bbox[2][0]), int(bbox[2][1])
            cropped_image = image[y:t, x:z]

            # Do template matching to find the places at the actual image because not every image has the same size.
            method = cv2.TM_SQDIFF_NORMED
            result = cv2.matchTemplate(cropped_image, preview_image, method)
            mn,_,mnLoc,_ = cv2.minMaxLoc(result)
            MPx,MPy = mnLoc
            trows,tcols = cropped_image.shape[:2]
            # --> left top, right top, right bottom, left bottom
            bbox = [[MPx,MPy], [MPx+tcols, MPy], [MPx+tcols, MPy+trows], [MPx, MPy+trows]]

            # # Apply KMeans to the text area
            # pixels = np.float32(cropped_image.reshape(-1, 3))
            # _, labels, palette = cv2.kmeans(pixels, n_colors, None, self.criteria, 10, self.flags)
            # palette = np.asarray(palette, dtype=np.int64)
            # palette_w_white = []

            # for i, color in enumerate(palette):
            #     x, y, z = color
            #     # Do not add white to the palette since it is the same background in every pic.
            #     if not (252 < x < 256 and 252 < y < 256 and 252 < z < 256):
            #         palette_w_white.append(color)
            #     else:
            #         labels = np.delete(labels, np.where(labels == i))

            # _, counts = np.unique(labels, return_counts=True)
            # dominant = palette_w_white[np.argmax(counts)]
            # palettes.append(palette_w_white)
            # dominants.append(dominant)
            new_bboxes.append(bbox)

        return new_bboxes, boxes, texts

    def compose_paragraphs(self, text_bboxes, text_palettes):

        '''
            Compose text data into paragraphs.
            Return: Grouped indices of detected text elements.
        '''

        num_text_boxes = len(text_bboxes)
        if num_text_boxes == 0:
            return False
        composed_text_idxs = [[0]]
        for i in range(num_text_boxes-1):
            palette1 = text_palettes[i]
            palette2 = text_palettes[i+1]
            if np.array_equal(palette1, palette2):
                bbox1 = text_bboxes[i]
                bbox2 = text_bboxes[i+1]
                height1 = bbox1[0][1] - bbox1[3][1]
                height2 = bbox2[0][1] - bbox2[3][1]
                if abs(bbox1[0][1]-bbox2[0][1]) <= abs(height1)+30:
                    if i != 0 and i not in composed_text_idxs[-1]:
                        composed_text_idxs.append([i])
                    composed_text_idxs[-1].append(i+1)
                else:
                    if i != 0 and i not in composed_text_idxs[-1]:
                        composed_text_idxs.append([i])
                    if i == num_text_boxes-2:
                        composed_text_idxs.append([i+1])
            else:
                if i != 0 and i not in composed_text_idxs[-1]:
                    composed_text_idxs.append([i])
                if i == (num_text_boxes-2):
                    composed_text_idxs.append([i+1])

        return composed_text_idxs

    def merge_bounding_boxes(self, composed_text_idxs, bboxes):
        '''
            openCV --> x: left-to-right, y: top--to-bottom
            bbox coordinates --> [[256.0, 1105.0], [1027.0, 1105.0], [1027.0, 1142.0], [256.0, 1142.0]]
                             --> left top, right top, right bottom, left bottom

            TODO: Also return color palettes for each merged box.
        '''
        
        biggest_borders = []
        if len(bboxes) == 0:
            return biggest_borders
        for idxs in composed_text_idxs:
            smallest_x = smallest_y = 10000
            biggest_y = biggest_x = 0
            if len(idxs) > 1:
                for idx in idxs:
                    bbox = bboxes[idx]
                    bbox_smallest_x, bbox_smallest_y = np.min(bbox, axis=0)
                    bbox_biggest_x, bbox_biggest_y = np.max(bbox, axis=0)

                    if smallest_x > bbox_smallest_x:
                        smallest_x = bbox_smallest_x
                    if smallest_y > bbox_smallest_y:
                        smallest_y = bbox_smallest_y
                    if biggest_x < bbox_biggest_x:
                        biggest_x = bbox_biggest_x
                    if biggest_y < bbox_biggest_y:
                        biggest_y =  bbox_biggest_y

                biggest_border = [[smallest_x, smallest_y], [biggest_x, smallest_y], [biggest_x, biggest_y], [smallest_x, biggest_y]]
                biggest_borders.append(biggest_border)
            else:
                biggest_borders.append(bboxes[idxs[0]])
        return biggest_borders

    def mini_kmeans(self, biggest_border, n_colors, image):
        # for text
        x, y = int(biggest_border[0][0]), int(biggest_border[0][1])
        z, t = int(biggest_border[2][0]), int(biggest_border[2][1])
        cropped_image = image[y:t, x:z]
        pixels = np.float32(cropped_image.reshape(-1, 3))
        _, labels, palette = cv2.kmeans(pixels, n_colors, None, self.criteria, 10, self.flags)
        palette = np.asarray(palette, dtype=np.int64)

        _, counts = np.unique(labels, return_counts=True)
        color = palette[np.argmin(counts)]
        return color


    def extract_text_directly(self, img_path, white_bg_texts):
        n_colors = 2

        result = self.ocr.ocr(img_path, cls=True)[0]

        image = Image.open(img_path).convert('RGB')
        boxes = [line[0] for line in result]
        texts = [line[1][0].replace(" ", "").lower() for line in result]
        white_bg_texts = [elem.replace(" ", "").lower() for elem in white_bg_texts]
        image = cv2.imread(img_path)
        same_idxs = []
        new_boxes = []
        
        composed_text_palettes = []
        for j, elem in enumerate(white_bg_texts):
            for i, text in enumerate(texts):
                if similar(elem, text) > 0.85:
                    new_boxes.append(boxes[i])
                    biggest_border = boxes[i]
                    composed_text_palettes.append(self.mini_kmeans(biggest_border, n_colors, image))
                elif i+1 != len(texts):
                    if similar(elem, text + texts[i+1]) > 0.85:
                        # merge boxes
                        bboxes = [boxes[i], boxes[i+1]]
                        smallest_x = 1000
                        smallest_x = smallest_y = 10000
                        biggest_y = biggest_x = 0
                        for idx in [0, 1]:
                            bbox = bboxes[idx]
                            bbox_smallest_x, bbox_smallest_y = np.min(bbox, axis=0)
                            bbox_biggest_x, bbox_biggest_y = np.max(bbox, axis=0)

                            if smallest_x > bbox_smallest_x:
                                smallest_x = bbox_smallest_x
                            if smallest_y > bbox_smallest_y:
                                smallest_y = bbox_smallest_y
                            if biggest_x < bbox_biggest_x:
                                biggest_x = bbox_biggest_x
                            if biggest_y < bbox_biggest_y:
                                biggest_y =  bbox_biggest_y

                        biggest_border = [[smallest_x, smallest_y], [biggest_x, smallest_y], [biggest_x, biggest_y], [smallest_x, biggest_y]]
                        new_boxes.append(biggest_border)
                        composed_text_palettes.append(self.mini_kmeans(biggest_border, n_colors, image))

        return new_boxes, composed_text_palettes

    def extract_decor_elements(self, decoration_path, preview_path):
        # Determine the number of dominant colors
        num_colors = 6
        
        # Load the image
        image = cv2.imread(decoration_path)
        preview_image = cv2.imread(preview_path)
        
        # Convert the image to the RGB color space
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image2 = image.copy()
        
        # Reshape the image to a 2D array of pixels
        pixels = image.reshape(-1, 3)
        
        # Apply K-means clustering with the determined number of colors
        kmeans = KMeans(n_clusters=num_colors)
        kmeans.fit(pixels)
        
        # Get the RGB values of the dominant colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Convert the colors to the HSV color space
        hsv_colors = [] 

        for i, color in enumerate(colors):
            x, y, z = color
            if not (252 < x < 256 and 252 < y < 256 and 252 < z < 256):
                x, y, z = rgb_to_hsv([x/255, y/255, z/255])
                hsv_colors.append([x*180, y*255, z*255])
        # Convert the image to the HSV color space
        hsv_image = cv2.cvtColor(image2, cv2.COLOR_RGB2HSV)
        
        # Create masks for each dominant color
        masks = []
        hsv_colors = np.asarray(hsv_colors, dtype=np.int32)
        
        colors = []
        for i in range(len(hsv_colors)):
            
            h, s, v = hsv_colors[i, :]
            lower_color = hsv_colors[i, :] - np.array([10, 50, 50])
            upper_color = hsv_colors[i, :] + np.array([10, 255, 255])
            mask = cv2.inRange(hsv_image, lower_color, upper_color)
            colors.append([h,s,v])
            masks.append(mask)
        
        # Find contours in each mask
        contours = []
        for mask in masks:
            contours_color, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours.append(contours_color)
        
        # Draw bounding boxes around the shapes
        image_with_boxes = image.copy()
        bboxes = []
        for i, contour_color in enumerate(contours):
            for contour in contour_color:
                x, y, w, h = cv2.boundingRect(contour)
                # left top, right top, right bottom, left bottom
                bboxes.append([[x,y], [x+w, y], [x+w, y+h], [x,y+h]])

        new_bboxes = delete_too_small_bboxes(np.asarray(bboxes))
        return colors, new_bboxes

    def map_decoration_coordinates(self, design_text_coordinate, text_coordinate, decoration_coordinates, prev_size, text_size):
        # --> [[256.0, 1105.0], [1027.0, 1105.0], [1027.0, 1142.0], [256.0, 1142.0]]
        # --> left top, right top, right bottom, left bottom

        prev_x, prev_y = prev_size
        text_x, text_y = text_size

        design_x, design_y = design_text_coordinate[0]
        text_x, text_y = text_coordinate[0]

        diff_x = text_x - design_x
        diff_y = text_y - design_y
        
        new_coordinates = []
        for coordinate in decoration_coordinates:
            new_coor = []
            for elem in coordinate:
                new_coor.append([elem[0]-diff_x, elem[1]-diff_y])
            new_coordinates.append(new_coor)

        return new_coordinates

    def extract_image(self, preview_path, image_path):
        '''
            Use Template Matching the put a bounding box around the main image. Use it as the position.
            Extract colors using KMeans.
            Return: image color palettes and position list (as bboxes).
        '''
        
        preview_image = cv2.imread(preview_path)
        cropped_image_path = trim_image(image_path, "02_image")
        image = cv2.imread(cropped_image_path)

        if image.shape[0] > preview_image.shape[0]:
            diff_x = image.shape[0] - preview_image.shape[0]
            image = image[(diff_x//2+1):image.shape[0]-(diff_x//2+1), :]
        
        if image.shape[1] > preview_image.shape[1]:
            diff_y = image.shape[1] - preview_image.shape[1]
            image = image[:, (diff_y//2+1):image.shape[1]-(diff_y//2+1)]

        method = cv2.TM_SQDIFF_NORMED
        result = cv2.matchTemplate(image, preview_image, method)
        mn,_,mnLoc,_ = cv2.minMaxLoc(result)
        MPx,MPy = mnLoc
        trows,tcols = image.shape[:2]
        bbox = [[MPx,MPy], [MPx+tcols,MPy+trows]]
        cropped_image = preview_image[MPx:MPx+tcols, MPy:MPy+trows]

        pixels = np.float32(image.reshape(-1, 3))

        n_colors = 6

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, self.criteria, 10, self.flags)
        palette = np.asarray(palette, dtype=np.int64)

        return [bbox], palette

    def annotate_dataset(self):

        for idx in range(388, self.dataset_size):
            print("CURRENTLY AT: ", idx)
            path_idx = "{:04d}".format(idx)
            preview = self.path_dict['preview'] + path_idx + '.png'
            decoration = self.path_dict['decoration'] + path_idx + '.png'
            image = self.path_dict['image'] + path_idx + '.png'
            text = self.path_dict['text'] + path_idx + '.png'

            text_bboxes, white_bg_text_boxes, texts = self.extract_text_bbox(text, preview)
            text_bboxes_from_design, composed_text_palettes = self.extract_text_directly(preview, texts)
            composed_text_idxs = self.compose_paragraphs(text_bboxes_from_design, composed_text_palettes)
            merged_bboxes = []
            if composed_text_idxs != False:
                merged_bboxes = self.merge_bounding_boxes(composed_text_idxs, text_bboxes_from_design)
            image_bboxes, image_palette = self.extract_image(preview, image)

            #decoration_hsv_xpalettes, decoration_bboxes = self.extract_decor_elements(decoration, preview)
            # image_prev = cv2.imread(preview)
            # image_text = cv2.imread(text)
            #mapped_decoration_bboxes = self.map_decoration_coordinates(text_bboxes_from_design[0], white_bg_text_boxes[0], decoration_bboxes, (image_prev.shape[0], image_prev.shape[1]), (image_text.shape[0], image_text.shape[1]))

            #create_xml("../destijl_dataset/xmls/03_decoration", path_idx+".xml", mapped_decoration_bboxes)
            if len(merged_bboxes) == 0:
                create_xml("../destijl_dataset/xmls/04_text", path_idx+".xml", [[[0,0],[0,0],[0,0],[0,0]]])
            else:
                create_xml("../destijl_dataset/xmls/04_text", path_idx+".xml", merged_bboxes)
            create_xml("../destijl_dataset/xmls/02_image", path_idx+".xml",  image_bboxes)

    def process_dataset(self, idx):
        '''
            Process each node. Construct graph features and save the features as pt files.
        '''
        path_idx = "{:04d}".format(idx)

        img_path_dict = {
            'preview': self.data_path + '/00_preview/' + path_idx + '.png',
            'background': self.data_path + '/01_background/' + path_idx + '.png',
            'image': self.data_path + '/02_image/' + path_idx + '.png',
            'text': self.data_path + '/04_text/' + path_idx + '.png',
        }

        annotation_path_dict = {
            'preview': self.data_path + '/xmls' +'/00_preview/' + path_idx + '.xml',
            'image': self.data_path + '/xmls' + '/02_image/' + path_idx + '.xml',
            'text': self.data_path + '/xmls' + '/04_text/' + path_idx + '.xml',
        }
       
        all_bboxes = {
            'image':[], 
            'background':[], 
            'text':[]
        }
        all_images = {
            'image':[], 
            'background':[], 
            'text':[]
        }

        for i, layer in enumerate(self.layers):
            if layer == "background":
                self.preview_img = cv2.imread(img_path_dict[layer])
                img = self.img_path_dict[layer]
                all_images[layer] = img
                all_bboxes[layer] = [[[0, 0], [self.preview_img.shape[0], 0], [self.preview_img.shape[0], self.preview_img.shape[1]], [0, self.preview_img.shape[1]]]]
            else:
                if layer == 'text':
                    img = self.img_path_dict['preview']
                    filename, bboxes = VOC2bbox(annotation_path_dict[layer])
                    all_bboxes[layer] = bboxes
                    all_images[layer] = img
                elif layer == 'image':
                    img_path = self.img_path_dict['image']
                    self.img_img = cv2.imread(img_path)
                    all_bboxes[layer] = [[[0, 0], [self.img_img.shape[0], 0], [self.img_img.shape[0], self.img_img.shape[1]], [0, self.img_img.shape[1]]]]
                    all_images[layer] = img_path

        DesignGraph(self.pretrained_model, all_images, all_bboxes, self.layers, img_path_dict['preview'])

    def trial(self):
        self.get(1)

if __name__ == "__main__":
    dataset = ProcessedDeStijl(data_path='../destijl_dataset')
    #dataset.process_dataset()
    dataset.trial()


