import torch
import torch.nn as nn
from torch.utils.data import Dataset

from paddleocr import PaddleOCR,draw_ocr

from PIL import Image, ImageFont
import numpy as np
import cv2

import os

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

        self.dataset_size = len(next(os.walk(self.path_dict['preview']))[2])
        self.text_model = PaddleOCR(use_angle_cls=True, lang='en')

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):

        path_idx = "{:04d}".format(idx)
        preview = self.path_dict['preview'] + path_idx + '.png'
        background = self.path_dict['background'] + path_idx + '.png'
        image = self.path_dict['image'] + path_idx + '.png'
        text = self.path_dict['text'] + path_idx + '.png'
        theme = self.path_dict['theme'] + path_idx + '.png'

        text_palettes, text_dominants, text_bboxes = self.extract_text_bbox(text)

    def extract_text_bbox(self, img_path) -> tuple[list[list[int]], list[int], list[list[float]]]:
        '''
            Input: path to the text image
            Extract text using paddleOCR.
            Crop text from bounding box.
            Extract colors using Kmeans inside the bbox.
            Return the dominant color and the position.
            
            TODO: Try to combine very close lines as paragraph bbox. 
            If the the distance between two bbox is smaller than the bbox height and color is the same,
            we can group them as paragraphs.

            Return: text color palettes, dominant colors for each text and position list (as bboxes).
        '''
        # Parameters for KMeans.
        n_colors = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        result = self.ocr(img_path, cls=True)[0]

        image = Image.open(img_path).convert('RGB')
        boxes = [line[0] for line in result]
        texts = [line[1][0] for line in result]
        image = cv2.imread(img_path)

        palettes = []
        dominants = []

        # Run KMeans for each text object
        for bbox in boxes:
            # Crop the text area
            x, y = int(bbox[0][0]), int(bbox[0][1])
            z, t = int(bbox[2][0]), int(bbox[2][1])
            cropped_image = image[y:t, x:z]
            pixels = np.float32(cropped_image.reshape(-1, 3))

            # Apply KMeans to the text area
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            palette = np.asarray(palette, dtype=np.int64)
            palette_w_white = []

            for i, color in enumerate(palette):
                x, y, z = color
                # Do not add white to the palette since it is the same background in every pic.
                if not (252 < x < 256 and 252 < y < 256 and 252 < z < 256):
                    palette_w_white.append(color)
                else:
                    labels = np.delete(labels, np.where(labels == i))

            _, counts = np.unique(labels, return_counts=True)
            dominant = palette_w_white[np.argmax(counts)]
            palettes.append(palette_w_white)
            dominants.append(dominant)

        return palettes, dominants, boxes

    def compose_paragraphs(text_bboxes, text_palettes):

        '''
            Compose text data into paragraphs.
            Return: Grouped indices of detected text elements.
        '''

        num_text_boxes = len(text_palettes)
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

    def compose_bounding_boxes(self, bboxes):
        pass

    def extract_decor_elements(self):
        pass

    def extract_image(self):
        '''
            Use Template Matching the put a bounding box around the main image. Use it as the position.
            Extract colors using KMeans.
            Return: image color palettes and position list (as bboxes).
        '''
        pass

    def generate_graph(self):
        pass