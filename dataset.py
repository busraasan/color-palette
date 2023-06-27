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

        text_palettes, text_dominants, text_pos = self.extract_text_bbox(text)

    def extract_text_bbox(self, img_path) -> tuple[list[int], list[float]]:
        '''
            Input: path to the text image
            Extract text using paddleOCR.
            Crop text from bounding box.
            Extract colors using Kmeans inside the bbox.
            Return the dominant color and the position.
            
            TODO: Try to combine very close lines as paragraph bbox.
            TODO: If text has more than one color, may be check the counts related to bbox 
            and return two colors accordingly.

            Return: text palettes, dominant colors for each text and position list (as bboxes).
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
        for bbox in boxes:
            x, y = int(bbox[0][0]), int(bbox[0][1])
            z, t = int(bbox[2][0]), int(bbox[2][1])
            cropped_image = image[y:t, x:z]
            pixels = np.float32(cropped_image.reshape(-1, 3))

            _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
            palette = np.asarray(palette, dtype=np.int64)
            palette_w_white = []

            for i, color in enumerate(palette):
                x, y, z = color
                if not (252 < x < 256 and 252 < y < 256 and 252 < z < 256):
                    palette_w_white.append(color)
                else:
                    labels = np.delete(labels, np.where(labels == i))

            _, counts = np.unique(labels, return_counts=True)
            dominant = palette_w_white[np.argmax(counts)]
            palettes.append(palette_w_white)
            dominants.append(dominant)

        return palettes, dominants, boxes

    def extract_decor_elements(self):
        pass

    def extract_image(self):
        pass

    def generate_graph(self):
        pass