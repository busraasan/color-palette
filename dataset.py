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

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        self.flags = cv2.KMEANS_RANDOM_CENTERS

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
        composed_text_idxs = self.compose_paragraphs(text_bboxes, text_palettes)
        merged_bboxes = self.merge_bounding_boxes(composed_text_idxs, text_bboxes)

        image_bboxes, image_palette = self.extract_image(preview, image)

    def extract_text_bbox(self, img_path) -> tuple[list[list[int]], list[int], list[list[float]]]:
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
            _, labels, palette = cv2.kmeans(pixels, n_colors, None, self.criteria, 10, self.flags)
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

    def merge_bounding_boxes(composed_text_idxs, bboxes):
        '''
            openCV --> x: left-to-right, y: top--to-bottom
            bbox coordinates --> [[256.0, 1105.0], [1027.0, 1105.0], [1027.0, 1142.0], [256.0, 1142.0]]
                             --> left top, right top, right bottom, left bottom

            TODO: Also return color palettes for each merged box.
        '''
        
        biggest_borders = []
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
                print(idxs[0])
                biggest_borders.append(bboxes[idxs[0]])
        return biggest_borders

    def extract_decor_elements(self):
        pass

    def extract_image(self, preview_path, image_path):
        '''
            Use Template Matching the put a bounding box around the main image. Use it as the position.
            Extract colors using KMeans.
            Return: image color palettes and position list (as bboxes).
        '''
        
        preview_image = cv2.imread(preview_path)
        image = cv2.imread(image_path)

        method = cv2.TM_SQDIFF_NORMED
        result = cv2.matchTemplate(image, preview_image, method)

        mn,_,mnLoc,_ = cv2.minMaxLoc(result)
        MPx,MPy = mnLoc
        trows,tcols = image.shape[:2]
        bbox = [[MPx,MPy], [MPx+tcols,MPy+trows]]
        cropped_image = image[MPx:MPx+tcols, MPy:MPy+trows]

        pixels = np.float32(cropped_image.reshape(-1, 3))

        n_colors = 6

        _, labels, palette = cv2.kmeans(pixels, n_colors, None, self.criteria, 10, self.flags)
        palette = np.asarray(palette, dtype=np.int64)

        return bbox, palette

    def generate_graph(self):
        pass