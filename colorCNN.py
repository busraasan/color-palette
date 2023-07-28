"""

    TODO: 
    *   Make the white backgrounds transparent.
    *   Locate the images in the bounding boxes in preview images in white background. 
        Cut the images and paste to the location on the decoration layaer.
    *   The pasting order: Decoration will have white bg. Paste image. Paste text.
    *   When the white bg images are done, feed them to CNN.
    *   The output will be the missing color which is the background color. 
    *   Use CIELab distances to train

"""

import cv2
import numpy as np
from utils import *
from dataset_processing import ProcessedDeStijl
from PIL import Image

class DestijlProcessorCNN():

    def __init__(self, data_path):
        self.path_dict = {
            'preview': data_path + '/00_preview/',
            'image': data_path + '/02_image/',
            'decoration': data_path + '/03_decoration/',
            'text': data_path + '/04_text/',
        }

        self.rgba_path_dict = {
            'preview': data_path + '/rgba_dataset/00_preview/',
            'image': data_path + '/rgba_dataset/02_image/',
            'decoration': data_path + '/rgba_dataset/03_decoration/',
            'text': data_path + '/rgba_dataset/04_text/',
            'temporary': data_path + '/rgba_dataset/05_temporary/',
        }

        self.xml_path_dict = {
            'preview': data_path + '/xmls/00_preview/',
            'image': data_path + '/xmls/02_image/',
            'decoration': data_path + '/xmls/03_decoration/',
            'text': data_path + '/xmls/04_text/',
        }

        self.processed_dataset = ProcessedDeStijl("../destijl_dataset")

    def whitebg_to_transparent(self, img_path, layer):

        """
            WORKING
        """
        image_bgr = cv2.imread(img_path)
        image_num = img_path[-8:]
        # get the image dimensions (height, width and channels)
        h, w, c = image_bgr.shape
        # append Alpha channel -- required for BGRA (Blue, Green, Red, Alpha)
        image_bgra = np.concatenate([image_bgr, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1)
        # create a mask where white pixels ([255, 255, 255]) are True
        white = np.all(image_bgr == [255, 255, 255], axis=-1)
        # change the values of Alpha to 0 for all the white pixels
        image_bgra[white, -1] = 0
        # save the image
        cv2.imwrite(self.rgba_path_dict[layer]+image_num, image_bgra)

    def locate_images_in_image_layer(self, idx):
        """
            NEEDS TESTING
        """
        method = cv2.TM_SQDIFF_NORMED
        path_idx = "{:04d}".format(idx)
        preview_img = cv2.imread(self.path_dict["preview"]+path_idx+".png")
        preview_bboxes = VOC2bbox(self.xml_path_dict["image"]+path_idx+".xml")[1]
        image_img = cv2.imread(self.path_dict["image"]+path_idx+".png")
        
        boxes = []
        design_boxes = []
        for box in preview_bboxes:
            xmin = box[0][0]
            xmax = box[1][0]
            ymin = box[0][1]
            ymax = box[2][1]
            cropped_img = preview_img[ymin:ymax, xmin:xmax]
            
            if(cropped_img.shape[0] > image_img.shape[0]):
                diff_x = abs(cropped_img.shape[0] - image_img.shape[0])
                image_img = cv2.copyMakeBorder(image_img, diff_x//2+5, diff_x//2+5, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            if(cropped_img.shape[1] > image_img.shape[1]):
                diff_y = abs(cropped_img.shape[1] - image_img.shape[1])
                image_img = cv2.copyMakeBorder(image_img, 0, 0, diff_y//2+5, diff_y//2+5, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            result = cv2.matchTemplate(cropped_img, image_img, method)
            mn,_,mnLoc,_ = cv2.minMaxLoc(result)
            MPx,MPy = mnLoc
            trows,tcols = cropped_img.shape[:2]
            boxes.append([MPx, MPx+tcols, MPy, MPy+trows])
            design_boxes.append([xmin, xmax, ymin, ymax])

        self.check_boxes(design_boxes, idx)
        return boxes, design_boxes
    
    def check_boxes(self, bboxes, idx):
        path_idx = "{:04d}".format(idx)
        im = cv2.imread("../destijl_dataset/02_image/" + path_idx + ".png")
        for box in bboxes:
            # [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            xmin = box[0]
            xmax = box[1]
            ymin = box[2]
            ymax = box[3]
            cv2.rectangle(im,(xmin, ymin),(xmax, ymax),(255,0,0),2)
        cv2.imwrite("check_boxes.jpg", im)

    def map_image_coordinates(self, text_coordinate, design_text_coordinate, design_img_coordinates, design_size, text_size):
        prev_x, prev_y = design_size
        text_x, text_y = text_size

        design_x, design_y = design_text_coordinate[0]
        text_x, text_y = text_coordinate[0]

        diff_x = text_x - design_x
        diff_y = text_y - design_y
        
        new_coordinates = []
        for coordinate in design_img_coordinates:
            for i in range(len(coordinate)):
                if i < 2:
                    coordinate[i] = int(coordinate[i] + diff_x)
                else:
                    coordinate[i] = int(coordinate[i] + diff_y)

                if coordinate[i] < 0:
                    coordinate[i] *= -1
            new_coordinates.append(coordinate)
        
        return new_coordinates
    
    def convert_to_min_max_coordinate(self, box):
        xmin, ymin = np.min(box, axis=0)
        xmax, ymax = np.max(box, axis=0)
        return [int(xmin), int(xmax), int(ymin), int(ymax)]

    def paste_onto_decoration_layer(self, idx):
        """
            Take bboxes from locate_images_in_image_layer
            Crop them and paste onto decoration layer one by one
            Paste text layer
            Save image to rgba_dataset/00_preview
        """
        path_idx = "{:04d}".format(idx)
        preview_path = self.path_dict["preview"] + path_idx + ".png"
        img_path = self.path_dict["image"] + path_idx + ".png"
        decoration_path = self.path_dict["decoration"] + path_idx + ".png"
        text_path = self.path_dict["text"] + path_idx + ".png"
        white_bg_text_path = self.rgba_path_dict["text"] + path_idx + ".png"
        white_bg_img_path = self.rgba_path_dict["image"] + path_idx + ".png"

        img = cv2.imread(white_bg_img_path)
        decoration_img = cv2.imread(decoration_path)
        prev = cv2.imread(preview_path)

        design_size = (prev.shape[0], prev.shape[1])
        text_size = (decoration_img.shape[0], decoration_img.shape[1])

        image_boxes, design_image_boxes = self.locate_images_in_image_layer(idx)

        text_bboxes, white_bg_text_boxes, texts = self.processed_dataset.extract_text_bbox(text_path, preview_path)
        text_bboxes_from_design, composed_text_palettes = self.processed_dataset.extract_text_directly(preview_path, texts)

        design_text_coordinate = text_bboxes_from_design[0]
        text_coordinate = white_bg_text_boxes[0]
        new_image_boxes = self.map_image_coordinates(text_coordinate, design_text_coordinate, design_image_boxes, design_size, text_size)

        white_bg = np.zeros( [decoration_img.shape[0], decoration_img.shape[1], 3] ,dtype=np.uint8)
        white_bg.fill(255)
        cv2.imwrite('bg.jpg', white_bg)

        white_bg = Image.open('bg.jpg')
        decoration_overlay = Image.open(self.rgba_path_dict["decoration"] + path_idx + ".png")
        text_overlay = Image.open(white_bg_text_path)
        white_bg.paste(decoration_overlay, mask=decoration_overlay)
        #white_bg.save(self.rgba_path_dict["temporary"] + path_idx + ".png")
        
        for j, box in enumerate(new_image_boxes):
            xmin1, xmax1, ymin1, ymax1 = box # box place on decoration
            xmin2, xmax2, ymin2, ymax2 = image_boxes[j] # box place on image
            cropped_img = img[ymin2:ymax2, xmin2:xmax2]

            cv2.imwrite(self.rgba_path_dict["temporary"] + path_idx + ".png", cropped_img)
            self.whitebg_to_transparent(self.rgba_path_dict["temporary"] + path_idx + ".png", "temporary")
            cropped_img = Image.open(self.rgba_path_dict["temporary"] + path_idx + ".png")

            offset = (xmin1, ymin1)
            white_bg.paste(cropped_img, offset, mask=cropped_img)

            #white_bg[ymin1:ymax1, xmin1:xmax1] = cropped_img
        
        #cv2.imwrite(self.rgba_path_dict["preview"] + path_idx + ".png", white_bg)
        white_bg.paste(text_overlay, mask=text_overlay)
        white_bg.save(self.rgba_path_dict["preview"] + path_idx + ".png")

    def pipeline(self):
        for idx in range(28, 706):
            print(idx)
            path_idx = "{:04d}".format(idx)

            img_path = self.path_dict["image"]+path_idx+".png"
            text_path = self.path_dict["text"]+path_idx+".png"
            decoration_path = self.path_dict["decoration"]+path_idx+".png"

            self.whitebg_to_transparent(img_path, "image")
            self.whitebg_to_transparent(text_path, "text")
            self.whitebg_to_transparent(decoration_path, "decoration")

            self.paste_onto_decoration_layer(idx)

processor = DestijlProcessorCNN("../destijl_dataset")
processor.pipeline()