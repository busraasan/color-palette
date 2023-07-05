import cv2
import numpy as np
from skimage import color, io

from collections import defaultdict
import os
import csv

from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET

from PIL import Image, ImageChops
from difflib import SequenceMatcher

def bgr2rgb(color):
    x, y, z = color
    color = [z, y, x]
    return color

def CIELab_distance(color1, color2, color_space="RGB"):
    
    if color_space == "BGR":
        color1 = bgr2rgb(color1)
        color2 = bgr2rgb(color2)

    if color_space == "RGB" or color_space == "BGR":
        lab1 = color.rgb2lab(color1)
        lab2 = color.rgb2lab(color2)
    elif color_space == "HSV":
        lab1 = color.hsv2lab(color1)
        lab2 = color.hsv2lab(color2)
    else:
        assert "Undefined input color space!"

    return np.sqrt((lab1[0] - lab2[0])**2 + (lab1[1] - lab2[0])**2 + (lab1[2] - lab2[0])**2)

def VOC2bbox(bbox):
    pass

def bbox2VOC(filename, layer_name, bbox):
    bbox_smallest_x, bbox_smallest_y = np.min(bbox, axis=0)
    bbox_biggest_x, bbox_biggest_y = np.max(bbox, axis=0)
    width = bbox_biggest_x - bbox_smallest_x
    height = bbox_biggest_y - bbox_smallest_y
    return filename, width, height, layer_name, bbox_smallest_x, bbox_smallest_y, bbox_biggest_x, bbox_biggest_y

def create_xml(folder, filename, bbox_list):

    x = folder.split("/")
    bbox_list_voc = [bbox2VOC(filename, x[-1], bbox) for bbox in bbox_list]

    root = Element('annotation')
    SubElement(root, 'folder').text = folder
    SubElement(root, 'filename').text = filename
    SubElement(root, 'path').text = './images' +  filename
    source = SubElement(root, 'source')
    SubElement(source, 'database').text = 'Unknown'

    # Details from first entry
    e_filename, e_width, e_height, e_class_name, e_xmin, e_ymin, e_xmax, e_ymax = bbox_list_voc[0]
    
    size = SubElement(root, 'size')
    SubElement(size, 'width').text = str(e_width)
    SubElement(size, 'height').text = str(e_height)
    SubElement(size, 'depth').text = '3'

    SubElement(root, 'segmented').text = '0'

    for entry in bbox_list_voc:
        e_filename, e_width, e_height, e_class_name, e_xmin, e_ymin, e_xmax, e_ymax = entry
        
        obj = SubElement(root, 'object')
        SubElement(obj, 'name').text = e_class_name
        SubElement(obj, 'pose').text = 'Unspecified'
        SubElement(obj, 'truncated').text = '0'
        SubElement(obj, 'difficult').text = '0'

        bbox = SubElement(obj, 'bndbox')
        SubElement(bbox, 'xmin').text = str(e_xmin)
        SubElement(bbox, 'ymin').text = str(e_ymin)
        SubElement(bbox, 'xmax').text = str(e_xmax)
        SubElement(bbox, 'ymax').text = str(e_ymax)

    #indent(root)
    tree = ElementTree(root)
    
    xml_filename = os.path.join('.', folder, os.path.splitext(filename)[0] + '.xml')
    tree.write(xml_filename)

def trim_image(img_path, layer_type):
    x = img_path.split("/")
    with Image.open(img_path) as im:
        bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            im = im.crop(bbox)
        im = im.save("../destijl_dataset/"+layer_type+"_cropped/"+x[-1])

    return "../destijl_dataset/"+layer_type+"_cropped/"+x[-1]

def NMS(boxes, overlapThresh = 0.98):
    #return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We have a least a box of one pixel, therefore the +1
    indices = np.arange(len(x1))
    for i,box in enumerate(boxes):
        temp_indices = indices[indices!=i]
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        if np.any(overlap) > overlapThresh:
            indices = indices[indices != i]
    return boxes[indices].astype(int)

def delete_same_bboxes(boxes):
    pass

def delete_too_small_bboxes(boxes):
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner

    for i, box in enumerate(boxes):
        xx1 = box[0]
        yy1 = box[1]
        xx2 = box[2]
        yy2 = box[3]
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        if w*h < 5:
            indices = indices[indices != i]
    return boxes[indices].astype(int)

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

    