import cv2
import numpy as np
from skimage import color, io
import torch
from collections import defaultdict
import os
import matplotlib.pyplot as plt

from model.GNN import *

from xml.etree.ElementTree import parse, Element, SubElement, ElementTree
import xml.etree.ElementTree as ET
from colormath.color_diff import delta_e_cie2000, delta_e_cmc
from colormath.color_objects import sRGBColor, HSVColor, LabColor, LCHuvColor, XYZColor, LCHabColor, AdobeRGBColor

from PIL import Image, ImageChops
from difflib import SequenceMatcher
import torchvision.ops.boxes as bops
from colormath.color_conversions import convert_color

from model.CNN import *

'''
    USE COLORMATH IN THE LOSS
'''

def normalize_CIELab(cielab_color) -> torch.Tensor:
    """
        Normalize CIELab tensors.
    """
    cielab_color += torch.Tensor([0, 127, 127])
    cielab_color /= torch.Tensor([100, 255, 255])
    return cielab_color

def model_switch_CNN(model_name, out_features=3):
    if "finetuneresnet18_classify" in model_name.lower():
        return FinetuneResNet18_classify()
    elif "resnet18" in model_name.lower():
        return ResNet18()
    elif "colorcnnbigger" in model_name.lower():
        return ColorCNNBigger()
    elif "colorcnn" in model_name.lower():
        return ColorCNN(out_feature=out_features)
    else:
        assert "There is no such model"

def model_switch(model_name, feature_size):

    # Search for some keywords in the model_name and load the correct model accordingly.
    # This way, we can have save more information related to the same model with different configurations.
    if "ColorGNNEmbeddingClassification" in model_name:
        return ColorGNNEmbeddingClassification(feature_size=feature_size)
    elif "ColorGNNSmallEmbedding" in model_name:
        return ColorGNNSmallEmbedding(feature_size=feature_size)
    elif "ColorGNNBiggerEmbedding" in model_name:
        return ColorGNNBiggerEmbedding(feature_size=feature_size)
    elif "ColorGNNEmbedding" in model_name:
        return ColorGNNEmbedding(feature_size=feature_size)
    elif "ColorGNNSmall" in model_name:
        return ColorGNNSmall(feature_size=feature_size)
    elif "ColorGNNBigger" in model_name:
        return ColorGNNBigger(feature_size=feature_size)
    elif "ColorGNN" in model_name:
        return ColorGNN(feature_size=feature_size)
    elif "ColorAttentionGNN" in model_name:
        return ColorAttentionGNN(feature_size=feature_size)
    else:
        assert "There is no such model."
        
def save_model(state_dict, train_losses, val_losses, epoch, save_path):
    torch.save(
        {
            "state_dict": state_dict,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "epoch": epoch,
        },
        os.path.join(save_path, "best.pth"),
    )

def save_plot(train_losses, val_losses, loss_type, loss_path):

    _, ax = plt.subplots()

    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    plt.savefig(os.path.join(loss_path, "loss_" + loss_type + ".png"), dpi=300)
    plt.close()

# COLOR CONVERSION FUNCTIONS
def CIELab2RGB(palette):
    obj_palette = []
    for color in palette:
        color = LabColor(*color)
        color = list(convert_color(color, sRGBColor, through_rgb_type=AdobeRGBColor).get_value_tuple())
        obj_palette.append(color)
    return obj_palette

def RGB2CIELab(palette):
    obj_palette = []
    for color in [palette]:
        color = sRGBColor(*color, is_upscaled=True)
        color = list(convert_color(color, LabColor, through_rgb_type=AdobeRGBColor).get_value_tuple())
        obj_palette.append(color)
    return obj_palette

def bgr2rgb(color):
    x, y, z = color
    color = [z, y, x]
    return color

# CIELAB DISTANCE FUNCTIONS (DEPRECATED)
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
        
    return np.sum((lab1[0] - lab2[0])**2 + (lab1[1] - lab2[1])**2 + (lab1[2] - lab2[2])**2)

def CIELab_distance2(lab1, lab2):
    l = (lab1[0] - lab2[0])**2
    a = (lab1[1] - lab2[1])**2
    b = (lab1[2] - lab2[2])**2
    return l,a,b, torch.sum( (lab1[0] - lab2[0])**2 + (lab1[1] - lab2[1])**2 + (lab1[2] - lab2[2])**2 )

# THESE ARE USED. HOWEVER, YOU NEED TO HAVE THE EDITED VERSION OF COLORMATH LIBRARY BY ME.
def colormath_CIE2000(color1, color2, normalized=False):
    # color1 = LabColor(*color1)
    # color2 = LabColor(*color2)
    x = delta_e_cie2000(color1, color2, normalized)
    return x

def colormath_CIECMC(color1, color2):
    x = delta_e_cmc(color1, color2)
    return x

# BOUNDING BOX TO VOC
# bbox form: [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
# (always like this)
def VOC2bbox(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(float(boxes.find("bndbox/ymin").text))
        xmin = int(float(boxes.find("bndbox/xmin").text))
        ymax = int(float(boxes.find("bndbox/ymax").text))
        xmax = int(float(boxes.find("bndbox/xmax").text))

        if(xmax-xmin) == 0 or (ymax-ymin) == 0:
            continue
        else:
            list_with_single_boxes = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
            list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes

# BBOX 2 VOC Conversion
# input bbox is in the shape: [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
def bbox2VOC(filename, layer_name, bbox):
    bbox_smallest_x, bbox_smallest_y = np.min(bbox, axis=0)
    bbox_biggest_x, bbox_biggest_y = np.max(bbox, axis=0)
    width = bbox_biggest_x - bbox_smallest_x
    height = bbox_biggest_y - bbox_smallest_y
    return filename, width, height, layer_name, bbox_smallest_x, bbox_smallest_y, bbox_biggest_x, bbox_biggest_y

# CREATING XML FILES.
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

def calculate_iou(box_1, box_2):
    # coordinate format [x1, y1, x2, y2] for bbox
    xmin1, ymin1 = np.min(box_1, axis=0)
    xmax1, ymax1 = np.max(box_1, axis=0)
    box1 = torch.tensor([[xmin1, ymin1, xmax1, ymax1]], dtype=torch.float)
    xmin2, ymin2 = np.min(box_2, axis=0)
    xmax2, ymax2 = np.max(box_2, axis=0)
    box2 = torch.tensor([[xmin2, ymin2, xmax2, ymax2]], dtype=torch.float)
    return bops.box_iou(box1, box2)

def calculate_overlap(box_1, box_2):
    xmin1, ymin1 = np.min(box_1, axis=0)
    xmax1, ymax1 = np.max(box_1, axis=0)
    # print(xmin1, ymin1,  xmax1, ymax1)
    w1 = xmax1 - xmin1
    h1 = ymax1 - ymin1
    box1 = torch.tensor([[xmin1, ymin1, xmax1, ymax1]], dtype=torch.float)
    xmin2, ymin2 = np.min(box_2, axis=0)
    xmax2, ymax2 = np.max(box_2, axis=0)
    # print(xmin2, ymin2,  xmax2, ymax2)
    w2 = xmax2 - xmin2
    h2 = ymax2 - ymin2
    box2 = torch.tensor([[xmin2, ymin2, xmax2, ymax2]], dtype=torch.float)

    # im = cv2.imread("../destijl_dataset/00_preview/0141.png")
    # cv2.rectangle(im,(xmin1,ymin1),(xmax1, ymax1),(0,255,0),2)
    # cv2.rectangle(im,(xmin2,ymin2),(xmax2, ymax2),(0,255,0),2)
    # cv2.imwrite("check_rects"+str(w1*h1)+".jpg", im)

    dx = min(xmax1, xmax2) - max(xmin1, xmin2)
    dy = min(ymax1, ymax2) - max(ymin1, ymin2)

    if (dx>=0) and (dy>=0):
        overlaping_area = dx*dy
    else:
        overlaping_area = 0

    a1 = w1*h1
    a2 = w2*h2

    #print(a1, a2, overlaping_area)
    smallest_rectangle_area  = a1 if a1 < a2 else a2

    overlap_ratio = overlaping_area/smallest_rectangle_area

    return overlap_ratio

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

def check_distributions(colors=None):
    
    # If you dont provide any colors, then just calculate the distribution of the 
    # toy dataset input.
    if colors == None:
        path = "../shape_dataset/processed_rgb_toy_dataset"
        colors = []
        for i in range(1000):
            data_path = path+"/data_{:04d}.pt".format(i)
            data = torch.load(data_path)
            for node in data.x:
                rgb = node[-3:]
                colors.append(rgb)

    # Otherwise get color list and calculate.
    n_bins = 20
    colors = np.array(colors)
    fig, axs = plt.subplots(1, 3, figsize=(30,10))
    plt.suptitle("Evaluation Channel Distributions")
    N, bins, patches = axs[0].hist(colors[:, -3], bins=n_bins)
    axs[0].set_title("Red Channel")
    axs[0].set_xlabel("RGB Value")
    axs[0].set_ylabel("Count")
    N, bins, patches = axs[1].hist(colors[:, -2], bins=n_bins)
    axs[1].set_title("Green Channel")
    axs[1].set_xlabel("RGB Value")
    axs[1].set_ylabel("Count")
    N, bins, patches = axs[2].hist(colors[:, -1], bins=n_bins)
    axs[2].set_title("Blue Channel")
    axs[2].set_xlabel("RGB Value")
    axs[2].set_ylabel("Count")
    plt.show()
    plt.savefig("dist")
    plt.close()


    