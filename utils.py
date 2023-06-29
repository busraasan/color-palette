import cv2
import numpy as np
from skimage import color, io

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