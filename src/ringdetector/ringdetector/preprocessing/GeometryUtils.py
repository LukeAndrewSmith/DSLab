import numpy as np
import cv2


def min_bounding_rectangle(points):
    # converts a list of points into the min are rectangle containing all points
    points = np.int0(np.asarray(points))
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    #NOTE: DONT WANT TO ROUND HERE OTHERWISE ITS NOT A RECTANGLE
    #box = np.int0(box) . 
    # NOTE: Rectangle points from opencv boxPoints are stored 
        # in order: clockwise from the top left:
        # 1 ----- 2
        # |       |
        # 0 ----- 3
    assert (box[0][1] > box[1][1] and #check top left vs bottom left
            box[0][0]<box[2][0] and # check left vs right
            box[0][0]<box[3][0]), "Assumed rectangle order wrong" 
    return box

def mm_to_pixel(mm, dpi):
    # 25.4 mm is one inch
    # formula:
    pixel = (mm * dpi) / 25.4
    return pixel


def pixel_to_mm(pixel, dpi):
    # 25.4 mm is one inch
    # formula:
    mm = (pixel * 25.4) / dpi
    return mm
