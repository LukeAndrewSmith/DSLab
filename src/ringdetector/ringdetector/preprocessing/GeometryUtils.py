from math import dist, atan2, degrees

import numpy as np
import cv2

def min_bounding_rectangle(points):
    # converts a list of points into the min are rectangle containing all points
    intpoints = np.int0(np.round_(np.asarray(points)))
    rect = cv2.minAreaRect(intpoints)
    box = __order_points(cv2.boxPoints(rect))
    # NOTE: Rectangle points are stored clockwise from the bottom left:
        # 1 ----- 2
        # |       |
        # 0 ----- 3
    
    # This setting is only correct when the rotation angle is very small, same for __transform_to_xywha
    assert (box[0][1]>=box[1][1] and #check top left vs bottom left
            box[0][0]<=box[2][0] and # check left vs right
            box[0][0]<=box[3][0]), "Assumed rectangle order wrong"
    
    bounding_box = __transform_to_xywha(box)
    
    return bounding_box

def __order_points(points):
    xSorted = points[np.argsort(points[:, 0]), :]
    leftPoints = xSorted[:2, :]
    leftPoints = leftPoints[np.argsort(leftPoints[:, 1]), :]
    (tl, bl) = leftPoints
    rightPoints = xSorted[2:, :]
    rightPoints = rightPoints[np.argsort(rightPoints[:, 1]), :]
    (tr, br) = rightPoints
    return np.array([bl, tl, tr, br], dtype="float32")

def __transform_to_xywha(box): # transform into compatible format for detectron2
    xc, yc = (box[0] + box[2])/2 # center point
    w = dist(box[0], box[3]) # width
    h = dist(box[0], box[1]) #height
    
    if box[0][0] >= box[1][0]:
        a = degrees(2 * atan2(h, w)) ## rotation angle in counter-clockwise
    else:
        a = - degrees(2 * atan2(h, w))

    return [xc, yc, w, h, a]

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
