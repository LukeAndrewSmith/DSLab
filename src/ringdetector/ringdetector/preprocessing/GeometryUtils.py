from math import dist, atan2, degrees

import numpy as np
import cv2
import math

def min_bounding_rectangle(points):
    # converts a list of points into the min are rectangle containing all points
    intpoints = np.int0(np.round_(np.asarray(points)))
    rect = cv2.minAreaRect(intpoints)
    box = __order_points(cv2.boxPoints(rect))
    # NOTE: Rectangle points are stored clockwise from the bottom left:
        # 1 ----- 2
        # |       |
        # 0 ----- 3
    
    assert (box[0][1]>=box[1][1] and #check top left vs bottom left
            box[0][0]<=box[2][0] and # check left vs right
            box[0][0]<=box[3][0]), "Assumed rectangle order wrong"
    
    return box


def __order_points(points):
    xSorted = points[np.argsort(points[:, 0]), :]
    leftPoints = xSorted[:2, :]
    leftPoints = leftPoints[np.argsort(leftPoints[:, 1]), :]
    (tl, bl) = leftPoints
    rightPoints = xSorted[2:, :]
    rightPoints = rightPoints[np.argsort(rightPoints[:, 1]), :]
    (tr, br) = rightPoints
    return np.array([bl, tl, tr, br], dtype="float32")


def transform_to_xywha(box):
    """transform into compatible format for detectron2"""
    xc, yc = (box[0] + box[2])/2 # center point
    w = dist(box[0], box[3]) # width
    h = dist(box[0], box[1]) #height
    
    if box[0][0] >= box[1][0]:
        a = degrees(2 * atan2(h, w)) ## rotation angle in counter-clockwise
    else:
        a = - degrees(2 * atan2(h, w))

    return [xc, yc, w, h, a]


def transform_to_xywh(box):  
    """transform into compatible format for detectron2 no angle"""
    # determine max outer coords:
    xmin = np.min(box[:][0])
    xmax = np.max(box[:][0])
    ymin = np.min(box[:][1])
    ymax = np.max(box[:][1])

    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2
    w = xmax - xmin  # width
    h = ymax - ymin  # height

    return [xc, yc, w, h]


def transform_to_xyxy(box): 
    """transform into compatible format for detectron2 no angle, equiv to max_bounding_box"""
    xmin = np.min(box[:, 0])
    xmax = np.max(box[:, 0])
    ymin = np.min(box[:, 1])
    ymax = np.max(box[:, 1])

    return [xmin, ymin, xmax, ymax]


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

def pixelDist(a, b):
    dx2 = (a[0]-b[0])**2
    dy2 = (a[1]-b[1])**2
    return math.sqrt(dx2 + dy2)

def rotateCoords(coords, rotMat):
    coords = [coords[0], coords[1], 1] # Pad with 1 as rotMat is 2x3 ( * 3x1 = 2x1 ), 1 as we want to take into account shift
    result = np.matmul(np.array(rotMat), np.array(coords))
    #if round: result = result.astype(int)
    return list(result)

def rotateListOfCoords(coordList, rotMat):
    shapes = [
        [rotateCoords(coords, rotMat) 
        for coords in shape] for shape in coordList
    ]
    return shapes

def shiftCoords(coord, shift):
    return list(np.array(coord) - np.array(shift))

def shiftListOfCoords(shape, shift):
    shape = [shiftCoords(coord,shift) for coord in shape]
    return shape

def roundCoords(coordList):
    return [
        [[round(coord) for coord in coords]
                for coords in shape] for shape in coordList
    ]