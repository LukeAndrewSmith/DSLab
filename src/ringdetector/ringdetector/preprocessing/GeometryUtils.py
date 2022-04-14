import numpy as np
import cv2

def min_bounding_rectangle(points):
    # converts a list of points into the min are rectangle containing all points
    intpoints = np.int0(np.round_(np.asarray(points)))
    rect = cv2.minAreaRect(intpoints)
    box = __order_points(cv2.boxPoints(rect))
    #NOTE: Rectangle points are stored clockwise from the bottom left:
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
