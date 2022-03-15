import numpy as np
import cv2
import json


def min_bounding_rectangle(points):
    # converts a list of points into the min are rectangle containing all points
    points = np.int0(np.asarray(points))
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
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




