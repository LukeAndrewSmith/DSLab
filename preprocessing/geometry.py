import numpy as np
import cv2
import json


def min_bounding_rectangle(points):
    # converts a list of points into the min are rectangle containing all points
    points = np.asarray(points)
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


if __name__ == "__main__":
    # read a polygon in and plot it
    # then find the rect
    # plot the rectangle
    image_path = "data/images/KunA08.jpg"
    json_path = "data/point_labels/KunA08.json"
    image = cv2.imread(image_path)
    f = open(json_path)
    annotations = json.load(f)

    # this extracts an inner polygon to check the implementation
    inner_poly = list()
    for shape in annotations['shapes']:
        if 'inner' in shape['label']:
            inner_poly.append(shape)

    poly = np.asarray(inner_poly[0]['points'])
    box = min_bounding_rectangle(poly)

    cv2.drawContours(image, [poly], 0, (0,0,0), 5)
    cv2.drawContours(image, [box], 0, (0, 0, 255), 10)

    cv2.imshow('output',image)

    #optional saving to inspect
    #cv2.imwrite('goodbutt.jpg', image)
    cv2.waitKey()

