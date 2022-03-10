import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage.interpolation import rotate
import cv2


def min_bounding_rectangle(points):
    #contour from points
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


if __name__ == "__main__":
    import json
    import matplotlib.pyplot as plt
    # read a polygon in and plot it
    # then find the rect
    # plot the rectangle
    image = cv2.imread("/Users/fredericboesel/Documents/master/frühling22/ds_lab/data/Scanned images/KunA08.jpg")
    f = open("/Users/fredericboesel/Documents/master/frühling22/ds_lab/data/labels/KunA08.json")
    annotations = json.load(f)
    inner_poly = list()
    for shape in annotations['shapes']:
        if 'bark' in shape['label']:
            inner_poly.append(shape)

    poly = np.asarray(inner_poly[0]['points'])

    cv2.drawContours(image, [poly], 0, (0,0,0), 5)

    box = min_bounding_rectangle(poly)
    cv2.drawContours(image, [box], 0, (0, 0, 255), 10)
    cv2.imshow('output',image)
    cv2.imwrite('goodbutt.jpg', image)
    cv2.waitKey()

