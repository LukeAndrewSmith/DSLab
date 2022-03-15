from preprocessing.geometry import *
import cv2
import os

if __name__ == "__main__":
    # read a polygon in and plot it
    # then find the rect
    # plot the rectangle
    data_path = "/Users/fredericboesel/Documents/master/frühling22/ds_lab/data"
    image_path = os.path.join(data_path, "Scanned images/KunA08.jpg")
    json_path = os.path.join(data_path, "labels/KunA08.json")
    image = cv2.imread(image_path)
    f = open(json_path)
    annotations = json.load(f)

    # this extracts an inner polygon to check the implementation
    inner_poly = list()
    for shape in annotations['shapes']:
        if 'inner' in shape['label']:
            inner_poly.append(shape)

    poly = np.int0(np.asarray(inner_poly[0]['points']))
    box = min_bounding_rectangle(poly)

    # draw the polygon from labelme and then also the adjusted rectangle
    cv2.drawContours(image, [poly], -1, (0,0,0), 5)
    cv2.drawContours(image, [box], -1, (0, 0, 255), 10)

    cv2.imshow('output',image)

    # optional saving to inspect
    # cv2.imwrite('test.jpg', image)

    cv2.waitKey()