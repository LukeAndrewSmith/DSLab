import cv2
import os
import numpy as np

from ringdetector.preprocessing.ImageAnnotation import ImageAnnotation

def draw_all_ImageAnnotations(annotation: ImageAnnotation):
    image = cv2.imread(os.path.join("/Users/fredericboesel/Documents/master/frühling22/ds_lab/data/labels",annotation.image_path))

    for core in annotation.core_annotations:
        cv2.drawContours(image, [np.int0(np.array(core.inner_bound))], -1, (0,0,0), 5)
        cv2.drawContours(image, [np.int0(np.array(core.outer_bound))], -1, (0,0,0), 5)
        cv2.drawContours(image, [np.int0(np.array(core.bark))], -1 ,(0,0,0), 5)
        cv2.drawContours(image, [np.int0(np.array(core.ctrmid))], -1, (0,0,0), 5)
        cv2.drawContours(image, [np.int0(np.array(core.ctrend))], -1, (0,0,0), 5)
        # cracks:
        for crack in core.cracks:
            cv2.drawContours(image, [np.int0(np.array(crack))], -1, (0,0,0), 5)
        for ring in core.rings:
            # can be multiple points on one ring
            for point in ring:
                cv2.circle(image, np.int0(np.array(point)), radius=5, color=(0,0,0), thickness=10)

    cv2.imshow('test', image)
    cv2.imwrite('goodass.jpg', image)
    cv2.waitKey(0)



if __name__ == "__main__":
    pos_path = "/Users/fredericboesel/Documents/master/frühling22/ds_lab/data/Data label pos files"
    json_path = "/Users/fredericboesel/Documents/master/frühling22/ds_lab/data/labels/KunA08.json"
    annotation = ImageAnnotation(json_path, pos_path)

    # plot image

    # draw rectangles and points
    print(annotation)
    draw_all_ImageAnnotations(annotation)

