from preprocessing.Image_Annotation import Image_Annotation
import cv2
import os
import numpy as np


def draw_all_image_annotations(annotation: Image_Annotation):
    image = cv2.imread(os.path.join(data_path,"labels",annotation.image_path))
    # draw all objects that are in each core
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
    cv2.imwrite(f'test_{str(annotation.cores)}.jpg', image)
    cv2.waitKey(0)



if __name__ == "__main__":
    data_path = "/Users/fredericboesel/Documents/master/frühling22/ds_lab/data"
    pos_path = os.path.join(data_path,"Data label pos files")
    json_path = os.path.join(data_path,"labels/KunA08.json")
    annotation = Image_Annotation(json_path, pos_path)
    print(annotation)
    draw_all_image_annotations(annotation)

