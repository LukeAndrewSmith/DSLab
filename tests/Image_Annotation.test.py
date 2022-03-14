from preprocessing.Image_Annotation import Image_Annotation
import cv2

if __name__ == "__main__":
    pos_path = "/Users/fredericboesel/Documents/master/frühling22/ds_lab/data/Data label pos files"
    json_path = "/Users/fredericboesel/Documents/master/frühling22/ds_lab/data/labels/KunA08.json"
    annotation = Image_Annotation(json_path, pos_path)

    # plot image

    # draw rectangles and points
    print(annotation)
