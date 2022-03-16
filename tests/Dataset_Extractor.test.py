# TODO: using syspath trick for now for import, setup nicer structure and use unittest 
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy as cp
import cv2
import numpy as np
from preprocessing.Dataset_Extractor import Dataset_Extractor
from preprocessing.Image_Annotation import Image_Annotation
from preprocessing.Paths import IMAGES, LABELME_JSONS, POINT_LABELS

def getTestCoreAnnotation():
    labelmeJson = LABELME_JSONS+'KunA08.json'
    return Image_Annotation(labelmeJson, POINT_LABELS).core_annotations[0]

def testCreateInnerDataset():
    extractor = Dataset_Extractor()
    extractor.createInnerDataset()
    # TODO: some sort of test

def _addAllAnnotations(image, core):
    # for shape in oldCore.shapes:
    #     image = _addPoints(image, shape, (0,255,0))
    # for rectangle in oldCore.rectangles:
    #     image = _addPoints(image, rectangle, (0,255,0))
    for shape in core.shapes:
        image = _addPoints(image, shape, (255,0,0))
    for rectangle in core.rectangles:
        image = _addPoints(image, rectangle, (255,0,0))
    
    for points in core.pointLabels:
        image = _addPoints(image, points, (0,0,255))
    for points in core.gapLabels:
        image = _addPoints(image, points, (0,0,255))

    return image

def _addPoints(image, points, color):
    for point in points:
        point = np.array(point).astype(int) # Points in original image aren't necessarily rounded
        image = cv2.circle(image, point, 50, color, -1)
    return image

def testRotateCoords():
    extractor = Dataset_Extractor()
    core = getTestCoreAnnotation()

    core = extractor._convertMMToPX(core)
    core, rotatedImage = extractor._rotateImagePointsShapes(core)

    pointsImage = _addAllAnnotations(rotatedImage, core)

    cv2.imshow("testRotateCoords", pointsImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testCropImage():
    extractor = Dataset_Extractor()
    core = getTestCoreAnnotation()

    core, rotatedImage = extractor._rotateImagePointsShapes(core)
    croppedImage = extractor._cropImage(core, rotatedImage)

    cv2.imshow("testCropImage", croppedImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def testShiftCoords():
    extractor = Dataset_Extractor()
    core = getTestCoreAnnotation()

    core = extractor._convertMMToPX(core)
    core, rotatedImage = extractor._rotateImagePointsShapes(core)
    croppedImage = extractor._cropImage(core, rotatedImage)
    core = extractor._shiftAllPoints(core)

    pointsImage = _addAllAnnotations(croppedImage, core)

    cv2.imshow("testCropImage", pointsImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    testCreateInnerDataset()
    testRotateCoords()
    testCropImage()
    testShiftCoords()