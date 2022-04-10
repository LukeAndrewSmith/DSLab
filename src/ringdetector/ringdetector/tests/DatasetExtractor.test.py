import os
import sys

import copy as cp
import cv2
import numpy as np
from ringdetector.preprocessing.DatasetExtractor import DatasetExtractor
from ringdetector.preprocessing.ImageAnnotation import ImageAnnotation
from ringdetector.Paths import GENERATED_DATASETS_TEST_INNER, IMAGES, POINT_LABELS

def getTestCoreAnnotation():
    labelmeJson = os.path.join(IMAGES, 'KunA08.json')
    return ImageAnnotation(labelmeJson, POINT_LABELS).core_annotations[0]

def testCreateInnerDataset():
    extractor = DatasetExtractor()
    extractor.createInnerDataset()
    # TODO: some sort of test

def __addAllAnnotations(image, core):
    # for shape in oldCore.shapes:
    #     image = _addPoints(image, shape, (0,255,0))
    # for rectangle in oldCore.rectangles:
    #     image = _addPoints(image, rectangle, (0,255,0))
    for shape in core.shapes:
        image = __addPoints(image, shape, (255,0,0))
    for rectangle in core.rectangles:
        image = __addPoints(image, rectangle, (255,0,0))
    
    for points in core.pointLabels:
        image = __addPoints(image, points, (0,0,255))
    for points in core.gapLabels:
        image = __addPoints(image, points, (0,0,255))

    return image

def __addPoints(image, points, color):
    for point in points:
        point = np.array(point).astype(int) # Points in original image aren't necessarily rounded
        image = cv2.circle(image, point, 30, color, -1)
    return image

def testRotateCoords():
    extractor = DatasetExtractor()
    core = getTestCoreAnnotation()

    core = extractor._DatasetExtractor__convertMMToPX(core)
    img = extractor._DatasetExtractor__getImage(core)
    core, rotatedImage = extractor._DatasetExtractor__rotateImagePointsShapes(core, img)

    pointsImage = __addAllAnnotations(rotatedImage, core)

    cv2.imwrite(
        os.path.join(GENERATED_DATASETS_TEST_INNER, "testRotateCoords.jpg"),
        pointsImage
    )
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def testCropImage():
    extractor = DatasetExtractor()
    core = getTestCoreAnnotation()

    core = extractor._DatasetExtractor__convertMMToPX(core)
    img = extractor._DatasetExtractor__getImage(core)
    core, rotatedImage = extractor._DatasetExtractor__rotateImagePointsShapes(core, img)
    croppedImage = extractor._DatasetExtractor__cropImage(
        rotatedImage, core.innerRectangle
        )

    cv2.imwrite(
        os.path.join(GENERATED_DATASETS_TEST_INNER, "testCropImage.jpg"), croppedImage
    )
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
def testShiftCoords():
    extractor = DatasetExtractor()
    core = getTestCoreAnnotation()

    core = extractor._DatasetExtractor__convertMMToPX(core)
    img = extractor._DatasetExtractor__getImage(core)
    core, rotatedImage = extractor._DatasetExtractor__rotateImagePointsShapes(core, img)
    croppedImage = extractor._DatasetExtractor__cropImage(
    rotatedImage, core.innerRectangle
    )
    core = extractor._DatasetExtractor__shiftAllPoints(core)

    pointsImage = __addAllAnnotations(croppedImage, core)

    cv2.imwrite(
        os.path.join(GENERATED_DATASETS_TEST_INNER, 
                     "testCropImagewPoints.jpg"), 
        pointsImage
    )
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
if __name__ == "__main__":
    #testCreateInnerDataset()
    testRotateCoords()
    testCropImage()
    testShiftCoords()