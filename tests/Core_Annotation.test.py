# TODO: using syspath trick for now for import, setup nicer structure and use unittest 
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.Core_Annotation import Core_Annotation
from preprocessing.Image_Annotation import Image_Annotation

def getTestCoreAnnotation():
    labelmeJson = 'data/labelme_jsons/KunA08.json'
    pointLabelDirectoryPath = 'data/point_labels'
    return Image_Annotation(labelmeJson, pointLabelDirectoryPath).core_annotations

def testInitPositionLabels():
    coreAnnotation = getTestCoreAnnotation()
    # print(coreAnnotation[0].pointLabels)
    print(coreAnnotation[0].innerBound)

    # assert()


if __name__ == "__main__":
    testInitPositionLabels()