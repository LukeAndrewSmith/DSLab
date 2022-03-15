# TODO: using syspath trick for now for import, setup nicer structure and use unittest 
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.Dataset_Extractor import Dataset_Extractor

# def getTestCoreAnnotation():
#     labelmeJson = 'data/labelme_jsons/KunA08.json'
#     pointLabelDirectoryPath = 'data/point_labels'
#     return Image_Annotation(labelmeJson, pointLabelDirectoryPath).core_annotations

def testCreateInnerDataset():
    extractor = Dataset_Extractor()
    extractor.createInnerDataset()

if __name__ == "__main__":
    testCreateInnerDataset()