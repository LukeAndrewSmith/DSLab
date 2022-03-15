import os
from Image_Annotation import Image_Annotation
from Core_Annotation import Core_Annotation

class DatasetExtractor:
# Class to extract datasets from image and core annotations

    def __init__(self):
        self._initCoreAnnotations()

    def _initCoreAnnotations(self):
        self.coreAnnotations = []
        dir = '../data/labelme_jsons/'
        for file in os.listdir(dir):
            self.coreAnnotations.append(Image_Annotation(dir+file, 'data/point_labels').core_annotations)

    def createInnerDataset(self):
        pass
        self.coreAnnotations.map(self._processCore)
        # For each rectangle

        # innerImages = self.coreAnnotations.map()

    def _processCore(self, core):
        # Shift points
        # Rotate rectangle/points
        # Crop image
        # Save image and pos file
        self._shiftPoints(core)
        self._rotateImageAndPoints()
        self._cropImage()
        self._saveImage()

    def _shiftPoints(self):
        
        pass

    def _rotateImageAndPoints(self):
        pass

    def _cropImage(self):
        pass

    def _saveImage(self):
        pass


if __name__=="__main__":
    extractor = DatasetExtractor()
    extractor.createInnerDataset()
