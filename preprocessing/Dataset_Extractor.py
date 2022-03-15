import os
import numpy as np
import cv2
from preprocessing.Image_Annotation import Image_Annotation
from preprocessing.geometry import mm_to_pixel

class Dataset_Extractor:
# Class to extract datasets from image and core annotations

    def __init__(self):
        self.coreAnnotations = self._initCoreAnnotations()

    def _initCoreAnnotations(self):
        coreAnnotations = []
        dir = 'data/labelme_jsons/'
        for file in os.listdir(dir):
            coreAnnotations = coreAnnotations + Image_Annotation(dir+file, 'data/point_labels').core_annotations
        return coreAnnotations

    def createInnerDataset(self):
        self._apply(self._processCore, self.coreAnnotations)

    def _processCore(self, core):
        # Convert mm->px
        # Shift points
        # Rotate rectangle/points
        # Crop image
        # Save image and pos file

        core = self._convertMMToPX(core)
        core = self._rotateImageAndPoints(core)
        core = self._shiftPoints(core)
        core = self._cropImage(core)
        core = self._saveImage(core)

    def _convertMMToPX(self, core):
        # TODO: Pixels are indexed from where?
        core.pointLabels = [[[mm_to_pixel(coord, core.dpi) for coord in coords]
                                                           for coords in shape] 
                                                           for shape in core.pointLabels]
        core.gapLabels = [[[mm_to_pixel(coord, core.dpi) for coord in coords]
                                                         for coords in shape] 
                                                         for shape in core.gapLabels]
        core.distToPith = mm_to_pixel(core.distToPith, core.dpi)

    def _rotateImageAndPoints(self, core):
        # print(core.innerRectangle)
        # NOTE: Points are stored in order: counter-clockwise from the top left coordinate:
        # 0 ----- 3
        # |       |
        # 1 ----- 2
        [topLeft, bottomLeft, topRight, bottomRight] = core.innerRectangle
        run  = topRight[0] - topLeft[0]
        rise = topRight[1] - topLeft[1] # TODO: (-ve or +ve)? if 3 heigher than 0, negative otherwise
        print("rise:", rise)
        print("run:", run)
        angle = np.degrees(np.arctan(rise/run))
        print('angle:', angle)
        center = (int(topLeft[0] + run/2), int(topLeft[1] + rise/2)) # TODO: Assuming top left pixel is (0,0)
        print('center:', center) # TODO: verify this center point
        cv2.getRotationMatrix2D(center, angle, 1.0)
        return core

    def rotatePoints(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    def _shiftPoints(self, core):
        return core

    def _cropImage(self, core):
        return core

    def _saveImage(self, core):
        return core

    ###############
    # Helpers
    def _apply(self, func, list):
        for item in list:
            func(item)