from cmath import rect
import os
import numpy as np
import cv2
from itertools import chain
import logging
import copy

from ringdetector.preprocessing.ImageAnnotation import ImageAnnotation
from ringdetector.preprocessing.GeometryUtils import mm_to_pixel
from ringdetector.Paths import GENERATED_DATASETS_INNER, IMAGES, POINT_LABELS


class DatasetExtractor:
# Class to extract datasets from image and core annotations

    def __init__(self):
        self.coreAnnotations = self.__initCoreAnnotations()

    def __initCoreAnnotations(self):
        coreAnnotations = []
        for file in os.listdir(IMAGES):
            if file.endswith(".json"):
                coreAnnotations.append(
                    ImageAnnotation(IMAGES+file, POINT_LABELS).core_annotations
                )
        return list(chain(*coreAnnotations))

    ########################################
    def createInnerDataset(self):
        for core in self.coreAnnotations:
            self.__processCore(core)
        # TODO: clean up
        #self._apply(self._processCore, self.coreAnnotations)
        #self._apply(self._processCore, [self.coreAnnotations[0]])

    def __processCore(self, core):
        # TODO: all these functions have side effects as they modify core variables directly, not clean, should update
        oldCore = copy.deepcopy(core)
        core             = self.__convertMMToPX(core)
        img              = self.__getImage(core)
        core, rotatedImg = self.__rotateImagePointsShapes(core, img)
        croppedImg       = self.__cropImage(rotatedImg, core.innerRectangle)
        core             = self.__shiftAllPoints(core)
        self.__saveImage(
            croppedImg, 
            os.path.join(GENERATED_DATASETS_INNER, core.sampleName+".jpg")
        )
        self.__savePosFile(core) # TODO: should we convertPXToMM() before saving
        #TODO: save core annotations

    #################
    def __convertMMToPX(self, core):
        core.pointLabels = [
            [[mm_to_pixel(coord, core.dpi) for coord in coords] 
            for coords in shape] for shape in core.mmPointLabels
        ]
        core.gapLabels = [
            [[mm_to_pixel(coord, core.dpi) for coord in coords]
            for coords in shape] for shape in core.mmGapLabels
        ]
        if core.mmDistToPith is not None:
            core.distToPith = mm_to_pixel(core.mmDistToPith, core.dpi)
        if core.mmPith: # empty list
            core.pith = [mm_to_pixel(coord, core.dpi) for coord in core.mmPith]
        return core

    #################
    def __getImage(self,core):
        imagePath = os.path.join(IMAGES, core.imagePath)
        img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        return img

    #################
    def __rotateImagePointsShapes(self, core, img):
        rotMat = self.__getRotationMatrix(core.innerRectangle)

        rotatedImg = cv2.warpAffine(img, rotMat, img.shape[1::-1])
        # ------ TODO: find a more elegant way of assigning the new variables, such that the lists are automatically updated----
        # e.g. use a dict
        core.innerRectangle, core.outerRectangle         = self.__rotateRectangles(core.rectangles, rotMat)
        core.cracks, core.bark, core.ctrmid, core.ctrend = self.__rotateListOfCoords(core.shapes, rotMat)
        core.rectangles = [core.innerRectangle, core.outerRectangle]
        core.shapes     = [core.cracks, core.bark, core.ctrmid, core.ctrend]
        # ----------------------------------------------------------------------------------------------------------------------
        core.pointLabels = self.__rotateListOfCoords(core.pointLabels, rotMat)
        core.gapLabels   = self.__rotateListOfCoords(core.gapLabels, rotMat)

        return core, rotatedImg

    def __getRotationMatrix(self,rectangle):
        bottomLeft, topLeft, topRight, bottomRight = rectangle
        run  = topRight[0] - topLeft[0]
        rise = topRight[1] - topLeft[1]
        center = (topLeft + bottomRight) / 2
        angle = np.degrees(np.arctan(rise/run))
        rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return rotMat

    def __rotateRectangles(self, rectangles, rotMat):
        rectangles = [
            [self.__rotateCoords(coords, rotMat) for coords in rectangle] 
            for rectangle in rectangles
        ]
        rectangles = [
            self.__roundRectangleCoords(rectangle) for rectangle in rectangles
        ]        
        return rectangles

    def __rotateListOfCoords(self, coordList, rotMat):
        shapes = [
            [self.__rotateCoords(coords, rotMat) 
            for coords in shape] for shape in coordList
        ]
        return shapes

    def __rotateCoords(self, coords, rotMat):
        coords = [coords[0], coords[1], 1] # Pad with 1 as rotMat is 2x3 ( * 3x1 = 2x1 ), 1 as we want to take into account shift
        result = np.matmul(np.array(rotMat), np.array(coords))
        #if round: result = result.astype(int)
        return list(result)

    def __roundRectangleCoords(self, rectangle):
        # x or y coords should match for certain combinations of the rectangle bounding coordinates
        # Hence take the same where they should match when rounding (TODO: randomly chose max instead of min, is there a problem with this?)
        bottomLeft, topLeft, topRight, bottomRight = rectangle
        topLeft[0]     = max(round(topLeft[0]),     round(bottomLeft[0]))
        topLeft[1]     = max(round(topLeft[1]),     round(topRight[1]))
        bottomLeft[0]  = topLeft[0]
        bottomLeft[1]  = max(round(bottomLeft[1]),  round(bottomRight[1]))
        topRight[0]    = max(round(topRight[0]),    round(bottomRight[0]))
        topRight[1]    = topLeft[1]
        bottomRight[0] = topRight[0]
        bottomRight[1] = bottomLeft[1]
        return [bottomLeft, topLeft, topRight, bottomRight]

    #################
    def __shiftAllPoints(self, core):
        [_,topLeft, _, _] = core.innerRectangle # Should now be (0,0), hence shift all points by topLeft
        # ------ TODO: find a more elegant way of assigning the new variables, such that the lists are automatically updated----
        core.innerRectangle, core.outerRectangle         = [self.__shiftListOfCoords(list, topLeft) for list in core.rectangles] 
        core.cracks, core.bark, core.ctrmid, core.ctrend = [self.__shiftListOfCoords(list, topLeft) for list in core.shapes]
        core.rectangles = [core.innerRectangle, core.outerRectangle] 
        core.shapes     = [core.cracks, core.bark, core.ctrmid, core.ctrend]
        # ----------------------------------------------------------------------------------------------------------------------
        core.pointLabels = [self.__shiftListOfCoords(list, topLeft) for list in core.pointLabels]
        core.gapLabels   = [self.__shiftListOfCoords(list, topLeft) for list in core.gapLabels]
        return core

    def __shiftListOfCoords(self, shape, shift):
        shape = [self.__shiftCoords(coord,shift) for coord in shape]
        return shape

    def __shiftCoords(self, coord, shift):
        return list(np.array(coord) - np.array(shift))

    #################
    def __cropImage(self, img, rectangle):
        _, topLeft, _, bottomRight = rectangle
        croppedImage = img[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]
        return croppedImage
    
    #################
    def __saveImage(self, img, path):
        cv2.imwrite(path,img)
    
    #################
    def __savePosFile(self, core):
        # TODO:
        pass


    ########################################
    #TODO: remove this
    # Helpers
    def __apply(self, func, list):
        for item in list:
            func(item)