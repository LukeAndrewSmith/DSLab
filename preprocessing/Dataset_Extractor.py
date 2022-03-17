from cmath import rect
import os
import numpy as np
import cv2
from preprocessing.Image_Annotation import Image_Annotation
from preprocessing.geometry import mm_to_pixel
from preprocessing.Paths import GENERATED_DATASETS_INNER, LABELME_JSONS, IMAGES, POINT_LABELS, GENERATED_DATASETS_INNER

class Dataset_Extractor:
# Class to extract datasets from image and core annotations

    def __init__(self):
        self.coreAnnotations = self._initCoreAnnotations()

    def _initCoreAnnotations(self):
        coreAnnotations = []
        for file in os.listdir(LABELME_JSONS):
            coreAnnotations = coreAnnotations + Image_Annotation(LABELME_JSONS+file, POINT_LABELS).core_annotations
        return coreAnnotations


    ########################################
    def createInnerDataset(self):
        # self._apply(self._processCore, self.coreAnnotations)
        self._apply(self._processCore, [self.coreAnnotations[0]]) # TODO: Only run for one core for now

    def _processCore(self, core):
        # TODO: all these functions have side effects as they modify core variables directly, not clean, should update
        core             = self._convertMMToPX(core)
        img              = self._getImage(core)
        core, rotatedImg = self._rotateImagePointsShapes(core, img)
        croppedImg       = self._cropImage(core, rotatedImg)
        core             = self._shiftAllPoints(core)
        self._saveImage(croppedImg, GENERATED_DATASETS_INNER+core.imageName)
        self._savePosFile(core) # TODO: should we convertPXToMM() before saving

    #################
    def _convertMMToPX(self, core):
        core.pointLabels = [[[mm_to_pixel(coord, core.dpi) for coord in coords]
                                                           for coords in shape] 
                                                           for shape in core.pointLabels]
        core.gapLabels = [[[mm_to_pixel(coord, core.dpi) for coord in coords]
                                                         for coords in shape] 
                                                         for shape in core.gapLabels]
        core.distToPith = mm_to_pixel(core.distToPith, core.dpi)
        return core

    #################
    def _getImage(self,core):
        # imagePath = self.coreAnnotations.imagePath # TODO: Wrong Image path for now as I changed the file structure
        imagePath = IMAGES + "KunA08.jpg" # TODO: Hardcode for testing
        img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        return img

    #################
    def _rotateImagePointsShapes(self, core, img):
        rotMat = self._getRotationMatrix(core.innerRectangle)

        rotatedImg = cv2.warpAffine(img, rotMat, img.shape[1::-1])
        # ------ TODO: find a more elegant way of assigning the new variables, such that the lists are automatically updated----
        core.innerRectangle, core.outerRectangle         = self._rotateRectangles(core.rectangles, rotMat)
        core.cracks, core.bark, core.ctrmid, core.ctrend = self._rotateListOfCoords(core.shapes, rotMat)
        core.rectangles = [core.innerRectangle, core.outerRectangle]
        core.shapes     = [core.cracks, core.bark, core.ctrmid, core.ctrend]
        # ----------------------------------------------------------------------------------------------------------------------
        core.pointLabels = self._rotateListOfCoords(core.pointLabels, rotMat)
        core.gapLabels   = self._rotateListOfCoords(core.gapLabels, rotMat)

        return core, rotatedImg

    def _getRotationMatrix(self,rectangle):
        _, topLeft, topRight, _ = rectangle
        run  = topRight[0] - topLeft[0]
        rise = topRight[1] - topLeft[1]
        angle = np.degrees(np.arctan(rise/run))
        center = (int(topLeft[0] + run/2), int(topLeft[1] + rise/2)) # Top left pixel is (0,0)
        rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return rotMat

    def _rotateRectangles(self, rectangles, rotMat):
        rectangles = [[self._rotateCoords(coords, rotMat) for coords in rectangle] 
                                                          for rectangle in rectangles]
        rectangles = [self._roundRectangleCoords(rectangle) for rectangle in rectangles]        
        return rectangles

    def _rotateListOfCoords(self, list, rotMat):
        shapes = [[self._rotateCoords(coords, rotMat, round=True) for coords in shape] 
                                                                  for shape in list]
        return shapes

    def _rotateCoords(self, coords, rotMat, round=False):
        coords = [coords[0], coords[1], 1] # Pad with 1 as rotMat is 2x3 ( * 3x1 = 2x1 ), 1 as we want to take into account shift
        result = np.matmul(np.array(rotMat), np.array(coords))
        if round: result = result.astype(int)
        return list(result)

    def _roundRectangleCoords(self, rectangle):
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
    def _shiftAllPoints(self, core):
        [_, topLeft, _, _] = core.innerRectangle # Should now be (0,0), hence shift all points by topLeft
        # ------ TODO: find a more elegant way of assigning the new variables, such that the lists are automatically updated----
        core.innerRectangle, core.outerRectangle         = [self._shiftListOfCoords(list, topLeft) for list in core.rectangles] 
        core.cracks, core.bark, core.ctrmid, core.ctrend = [self._shiftListOfCoords(list, topLeft) for list in core.shapes]
        core.rectangles = [core.innerRectangle, core.outerRectangle] 
        core.shapes     = [core.cracks, core.bark, core.ctrmid, core.ctrend]
        # ----------------------------------------------------------------------------------------------------------------------
        core.pointLabels = [self._shiftListOfCoords(list, topLeft) for list in core.pointLabels]
        core.gapLabels   = [self._shiftListOfCoords(list, topLeft) for list in core.gapLabels]
        return core

    def _shiftListOfCoords(self, shape, shift):
        shape = [self._shiftCoords(coord,shift) for coord in shape]
        return shape

    def _shiftCoords(self, coord, shift):
        return list(np.array(coord) - np.array(shift))

    #################
    def _cropImage(self, core, img):
        _, topLeft, _, bottomRight = core.innerRectangle
        croppedImage = img[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]
        return croppedImage
    
    #################
    def _saveImage(self, img, path):
        cv2.imwrite(path,img)
    
    #################
    def _savePosFile(self, core):
        # TODO:
        pass


    ########################################
    # Helpers
    def _apply(self, func, list):
        for item in list:
            func(item)