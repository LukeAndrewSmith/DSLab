import os
import numpy as np
import cv2
from itertools import chain
import logging
from tqdm import tqdm

from ringdetector.preprocessing.ImageAnnotation import ImageAnnotation
from ringdetector.preprocessing.GeometryUtils import mm_to_pixel,\
    rotateCoords, rotateListOfCoords, shiftListOfCoords
from ringdetector.Paths import GENERATED_DATASETS, GENERATED_DATASETS_INNER, \
    LABELME_JSONS, GENERATED_DATASETS_INNER_CROPS,\
    GENERATED_DATASETS_INNER_PICKLES, POINT_LABELS
        

################################################################################
#                                 Main Function
################################################################################
def extractInnerCrops(labelmeJsonPath=None, openLabelme=False,
                      saveDataset=False):
    
    if saveDataset:
        __setupDirectoriesForSaving()

    #TODO: not sure whether this belongs here
    if labelmeJsonPath and openLabelme:
        os.system("echo Opening labelme. Please be patient for one moment, labelme can be slow to start")
        os.system(f'labelme {labelmeJsonPath} --logger-level fatal &') # Open in background TODO: maybe detect if windows and change the command... also not sure if this will work in docker?
        input("Press Enter to continue...")
    
    coreAnnotations = __getCoreAnnotations(labelmeJsonPath)

    innerCrops = []
    for core in tqdm(coreAnnotations, desc="Extracting Inner Crops"):
        innerCrops.append(__getCroppedImg(core, saveDataset))

    return innerCrops


################################################################################
#                                    Setup                                    
################################################################################

##########################################
# saving
def __setupDirectoriesForSaving():
    paths = [
        GENERATED_DATASETS,
        GENERATED_DATASETS_INNER,
        GENERATED_DATASETS_INNER_CROPS,
        GENERATED_DATASETS_INNER_PICKLES
    ]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
            logging.info(f"Created directory {path} for inner dataset.")
        else:
            logging.info(
                f"Directory {path} already exists, overwriting any "
                "existing files.")


##########################################
# Core annotations
def __getCoreAnnotations(labelmeJsonPath=None):
    if labelmeJsonPath:
        return __initCoreAnnotationsOneImage(labelmeJsonPath)
    else:
        return __initCoreAnnotations()

def __initCoreAnnotationsOneImage(labelmeJsonPath):
    return ImageAnnotation( 
                labelmeJsonPath, 
                POINT_LABELS
            ).core_annotations   

def __initCoreAnnotations():
    logging.info(f"Collecting cores from labelme jsons in {LABELME_JSONS}")
    coreAnnotations = []
    for file in os.listdir(LABELME_JSONS):
        if file.endswith(".json"):
            coreAnnotations.append(
                ImageAnnotation(
                    os.path.join(LABELME_JSONS, file), 
                    POINT_LABELS
                ).core_annotations
            )
    return list(chain(*coreAnnotations))


################################################################################
#                                Processing                            
################################################################################
def __getCroppedImg(core, saveDataset):
    # TODO: all these functions have side effects as they modify core variables directly, not clean, should update
    core             = __convertMMToPX(core)
    img              = core.getOriginalImage()
    core, rotatedImg = __rotateImagePointsShapes(core, img)
    croppedImg       = __cropImage(rotatedImg, core.innerRectangle)
    core             = __shiftAllPoints(core)
    if saveDataset:
        __saveImage(
            croppedImg, 
            os.path.join(
                GENERATED_DATASETS_INNER_CROPS, core.sampleName+".jpg") 
        )
        core.toPickle(GENERATED_DATASETS_INNER_PICKLES)

    #TODO: in principle based on function name (and core as input) this should
    # return just the croppedImg, then we zip in extractInnerCrops.
    return (core, croppedImg) # TODO: is this the format we want it in?


##########################################
# mm to px
def __convertMMToPX(core):
    if core.corePosPath:
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


##########################################
# Rotate
def __rotateImagePointsShapes(core, img):
    core.center, core.angle = __getRotation(core.innerRectangle)

    rotMat = cv2.getRotationMatrix2D(core.center, core.angle, 1.0)
    rotatedImg = cv2.warpAffine(img, rotMat, img.shape[1::-1])
    # ------ TODO: find a more elegant way of assigning the new variables, such that the lists are automatically updated----
    # e.g. use a dict
    core.innerRectangle, core.outerRectangle = __rotateRectangles(core.rectangles, rotMat)
    core.cracks, core.bark, core.ctrmid, core.ctrend = rotateListOfCoords(core.shapes, rotMat)
    core.rectangles = [core.innerRectangle, core.outerRectangle]
    core.shapes     = [core.cracks, core.bark, core.ctrmid, core.ctrend]
    # ----------------------------------------------------------------------------------------------------------------------
    if core.corePosPath:
        core.pointLabels = rotateListOfCoords(core.pointLabels, rotMat)
        core.gapLabels = rotateListOfCoords(core.gapLabels, rotMat)

    return core, rotatedImg

def __getRotation(rectangle):
    _, topLeft, topRight, bottomRight = rectangle
    run  = topRight[0] - topLeft[0]
    rise = topRight[1] - topLeft[1]
    center = (topLeft + bottomRight) / 2
    angle = np.degrees(np.arctan(rise/run))
    return center, angle

def __rotateRectangles(rectangles, rotMat):
    rectangles = [
        [rotateCoords(coords, rotMat) for coords in rectangle] 
        for rectangle in rectangles
    ]
    rectangles = [
        __roundRectangleCoords(rectangle) for rectangle in rectangles
    ]        
    return rectangles

def __roundRectangleCoords(rectangle):
    # x or y coords should match for certain combinations of the rectangle bounding coordinates
    # Hence take the same where they should match when rounding (TODO: randomly chose max instead of min, is there a problem with this?)
    bottomLeft, topLeft, topRight, bottomRight = rectangle
    topLeft[0]     = max(round(topLeft[0]), round(bottomLeft[0]))
    topLeft[1]     = max(round(topLeft[1]), round(topRight[1]))
    bottomLeft[0]  = topLeft[0]
    bottomLeft[1]  = max(round(bottomLeft[1]), round(bottomRight[1]))
    topRight[0]    = max(round(topRight[0]), round(bottomRight[0]))
    topRight[1]    = topLeft[1]
    bottomRight[0] = topRight[0]
    bottomRight[1] = bottomLeft[1]
    return [bottomLeft, topLeft, topRight, bottomRight]


##########################################
# Shift
def __shiftAllPoints(core):
    """ Shift s.t. topLeft is now (0,0), hence shift all points by 
    topLeft"""
    # ------ TODO: find a more elegant way of assigning the new variables, such that the lists are automatically updated----
    [_,topLeft, _, _] = core.innerRectangle 
    core.shift = topLeft

    core.innerRectangle, core.outerRectangle         = [shiftListOfCoords(list, core.shift) for list in core.rectangles] 
    core.cracks, core.bark, core.ctrmid, core.ctrend = [shiftListOfCoords(list, core.shift) for list in core.shapes]
    core.rectangles = [core.innerRectangle, core.outerRectangle] 
    core.shapes     = [core.cracks, core.bark, core.ctrmid, core.ctrend]
    # ----------------------------------------------------------------------------------------------------------------------
    if core.corePosPath:
        core.pointLabels = [shiftListOfCoords(list, core.shift) for list in core.pointLabels]
        core.gapLabels   = [shiftListOfCoords(list, core.shift) for list in core.gapLabels]
    return core


##########################################
# Crop
def __cropImage(img, rectangle):
    _, topLeft, _, bottomRight = rectangle
    croppedImage = img[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]
    return croppedImage


##########################################
# Helpers
def __saveImage(img, path):
    cv2.imwrite(path,img)
