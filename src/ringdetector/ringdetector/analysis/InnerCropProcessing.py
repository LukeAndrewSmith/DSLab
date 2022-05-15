import numpy as np
import os
import cv2

from ringdetector.analysis.RingDetection import findRings
from ringdetector.preprocessing.GeometryUtils import pixel_to_mm, rotateCoords,\
    rotateListOfCoords, shiftListOfCoords, roundCoords

#####################################################################################
#                                 Main Function                                    
#####################################################################################
def processInnerCrops(innerCrops, savePath,
                        denoiseH=25, denoiseTemplateWindowSize=10,
                        denoiseSearchWindowSize=21, cannyMin=50, cannyMax=75,
                        rightEdgeMethod="simple", invertedEdgeWindowSize=25, 
                        mergeShapes1Ball=(10,5), mergeShapes1Angle=np.pi/4,
                        mergeShapes2Ball=(20,20), mergeShapes2Angle=np.pi/4, 
                        filterLengthImgProportion=0.5,
                        filterRegressionAnglesAngleThreshold=np.pi/4):

    allCoords = []
    for core, croppedImg in innerCrops:
        rings = findRings(croppedImg, denoiseH=denoiseH,
                denoiseTemplateWindowSize=denoiseTemplateWindowSize,
                denoiseSearchWindowSize=denoiseSearchWindowSize, cannyMin=cannyMin, cannyMax=cannyMax,
                rightEdgeMethod=rightEdgeMethod, invertedEdgeWindowSize=invertedEdgeWindowSize, 
                mergeShapes1Ball=mergeShapes1Ball, mergeShapes1Angle=mergeShapes1Angle,
                mergeShapes2Ball=mergeShapes2Ball, mergeShapes2Angle=mergeShapes2Angle, 
                filterLengthImgProportion=filterLengthImgProportion,
                filterRegressionAnglesAngleThreshold=filterRegressionAnglesAngleThreshold)
        ringCoords = __selectCoordsFromRings(rings)
        __savePosFile(ringCoords, core, savePath)
        allCoords += __undoShiftRotation(ringCoords, core)

    __saveSanityCheckImage(allCoords, innerCrops[0][0].getOriginalImage(), savePath)

def __selectCoordsFromRings(rings):
    ringCoords = []
    for ring in rings:
        p1 = ring.predCoords[int(len(ring.predCoords)/2)]
        ringCoords.append([p1])
    return ringCoords
    
def __undoShiftRotation(ringCoords, core):
    # undo shift
    shiftedCoords = shiftListOfCoords(ringCoords, [-core.shift[0], -core.shift[1]])
    # reverse rotation
    rotMat = cv2.getRotationMatrix2D(core.center, -core.angle, 1.0)
    rotatedCoords = rotateListOfCoords(shiftedCoords, rotMat)
    return rotatedCoords

def __undoShiftRotationMMToPixel(ringCoords, core):
    rotatedCoords = __undoShiftRotation(ringCoords, core)
    # conver to mm
    mmCoords = [
        [[pixel_to_mm(coord, core.dpi) for coord in coords] 
        for coords in shape] for shape in rotatedCoords
    ]
    return mmCoords

def __savePosFile(coords, core, savePath):
    cleanCoords = __undoShiftRotationMMToPixel(coords, core)
    pos = __getPosFileLines(cleanCoords, core)
    posPath = os.path.join(savePath, f"{core.sampleName}.pos")
    with open(posPath, "w") as f:
        f.writelines(pos)

def __getPosFileLines(coords, core):
    lines = [
        f"#DENDRO (Cybis Dendro program compatible format) sample: {core.sampleName} \n",
        f"#Imagefile {core.imageName} \n",
        f"#DPI {core.dpi} \n",
        "#All coordinates in millimeters (mm) \n",
        "SCALE 1 \n",
        "#C DATED 2020 \n", # TODO: wrong dating
        "#C licensedTo=Justine Charlet de Sauvage, justine.charlet@usys.ethz.ch; \n"
    ]
    for coord in coords:
        lines.append(f'{__coordListToString(coord)} \n')
    return lines

def __coordListToString(coordList):
    ## Expected input format [[x,y],[x,y]]
    coordStrings = []
    for coordSet in coordList:
        coordStrings.append(__coordsToString(coordSet))
    return "  ".join(coordStrings)

def __coordsToString(coordSet):
    return f"{str(round(coordSet[0],3))},{str(round(coordSet[1],3))}"

def __saveSanityCheckImage(allCoords, image, savePath):
    img = __plotPointsOnImage(allCoords, image)
    cv2.imwrite(os.path.join(savePath,"sanityImg.png"), img) # TODO: save path

def __plotPointsOnImage(coords, img):
    roundedCoords = roundCoords(coords)
    for coord in roundedCoords:
        for point in coord:
            cv2.circle(img, point, 10, [0,0,255], -1)
    return img
