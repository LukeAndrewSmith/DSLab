import os
import cv2

from ringdetector.preprocessing.GeometryUtils import pixel_to_mm, \
    rotateListOfCoords, shiftListOfCoords, roundCoords

#####
# Two main functions
def savePosFile(coords, core, savePath):
    """
    Exports pos file with core info and coordinates.

    :param coords: list of coords to add to pos file, no subselection performed
    in this function
    :param core: CoreAnnotation object containing info for export
    :param savePath: export directory
    :return: None
    """
    cleanCoords = __undoShiftRotationPixelToMM(coords, core)
    pos = __getPosFileLines(cleanCoords, core)
    posPath = os.path.join(savePath, f"{core.sampleName}.pos")
    with open(posPath, "w") as f:
        f.writelines(pos)

def selectCoordsFromRings(rings):
    """
    Selects coordinates from a list of rings (for pos file export and width
     measurement)

    :param rings: list of Ring objects
    :return: list of coordinates [[ring 1 coords], [ring 2 coords], ...]
    """
    # TODO: perpendicular line across rings
    ringCoords = []
    for ring in rings:
        p1 = ring.predCoords[int(len(ring.predCoords)/2)]
        ringCoords.append([p1])
    return ringCoords
    
#####
# Undo Shift and Rotation
def undoShiftRotation(ringCoords, core):
    # undo shift
    shiftedCoords = shiftListOfCoords(
        ringCoords, [-core.shift[0], -core.shift[1]]
    )
    # reverse rotation
    rotMat = cv2.getRotationMatrix2D(core.center, -core.angle, 1.0)
    rotatedCoords = rotateListOfCoords(shiftedCoords, rotMat)
    return rotatedCoords

def __undoShiftRotationPixelToMM(ringCoords, core):
    rotatedCoords = undoShiftRotation(ringCoords, core)
    # conver to mm
    mmCoords = [
        [[pixel_to_mm(coord, core.dpi) for coord in coords] 
        for coords in shape] for shape in rotatedCoords
    ]
    return mmCoords

####
# Pos File Creation
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

####
# Sanity check image
def saveSanityCheckImage(coords, image, savePath):
    img = __plotPointsOnImage(coords, image)
    cv2.imwrite(os.path.join(savePath,"sanityImg.png"), img) # TODO: save path

def __plotPointsOnImage(coords, img):
    roundedCoords = roundCoords(coords)
    for coord in roundedCoords:
        for point in coord:
            cv2.circle(img, point, 10, [0,0,255], -1)
    return img
