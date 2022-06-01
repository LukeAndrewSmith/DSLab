import os
import cv2

from ringdetector.preprocessing.GeometryUtils import pixel_to_mm, \
    rotateListOfCoords, shiftListOfCoords, roundCoords, pixelDist

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

def selectCoordsFromRings(rings, imgHeight):
    """
    Selects coordinates from a list of rings (for pos file export and width
     measurement)

    :param rings: list of Ring objects
    :param imgHeight: height of the image (max y value)
    :return: list of coordinates [[ring 1 coords], [ring 2 coords], ...]
    """
    perpCoords, _ = getPerpCoords(rings, imgHeight, distance=100)
    return [[perpCoord] for perpCoord in perpCoords]

################################################################################
# Perpendicular Line Helpers
################################################################################
def getIntersectCoord(ring, slope, bottomY):
    intersectCoord = None
    for coord in ring.predCoords:
        if slope < 0 and coord[1] == bottomY:
            intersectCoord = coord
            break
        elif slope > 0 and coord[1] == 0:
            intersectCoord = coord
            break
    return intersectCoord

def getPerpendicularLine(ring, imgHeight):
    firstCoord = ring.predCoords[0]
    lastCoord = ring.predCoords[-1]
    ringSlope = (
        (lastCoord[0] - firstCoord[0]) / (lastCoord[1] - firstCoord[1])
    )
    try:
        slopePerp = -1/ringSlope
    except FloatingPointError:
        # Divide by 0 case
        slopePerp = 99999999999999
    intersectCoord = getIntersectCoord(ring, slopePerp, imgHeight)
    intercept = intersectCoord[0] - slopePerp*intersectCoord[1]
    return slopePerp, intercept

def getXCoordFromY(y, slope, intercept):
    return [int(round(y*slope + intercept)), y]

def getMatchedCoord(ring, slope, intercept, maxDist):
    matchedCoords = []
    minDist = 1000000
    for coord in ring.predCoords:
        perpCoord = getXCoordFromY(coord[1], slope, intercept)
        dist = pixelDist(coord, perpCoord)
        if dist <= maxDist:
            matchedCoords.append((coord[0], coord[1], dist))
        if dist <= minDist:
            minDist = dist
    if matchedCoords:
        matchedCoords.sort(key=lambda x: x[2])
        return (matchedCoords[0][0], matchedCoords[0][1])
    else:
        #print(minDist)
        return None

def checkInBounds(matchedCoord, imgHeight):
    if matchedCoord is None:
        return False
    elif matchedCoord[1] >= 0 and matchedCoord[1] <= imgHeight:
        return True
    else:
        return False

def getPerpCoords(rings, imgHeight, distance):
    lines = []
    slope, intercept = getPerpendicularLine(rings[0], imgHeight)
    lines.append((slope, intercept))
    perpCoords = []

    for i in range(len(rings)):
        ring = rings[i]
        slope, intercept = lines[-1]
        matchedCoord = getMatchedCoord(ring, slope, intercept, distance)
        if matchedCoord:
            perpCoords.append(matchedCoord)
        #else: 
            #print(f"Ring {i} didn't have matched coord")
        inBounds = checkInBounds(matchedCoord, imgHeight)
        # TODO: should introduce a method where if boundDistance or 
        # matchedDistance is too big, reset the line on the previous ring
        # and try again. 
        if not inBounds:
            slope, intercept = getPerpendicularLine(ring, imgHeight)
            lines.append((slope, intercept))
            matchedCoord = getMatchedCoord(ring, slope, intercept, distance)
            perpCoords.append(matchedCoord)
    
    return perpCoords, lines

################################################################################
# Undo Shift and Rotation
################################################################################
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
        f"#C DATED {core.firstYear} \n",
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
