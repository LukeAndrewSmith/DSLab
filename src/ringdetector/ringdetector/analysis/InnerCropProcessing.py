import numpy as np
import wandb
import cv2
from tqdm import tqdm

from ringdetector.analysis.RingDetection import findRings
from ringdetector.utils.posFileExport import undoShiftRotation, \
    saveSanityCheckImage, selectCoordsFromRings, savePosFile
from ringdetector.preprocessing.GeometryUtils import roundCoords
from ringdetector.analysis.Plotters import exportInferenceLinePlot

################################################################################
#                                 Inference Workflow
################################################################################

def inferInnerCrops(innerCrops, savePath):
    """
    Inner Crop handler for inference workflow, detects rings and exports pos
    files.

    :param innerCrops: list of (core, croppedImg)
    :param savePath: export folder for POS files
    :return: nothing, exports pos files and a sanity check pos file img
    """ 

    allCoords = []
    for core, croppedImg in tqdm(innerCrops, "Predicting rings"):
        rings = findRings(croppedImg)
        # TODO: could we call selectcoords inside savePosFile?
        exportInferenceLinePlot(croppedImg, core.sampleName, rings, savePath)
        ringCoords = selectCoordsFromRings(rings, croppedImg.shape[0])
        savePosFile(ringCoords, core, savePath)
        allCoords += undoShiftRotation(ringCoords, core)

    saveSanityCheckImage(
        allCoords, innerCrops[0][0].getOriginalImage(), savePath)
    
################################################################################
# Scoring (for supervised workflow)
################################################################################
def scoreCore(rings, pointLabels):
    """ Scoring of Core given list of rings and POS file labels.
    """
    pointLabels = roundCoords(pointLabels)
    rings = __scoreRings(rings, pointLabels)
    scoreDict = __computeScore(rings, pointLabels)
    return scoreDict

#### 
# Scoring helpers
def __scoreRings(rings, pointLabels): 
    for ring in rings:
        ring.scoreRing(pointLabels)
    return rings

def __collectRings(rings, ringLabel):
    """ General idea: each ring has picked a closest point. With a specific set of ring points (one ring can be indicated by two ground truth points), we loop through the rings and find rings that have picked one of the ring points as their closest label. We add any ring that has picked this ring label to the list of matched rings (later, rings that are matched but not closest will count as false positives). If a matched ring is also the closest (or equally close) seen so far for this ring, it is added to closestRings. 

    :param rings: list of Ring objects
    :param ringLabel: single ring label from ground truth pos file, 
    which can consist of one or two sets of coordinates. [[c1,c1],[c2,c2]]
    :return closestRings:
    """
    matchedRings = []
    closestRingsDist = 100000
    closestRings = []
    for point in ringLabel:
        for ring in rings:
            if ring.closestLabelPoint == point:
                matchedRings.append(ring)
                if ring.minDist < closestRingsDist:
                    closestRings = [ring]
                    closestRingsDist = ring.minDist
                elif ring.minDist == closestRingsDist:
                    if ring not in closestRings:
                        closestRings.append(ring)
    return closestRings, closestRingsDist, matchedRings 

def __computePrecisionRecall(truePos, falsePos, falseNeg):
    precision = 0
    recall = 0
    if (truePos + falsePos) != 0:
        precision = truePos / (truePos + falsePos)
    
    if (truePos + falseNeg) != 0:
        recall = truePos / (truePos + falseNeg)
    
    return precision, recall

def __computeScore(rings, pointLabels, distance=10):
    """ For each labeled ring (can have two labels for one ring), 
    find rings that picked this ring as their closest label,
    then assign the ring label and its rings to TP, FP, FN according to
    criteria below.
    """
    truePosRings = []
    truePosLabels = []
    falsePosRings = []
    falseNegLabels = []
    for ringLabels in pointLabels:
        closestRings, closestRingsDist, matchedRings = __collectRings(
            rings, ringLabels
        )
        ringFalsePosRings = set(matchedRings) - set(closestRings)
        falsePosRings.extend(list(ringFalsePosRings))
        # next, deal with closest rings
        if closestRingsDist < distance and len(closestRings) == 1:
            # Single true positive case
            truePosRings.append(closestRings[0])
            truePosLabels.append(ringLabels)
        elif closestRingsDist < distance and len(closestRings) > 1:
            #bestRing, otherRings = pickBestRing(closestRings)
            # pick best ring by min mse
            closestRings.sort(key=lambda x: x.mse)
            # add best ring to true positives
            bestRing = closestRings.pop(0)
            truePosRings.append(bestRing)
            truePosLabels.append(ringLabels)
            # add other rings to false postivies
            falsePosRings.extend(closestRings)
        else:
            # Two subcases here: distance > distance or no closestRings
            falsePosRings.extend(closestRings)

            falseNegLabels.append(ringLabels)

    truePos = len(truePosLabels)
    falsePos = len(falsePosRings)
    falseNeg = len(falseNegLabels)

    precision, recall = __computePrecisionRecall(truePos, falsePos, falseNeg)

    scoreDict = dict(
        truePosRings = truePosRings,
        truePosLabels = truePosLabels,
        falsePosRings = falsePosRings,
        falseNegLabels = falseNegLabels,
        truePos = truePos,
        falsePos = falsePos,
        falseNeg = falseNeg,
        precision = precision,
        recall = recall
    )
    return scoreDict

################################################################################
# Reporting (for supervised workflow)
################################################################################
def reportCore(sampleName, rings, scoreDict):
    """ Logs core processor metrics to wandb
    """
    report = dict(
        core=sampleName,
        edgeCount=len(rings),
        truePos=scoreDict["truePos"],
        falsePos=scoreDict["falsePos"],
        falseNeg=scoreDict["falseNeg"],
        precision=scoreDict["precision"],
        recall=scoreDict["recall"],
    )
    wandb.log(report)

