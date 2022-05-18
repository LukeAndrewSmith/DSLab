import cv2
import os
import numpy as np 

################################################################################
# Diagnostic Plots (inference and training workflows)
################################################################################
def __plotLabels(img, labels, color=(0,255,0)):
    for label in labels:
        for point in label:
            if point[0] < 0:
                point[0] = 0
            if point[1] < 0:
                point[1] = 0
            cv2.circle(
                img,
                [point[0], point[1]],
                4,
                color,
                -1
            )

def __plotRings(img, rings, color=(0,0,255)):
    for ring in rings:
        p1 = ring.predCoords[0]
        p2 = ring.predCoords[-1]
        cv2.line(
            img,
            [p1[0], p1[1]], 
            [p2[0], p2[1]], 
            color,
            2
        )

def exportSupervisedLinePlot(rgbCropped, sampleName, resultDict, saveDir):
    orig = cv2.cvtColor(rgbCropped, cv2.COLOR_BGR2GRAY)
    bgnd = np.dstack([orig,orig,orig])

    __plotRings(bgnd, resultDict["truePosRings"], (255,0,0))
    __plotRings(bgnd, resultDict["falsePosRings"], (0,0,255))
    __plotLabels(bgnd, resultDict["truePosLabels"], (0,255,0))
    __plotLabels(bgnd, resultDict["falseNegLabels"], (0,165,255))

    splits = np.floor(np.shape(bgnd)[1]/1500.0).astype(int)
    vertiList = [bgnd[:,(i*1500):(i*1500)+1500,:] for i in range(splits)]
    verti = np.concatenate(vertiList, axis=0)
    
    exportPath = os.path.join(
        saveDir, f'{sampleName}_lineplot.jpg'
    )
    cv2.imwrite(exportPath, verti)

def exportInferenceLinePlot(rgbCropped, sampleName, rings, saveDir):
    orig = cv2.cvtColor(rgbCropped, cv2.COLOR_BGR2GRAY)
    bgnd = np.dstack([orig,orig,orig])

    __plotRings(bgnd, rings, (255,0,0))
    
    splits = np.floor(np.shape(bgnd)[1]/1500.0).astype(int)
    vertiList = [bgnd[:,(i*1500):(i*1500)+1500,:] for i in range(splits)]
    verti = np.concatenate(vertiList, axis=0)
    
    exportPath = os.path.join(
        saveDir, f'{sampleName}_lineplot.jpg'
    )
    cv2.imwrite(exportPath, verti)

def exportShapePlot(rgbCropped, sampleName, rings, saveDir, resultDict = None):
    """ Plot of shape coordinates, works for both inference and supervised 
    workflows s.t. supervised plots ground truth labels """
    orig = cv2.cvtColor(rgbCropped, cv2.COLOR_BGR2GRAY)
    shapeImg = np.dstack([orig,orig,orig])

    c1 = (255,255,187)
    c2 = (159,84,255)
    
    for i, ring in enumerate(rings):
        if i%2 == 0:
            for point in ring.ring:
                shapeImg[point[1], point[0]] = c1
        else:
            for point in ring.ring:
                shapeImg[point[1], point[0]] = c2
    
    if resultDict: 
        __plotLabels(shapeImg, resultDict["truePosLabels"], (0,255,0))
        __plotLabels(shapeImg, resultDict["falseNegLabels"], (0,165,255))
    splits = np.floor(np.shape(shapeImg)[1]/1500.0).astype(int)
    vertiList = [
        shapeImg[:,(i*1500):(i*1500)+1500,:] for i in range(splits)
    ]
    verti = np.concatenate(vertiList, axis=0)
    
    exportPath = os.path.join(
        saveDir, f'{sampleName}_shapeplot.jpg'
    )
    cv2.imwrite(exportPath, verti)
