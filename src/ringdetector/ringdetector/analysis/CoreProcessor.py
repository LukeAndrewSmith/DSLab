import numpy as np
import os
import cv2
import pickle
import wandb 


from ringdetector.analysis.RingDetection import findRings
from ringdetector.preprocessing.GeometryUtils import pixel_to_mm,\
    rotateListOfCoords, shiftListOfCoords, roundCoords

from ringdetector.Paths import GENERATED_DATASETS_INNER_CROPS, \
    GENERATED_DATASETS_INNER_PICKLES

class CoreProcessor:

    def __init__(self, 
                sampleName, readType="grayscale",
                denoiseH=25, denoiseTemplateWindowSize=10,
                denoiseSearchWindowSize=21, cannyMin=50, cannyMax=75,
                rightEdgeMethod="simple", invertedEdgeWindowSize=25, 
                mergeShapes1Ball=(10,5), mergeShapes1Angle=np.pi/4,
                mergeShapes2Ball=(20,20), mergeShapes2Angle=np.pi/4, 
                filterLengthImgProportion=0.5,
                filterRegressionAnglesAngleThreshold=np.pi/4):
        self.sampleName = sampleName

        picklePath = os.path.join(
            GENERATED_DATASETS_INNER_PICKLES, f"{sampleName}.pkl"
        )
        with open(picklePath, "rb") as f:
            self.core = pickle.load(f)

        #NOTE: round pointLabels to nearest int (whole pixel)
        self.core.pointLabels = roundCoords(self.core.pointLabels)

        impath = os.path.join(GENERATED_DATASETS_INNER_CROPS, 
            f"{sampleName}.jpg")

        self.imgPath = impath # TODO: pass this in a better manner...
        img = cv2.imread(impath, cv2.IMREAD_COLOR)

        edges = findRings(img, denoiseH=denoiseH,
                denoiseTemplateWindowSize=denoiseTemplateWindowSize,
                denoiseSearchWindowSize=denoiseSearchWindowSize, cannyMin=cannyMin, cannyMax=cannyMax,
                rightEdgeMethod=rightEdgeMethod, invertedEdgeWindowSize=invertedEdgeWindowSize, 
                mergeShapes1Ball=mergeShapes1Ball, mergeShapes1Angle=mergeShapes1Angle,
                mergeShapes2Ball=mergeShapes2Ball, mergeShapes2Angle=mergeShapes2Angle, 
                filterLengthImgProportion=filterLengthImgProportion,
                filterRegressionAnglesAngleThreshold=filterRegressionAnglesAngleThreshold)
        edges = self.__scoreEdges(edges, self.core.pointLabels)

        # TODO: placeholder for some function where we remove edges with 
        # high MSE, or generally filter edges further 
        # (could be in EdgeProcessor)
        # TODO: rename edges to rings
        self.filteredRings = edges

        self.truePosEdges = []
        self.truePosLabels = []
        self.truePos = 0
        self.falsePosEdges = []
        self.falsePos = 0
        self.falseNegLabels = []
        self.falseNeg = 0

    
    def __scoreEdges(self, edges, pointLabels): # TODO: this should find another home
        for edge in edges:
            edge.scoreEdge(pointLabels)
        return edges
        
    def __collectEdges(self, ringLabel):
        """ General idea: each edge has picked a closest point. With a specific set of ring points (one ring can be indicated by two ground truth points), we loop through the edges and find edges that have picked one of the ring points as their closest label. We add any edge that has picked this ring label to the list of matched edges (later, edges that are matched but not closest will count as false positives). If a matched edge is also the closest (or equally close) seen so far for this ring, it is added to closestEdges. 
        """
        matchedEdges = []
        closestEdgesDist = 100000
        closestEdges = []
        for point in ringLabel:
            for edge in self.filteredRings:
                if edge.closestLabelPoint == point:
                    matchedEdges.append(edge)
                    if edge.minDist < closestEdgesDist:
                        closestEdges = [edge]
                        closestEdgesDist = edge.minDist
                    elif edge.minDist == closestEdgesDist:
                        if edge not in closestEdges:
                            closestEdges.append(edge)
        return closestEdges, closestEdgesDist, matchedEdges 

    def scoreCore(self, distance=10):
        """ For each labeled ring (can have two labels for one ring), 
        find edges that picked this ring as their closest label,
        then assign the ring label and its edges to TP, FP, FN according to
        criteria below.
        """
        for ringLabels in self.core.pointLabels:
            closestEdges, closestEdgesDist, matchedEdges = self.__collectEdges(
                ringLabels)
            ringFalsePosEdges = set(matchedEdges) - set(closestEdges)
            self.falsePosEdges.extend(list(ringFalsePosEdges))
            # next, deal with closest edges
            if closestEdgesDist < distance and len(closestEdges) == 1:
                # Single true positive case
                self.truePosEdges.append(closestEdges[0])
                self.truePosLabels.append(ringLabels)
            elif closestEdgesDist < distance and len(closestEdges) > 1:
                #bestEdge, otherEdges = pickBestEdge(closestEdges)
                # pick best edge by min mse
                closestEdges.sort(key=lambda x: x.mse)
                # add best edge to true positives
                bestEdge = closestEdges.pop(0)
                self.truePosEdges.append(bestEdge)
                self.truePosLabels.append(ringLabels)
                # add other edges to false postivies
                self.falsePosEdges.extend(closestEdges)
            else:
                # Two subcases here: distance > distance or no closestEdges
                self.falsePosEdges.extend(closestEdges)

                self.falseNegLabels.append(ringLabels)

        self.truePos = len(self.truePosLabels)
        self.falsePos = len(self.falsePosEdges)
        self.falseNeg = len(self.falseNegLabels)

        # TODO: I just put these checks in for a quick debug, should dive in and find out why they happen
        if (self.truePos + self.falsePos) != 0:
            self.precision = self.truePos / (self.truePos + self.falsePos)
        else:
            self.precision = 0

        if (self.truePos + self.falseNeg) != 0:
            self.recall = self.truePos / (self.truePos + self.falseNeg)
        else:
            self.recall = 0

    ### Wandb reporting
    def reportCore(self):
        """ Logs core processor metrics to wandb (only run after scoreCore)
        """
        report = dict(
            core=self.sampleName,
            edgeCount=len(self.filteredRings),
            truePos=self.truePos,
            falsePos=self.falsePos,
            falseNeg=self.falseNeg,
            precision=self.precision,
            recall=self.recall
        )
        wandb.log(report)

    ### Plotting the processed Core
    def __plotLabels(self, img, labels, color=(0,255,0)):
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

    def __plotEdges(self, img, edges, color=(0,0,255)):
        for edge in edges:
            p1 = edge.predCoords[0]
            p2 = edge.predCoords[-1]
            cv2.line(
                img,
                [p1[0], p1[1]], 
                [p2[0], p2[1]], 
                color,
                2
            )

    def exportLinePlot(self, dir):
        orig = cv2.imread(self.imgPath, cv2.IMREAD_GRAYSCALE)
        bgnd = np.dstack([orig,orig,orig])
        self.__plotEdges(bgnd, self.truePosEdges, (255,0,0))
        self.__plotEdges(bgnd, self.falsePosEdges, (0,0,255))
        self.__plotLabels(bgnd, self.truePosLabels, (0,255,0))
        self.__plotLabels(bgnd, self.falseNegLabels, (0,165,255))

        splits = np.floor(np.shape(bgnd)[1]/1500.0).astype(int)
        vertiList = [bgnd[:,(i*1500):(i*1500)+1500,:] for i in range(splits)]
        verti = np.concatenate(vertiList, axis=0)
        
        exportPath = os.path.join(
            dir, f'{self.sampleName}_lineplot.jpg'
        )
        cv2.imwrite(exportPath, verti)
    
    def exportShapePlot(self, dir):
        shapeImgBase = cv2.imread(self.imgPath, cv2.IMREAD_GRAYSCALE)
        shapeImg = np.dstack([shapeImgBase,shapeImgBase,shapeImgBase])

        c1 = (255,255,187)
        c2 = (159,84,255)
        
        for i, ring in enumerate(self.filteredRings):
            if i%2 == 0:
                for point in ring.ring:
                    shapeImg[point[1], point[0]] = c1
            else:
                for point in ring.ring:
                    shapeImg[point[1], point[0]] = c2
        
        self.__plotLabels(shapeImg, self.truePosLabels, (0,255,0))
        self.__plotLabels(shapeImg, self.falseNegLabels, (0,165,255))
        splits = np.floor(np.shape(shapeImg)[1]/1500.0).astype(int)
        vertiList = [
            shapeImg[:,(i*1500):(i*1500)+1500,:] for i in range(splits)
        ]
        verti = np.concatenate(vertiList, axis=0)
        
        exportPath = os.path.join(
            dir, f'{self.sampleName}_shapeplot.jpg'
        )
        cv2.imwrite(exportPath, verti)

    ### Exports
    def toPickle(self, dir):
        filePath = os.path.join(dir, self.sampleName + "_processed.pkl")
        with open(filePath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def __coordsToString(self, coordSet):
        """ Expected input format [x,y] with x,y floats"""
        return f"{str(round(coordSet[0],3))},{str(round(coordSet[1],3))}"
    
    def __coordListToString(self, coordList):
        """ Expected input format [[x,y],[x,y]] """
        coordStrings = []
        for coordSet in coordList:
            coordStrings.append(self.__coordsToString(coordSet))
        return "  ".join(coordStrings)

    def __getPos(self, mmCoords):
        #TODO: what do we do about Pith info long term for unsupervised?
        pos = [
            f"#DENDRO (Cybis Dendro program compatible format) Coordinate file written as P:\Documents\PhD\Dendro data\Kun\{self.core.sampleName}.pos 2021-07-29 16:16:56 \n",
            f"#Imagefile {self.core.imageName} \n",
            f"#DPI {self.core.dpi} \n",
            "#All coordinates in millimeters (mm) \n",
            "SCALE 1 \n",
            "#C DATED 2020 \n",
            f"#C PithCoordinates=310.409,93.620; DistanceToPith=79.3; YearsToPith=11; \n",
            f"#C Radius=262.001; CalcRadius=Yes; Written=2021-07-29 16:16:56; "
            f"#C CooRecorder=9.6 Nov 25 2020; \n",
            "#C licensedTo=Justine Charlet de Sauvage, justine.charlet@usys.ethz.ch; \n"
        ]
        for coords in mmCoords:
            pos.append(self.__coordListToString(coords) + "\n")
        return pos

    def __plotPosImage(self, dir, coords):
        """ Plots edge candidate coordinates onto original scanned image """
        roundedCoords = roundCoords(coords)
        sc = self.core.getOriginalImage()
        self.__plotLabels(sc, roundedCoords, (0,0,0))
        exportPath = os.path.join(
            dir, f'{self.sampleName}_possc.jpg'
        )
        cv2.imwrite(exportPath, sc)


    def exportPos(self, dir, sanityCheck=False):
        """ creates edge coordinates for pos export and exports pos"""
        # creating specific coordinates per edge
        #NOTE: plotting two points per edge for now
        edgeCoords = []
        for edge in self.filteredRings:
            p1 = edge.predCoords[20]
            p2 = edge.predCoords[-20]
            edgeCoords.append([p1, p2])
        
        # undo shift
        shiftedCoords = shiftListOfCoords(
            edgeCoords, [self.core.shift[0] * -1, self.core.shift[1] * -1]
        )
        
        # opposite rotation
        rotMat = cv2.getRotationMatrix2D(
            self.core.center, -1*self.core.angle, 1.0
        )
        rotatedCoords = rotateListOfCoords(shiftedCoords, rotMat)

        # sanity check: round pixel coords and display on orig image
        if sanityCheck:
            self.__plotPosImage(dir, rotatedCoords)

        # pos file export
        mmCoords = [
            [[pixel_to_mm(coord, self.core.dpi) for coord in coords] 
            for coords in shape] for shape in rotatedCoords
        ]
        pos = self.__getPos(mmCoords)
        posPath = os.path.join(
            dir, f"{self.core.sampleName}.pos"
        )
        with open(posPath, "w") as f:
            f.writelines(pos)
