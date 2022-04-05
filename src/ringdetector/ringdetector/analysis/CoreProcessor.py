import numpy as np
import os
import logging
import cv2
import pickle 
from tqdm import tqdm

from ringdetector.analysis.ImageProcessor import ImageProcessor
from ringdetector.analysis.EdgeProcessor import EdgeProcessor
from ringdetector.analysis.Edge import Edge
from ringdetector.Paths import GENERATED_DATASETS_INNER_CROPS, \
    GENERATED_DATASETS_INNER_PICKLES, IMAGES, POINT_LABELS

class CoreProcessor:

    def __init__(self, sampleName):
        self.sampleName = sampleName

        picklePath = os.path.join(
            GENERATED_DATASETS_INNER_PICKLES, f"{sampleName}.pkl"
        )
        with open(picklePath, "rb") as f:
            self.core = pickle.load(f)

        #NOTE: round pointLabels to nearest int (whole pixel)
        self.core.pointLabels = [
            [[round(coord) for coord in coords]
                for coords in shape] for shape in self.core.pointLabels
        ]

        impath = os.path.join(GENERATED_DATASETS_INNER_CROPS, 
            f"{sampleName}.jpg")
        self.procImg = ImageProcessor(impath)
        self.procImg.computeGradients(
            method='Canny', threshold1=50, threshold2=100
        )
    
        self.procEdges = EdgeProcessor(self.procImg.gXY)
        self.procEdges.processEdgeInstances(minLength=80)

        self.edges = []
        for shape in tqdm(self.procEdges.processedEdges, desc="Fitting edges"):
            edge = Edge(shape, self.procImg.denoisedImage)
            edge.fitPredict()
            edge.scoreEdge(self.core.pointLabels)
            self.edges.append(edge)

        # TODO: placeholder for some function where we remove edges with 
        # high MSE
        self.filteredEdges = self.edges

        self.truePosEdges = []
        self.truePosLabels = []
        self.truePos = 0
        self.falsePosEdges = []
        self.falsePos = 0
        self.falseNegLabels = []
        self.falseNeg = 0

        
    def __collectEdges(self, ringLabel):
        """ General idea: each edge has picked a closest point. With a specific set of ring points (one ring can be indicated by two ground truth points), we loop through the edges and find edges that have picked one of the ring points as their closest label. We add any edge that has picked this ring label to the list of matched edges (later, edges that are matched but not closest will count as false positives). If a matched edge is also the closest (or equally close) seen so far for this ring, it is added to closestEdges. 
        """
        matchedEdges = []
        closestEdgesDist = 100000
        closestEdges = []
        for point in ringLabel:
            for edge in self.filteredEdges:
                comparePoint = [
                    edge.closestLabelPoint[1], edge.closestLabelPoint[0]
                ]
                if comparePoint == point:
                    matchedEdges.append(edge)
                    if edge.minDist < closestEdgesDist:
                        closestEdges = [edge]
                        closestEdgesDist = edge.minDist
                    elif edge.minDist == closestEdgesDist:
                        if edge not in closestEdges:
                            closestEdges.append(edge)
        return closestEdges, closestEdgesDist, matchedEdges 

    def scoreCore(self):
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
            if closestEdgesDist < 5 and len(closestEdges) == 1:
                # Single true positive case
                self.truePosEdges.append(closestEdges[0])
                self.truePosLabels.append(ringLabels)
            elif closestEdgesDist < 5 and len(closestEdges) > 1:
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
                # Two subcases here: distance > 5 or no closestEdges
                self.falsePosEdges.extend(closestEdges)

                self.falseNegLabels.append(ringLabels)

        self.truePos = len(self.truePosLabels)
        self.falsePos = len(self.falsePosEdges)
        self.falseNeg = len(self.falseNegLabels)

        self.precision = self.truePos / (self.truePos + self.falsePos)
        self.recall = self.truePos / (self.truePos + self.falseNeg)

    ### Plotting the processed Core
    def __plotLabels(self, img, labels, color=(0,255,0)):
        for label in labels:
            for point in label:
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
                [p1[1], p1[0]], 
                [p2[1], p2[0]], 
                color,
                2
            )

    def exportCoreImg(self, dir):
        orig = self.procImg.image
        bgnd = np.dstack([orig,orig,orig])
        self.__plotEdges(bgnd, self.truePosEdges, (255,0,0))
        self.__plotEdges(bgnd, self.falsePosEdges, (0,0,255))
        self.__plotLabels(bgnd, self.truePosLabels, (0,255,0))
        self.__plotLabels(bgnd, self.falseNegLabels, (0,165,255))

        verti = np.concatenate([
            bgnd[:,0:1500,:], 
            bgnd[:,1500:3000,:], 
            bgnd[:,3000:4500,:], 
            bgnd[:,4500:6000,:], 
        ], axis=0)

        exportPath = os.path.join(
            dir, f'{self.sampleName}_processed.jpg'
        )
        cv2.imwrite(exportPath, verti)

