from operator import truediv
import os
import re
import logging
import pickle
from math import dist, atan2, degrees
import numpy as np

from ringdetector.preprocessing.GeometryUtils import min_bounding_rectangle

class CoreAnnotation:
    def __init__(self, labelmeAnnotations, sampleName, corePosPath, imagePath):
        self.sampleName  = sampleName
        self.corePosPath = corePosPath
        self.imagePath = imagePath
        self.imageName = os.path.basename(self.imagePath) ## this var is not used
        self.labelmeAnnotations = labelmeAnnotations

        # Shapes: [ [x,y], ... ]
            # NOTE: do not modify self.shape or self.rectangles, this 
            # wont modify the original variables
        self.innerRectangle = self.__initRectangle("INNER")
        self.innerRectangleAngle = self.__transform_to_xywha(self.innerRectangle)
        self.innerRectangleNoAngle = self.__transform_to_xyxy(self.innerRectangle)
        
        self.outerRectangle = self.__initRectangle("OUTER")
        self.outerRectangleNoAngle = self.__transform_to_xyxy(self.outerRectangle)
        
        self.rectangles      = [self.innerRectangle, self.outerRectangle]
        
        self.cracks = self.__findShape('CRACK', [])
        self.bark   = self.__findShape('BARK', [])
        self.ctrmid = self.__findShape('CTRMID', [])
        self.ctrend = self.__findShape('CTREND', [])
        self.shapes = [self.cracks, self.bark, self.ctrmid, self.ctrend]
        
        self.tricky = self.__initTricky()
        
        # Parsing the pos file, splitting lines into groups
        self.headerLines = []
        self.labelLines = []
        self.gapLines = []
        self.__getLines()

        # Point/Gap Labels: [ [ [x,y], ... ], ... ]
        self.mmPointLabels = self.__initPointLabels()
        self.mmGapLabels = self.__initGapLabels()
        self.pointLabels = []
        self.gapLabels = []

        # Point Label Info:
        self.dpi = self.__getDPI() 
        self.mmPith, self.mmDistToPith, self.yearsToPith = self.__getPithData()

        self.pith = []
        self.distToPith = None

    def __repr__(self) -> str:
        return (f"CoreAnnotation for {self.sampleName} in "
                f"{self.imagePath}")

    ######################
    # labelme Annotations
    def __initRectangle(self, rectType):
        points = self.__findShape(rectType, [])
        assert len(points) > 0, f"Core {self.sampleName} missing inner or "\
            "outer crop label in JSON."
        boundingRect = min_bounding_rectangle(points)
        return boundingRect

    def __initTricky(self):
        if self.__findShape('TRICKY', False): return True
        return False

    def __findShape(self, label, default):
        return next(
            (shape['points'] for shape in self.labelmeAnnotations['shapes'] \
            if shape["label"] == f'{self.sampleName}_{label}'), default)


    ##############################
    # POS File Processing
    ##############################
    def __getDPI(self):
        """ Loop through header lines and extract DPI"""
        result = None
        for line in self.headerLines:
            if '#DPI' in line:
                result = float(self.__safeRegexSearch(line, '#DPI (\d+\.\d+)'))
                break
        return result
    
    def __getPithData(self):
        """ Loop through header lines and extract pith info"""
        pithCoordinates = None
        distanceToPith = None
        yearsToPith = None
        for line in self.headerLines:
            if 'PithCoordinates' in line:
                pithCoordinates = self.__positionStringToFloatArray(
                    self.__safeRegexSearch(
                        line, 'PithCoordinates=(-?\d+\.\d+,-?\d+\.\d+)')
                )
            if 'DistanceToPith' in line:
                distanceToPith = self.__getPithNumbers(
                    line, 'DistanceToPith=(\d+\.\d+)'
                )
            if 'YearsToPith' in line:
                yearsToPith = self.__getPithNumbers(line, 'YearsToPith=(\d+)')
        return pithCoordinates, distanceToPith, yearsToPith

    def __getPithNumbers(self, line, regex):
        regexStr = self.__safeRegexSearch(line, regex)
        floatResult = None
        if regexStr: 
            floatResult = float(regexStr)
        return floatResult

    ##############################
    def __initPointLabels(self):
        pointLabels = [self.__processPositionLine(x) for x in self.labelLines]
        return pointLabels

    def __processPositionLine(self, line):
        lineSplit = line.split(' ')
        positionStrings = [ 
            self.__positionStringToFloatArray(x) for x in lineSplit \
                if x not in ['','\n','.']
        ]
        return positionStrings
 
    def __positionStringToFloatArray(self, positionString):
        if positionString is None:
            return []
        else:
            positionStringSplit = positionString.split(',')
            return [float(x) for x in positionStringSplit]

    ##############################
    def __initGapLabels(self):
        gapLabels = [ 
            self.__processGapPositionLine(x) for x in self.gapLines
        ]
        gapLabels = [ x for x in gapLabels if ( len(x) != 0 )] 
        return gapLabels      

    def __processGapPositionLine(self, line):
        lineSplit = line.split(' ')
        gapStrings = [ self.__processGaps(x) for x in lineSplit \
            if x.startswith("D") and "#%gap" not in x]
        return [ self.__positionStringToFloatArray(x) for x in gapStrings ]
        
    def __processGaps(self, element):
        element = self.__safeRegexSearch(element,'D(\d+\.\d+,\d+\.\d+)')
        return element

    ######################
    def toPickle(self, dir):
        filePath = os.path.join(dir, self.sampleName + ".pkl")
        with open(filePath, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    ##############################
    # Helpers
    def __getLines(self):
        """ Reads pos file and splits lines into three categories,
        adding them to corresponding lists
        """
        with open(self.corePosPath) as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("#") or line.startswith("SCALE"):
                self.headerLines.append(line)
            elif line.startswith("D"): #and "#%gap" in line
                self.gapLines.append(line)
            else:
                self.labelLines.append(line)

    def __safeRegexSearch(self,string,pattern):
        try:
            return re.search(pattern, string).group(1)
        except AttributeError:
            print(f'{self.sampleName}: {pattern} not found in {string}')
            return None

    def __transform_to_xywha(self, box):  # transform into compatible format for detectron2
        xc, yc = (box[0] + box[2]) / 2  # center point
        w = dist(box[0], box[3])  # width
        h = dist(box[0], box[1])  # height

        if box[0][0] >= box[1][0]:
            a = degrees(2 * atan2(h, w))  ## rotation angle in counter-clockwise
        else:
            a = - degrees(2 * atan2(h, w))

        return [xc, yc, w, h, a]
    
    # NOTE: this is wrong. see: https://detectron2.readthedocs.io/en/latest/_modules/detectron2/structures/boxes.html#BoxMode.convert
    # def __transform_to_xywh(self, box):  # transform into compatible format for detectron2 no angle
    #     # determine max outer coords:
    #     xmin = np.min(box[:, 0])
    #     xmax = np.max(box[:, 0])
    #     ymin = np.min(box[:, 1])
    #     ymax = np.max(box[:, 1])

    #     xc = (xmin + xmax) / 2
    #     yc = (ymin + ymax) / 2
    #     w = xmax - xmin  # width
    #     h = ymax - ymin  # height

    #     return [xc, yc, w, h]

    def __transform_to_xyxy(self, box):  # transform into compatible format for detectron2 no angle, equiv to max_bounding_box
        xmin = np.min(box[:, 0])
        xmax = np.max(box[:, 0])
        ymin = np.min(box[:, 1])
        ymax = np.max(box[:, 1])

        return [xmin, ymin, xmax, ymax]

