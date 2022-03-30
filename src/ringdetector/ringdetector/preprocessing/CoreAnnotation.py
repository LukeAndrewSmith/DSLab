from operator import truediv
import os
import re
import logging

from ringdetector.preprocessing.GeometryUtils import min_bounding_rectangle

class CoreAnnotation:
    def __init__(self, labelmeAnnotations, sampleName, corePosPath, imagePath):
        self.sampleName  = sampleName
        self.corePosPath = corePosPath
        self.imagePath = imagePath
        self.imageName = os.path.basename(self.imagePath)
        self.labelmeAnnotations = labelmeAnnotations

        # Shapes: [ [x,y], ... ]
            # NOTE: do not modify self.shape or self.rectangles, this 
            # wont modify the original variables
        self.innerRectangle  = self.__initRectangle("inner")
        self.outerRectangle  = self.__initRectangle("outer")
        self.rectangles      = [self.innerRectangle, self.outerRectangle]
        self.cracks = self.__initCracks()
        self.bark   = self.__initBark()
        self.ctrmid = self.__initCtrmid()
        self.ctrend = self.__initCtrend()
        self.shapes = [self.cracks, self.bark, self.ctrmid, self.ctrend]
        
        self.tricky   = self.__initTricky()
        
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
        self.dpi, self.mmPith, self.mmDistToPith, self.yearsToPith = self.__initPointLabelInfo()

        self.pith = []
        self.distToPith = None

    def __repr__(self) -> str:
        return (f"CoreAnnotation for {self.sampleName} in "
                f"{self.imagePath}")

    ######################
    # labelme Annotations
    def __initRectangle(self, type):
        points = self.__findShape(type, [])
        boundingRect = min_bounding_rectangle(points)
        return boundingRect

    def __initCracks(self):
        return self.__findShape('crack', [])

    def __initBark(self):
        return self.__findShape('bark', [])

    def __initCtrmid(self):
        return self.__findShape('ctrmid', [])

    def __initCtrend(self):
        return self.__findShape('ctrend', [])

    def __initTricky(self):
        if self.__findShape('tricky', False): return True
        return False

    def __findShape(self, label, default):
        return next(
            (shape['points'] for shape in self.labelmeAnnotations['shapes'] \
            if shape["label"] == f'{self.sampleName}_{label}'), default)


    ##############################
    def __initPointLabelInfo(self):
        # Some lines return multiple values (pith), hence return all lines in an array and unpack
        pointLabelInfoDict = dict(
            [ i for x in self.headerLines for i in self.__processInfoLine(x)]
        )
        return self.__unpackPointLabelInfoDict(pointLabelInfoDict)
        
    def __processInfoLine(self, line):
        if '#DPI' in line:
            return [('dpi', float(self.__safeRegexSearch(line, '#DPI (\d+\.\d+)')))]
        if 'Pith' in line:
            pithCoordinates = self.__positionStringToFloatArray(
                self.__safeRegexSearch(
                    line, 'PithCoordinates=(\d+\.\d+,\d+\.\d+)')
            )
            # TODO: clean this up, doing quick cuz some pith lines have no years)
            distanceToPith = self.__getPithNumbers(line, 'DistanceToPith=(\d+\.\d+)')
            yearsToPith = self.__getPithNumbers(line, 'YearsToPith=(\d+)')
            return [('pithCoordinates', pithCoordinates), ('distanceToPith', distanceToPith), ('yearsToPith', yearsToPith)]
        else:
            return [('',None)]

    def __getPithNumbers(self, line, regex):
        regexStr = self.__safeRegexSearch(line, regex)
        floatResult = None
        if regexStr: 
            floatResult = float(regexStr)
        return floatResult

    def __unpackPointLabelInfoDict(self, d):
        return [ d.get('dpi'), d.get('pithCoordinates'), d.get('distanceToPith'), d.get('yearsToPith') ]


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
        # Empty array returned if no gap, hence needs to be removed, 
        # TODO: maybe maket this cleaner
        # clem: not sure what you're wanting here, empty list seems fine to me
        return gapLabels      

    def __processGapPositionLine(self, line):
        lineSplit = line.split(' ')
        gapStrings = [ self.__processGaps(x) for x in lineSplit \
            if x.startswith("D") and "#%gap" not in x]
        return [ self.__positionStringToFloatArray(x) for x in gapStrings ]
        
    def __processGaps(self, element):
        element = self.__safeRegexSearch(element,'D(\d+\.\d+,\d+\.\d+)')
        return element

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
        except:
            print(f'Error: { pattern } not found in { string }')
            return None