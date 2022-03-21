from operator import truediv
import os
import re
import logging

from preprocessing.geometry import min_bounding_rectangle

class CoreAnnotation:
    def __init__(self, labelmeAnnotations, sampleName, corePosPath, imagePath):
        self.sampleName  = sampleName
        self.corePosPath = corePosPath
        self.imagePath = imagePath
        self.imageName = os.path.basename(self.imagePath)
        self.labelmeAnnotations = labelmeAnnotations

        # Shapes: [ [x,y], ... ]
            # TODO: double check that the following is true...
            # NOTE: Rectangle points are stored in order: clockwise from the bottom left:
            # 1 ----- 2
            # |       |
            # 0 ----- 3
            # NOTE: do not modify self.shape or self.rectangles, this wont modify the original variables
        self.innerRectangle  = self._initInnerRectangle()
        self.outerRectangle  = self._initOuterRectangle()
        self.rectangles      = [self.innerRectangle, self.outerRectangle]
        self.cracks = self._initCracks()
        self.bark   = self._initBark()
        self.ctrmid = self._initCtrmid()
        self.ctrend = self._initCtrend()
        self.shapes = [self.cracks, self.bark, self.ctrmid, self.ctrend]
        
        self.is_tricky   = self._initTricky()
        
        # Parsing the pos file, splitting lines into groups
        self.header_lines = []
        self.label_lines = []
        self.gap_lines = []
        self._getLines()

        # Point/Gap Labels: [ [ [x,y], ... ], ... ]
        self.pointLabels = self._initPointLabels()
        self.gapLabels = self._initGapLabels()

        # Point Label Info:
        self.dpi, self.pith, self.distToPith, self.yearsToPith = self._initPointLabelInfo()

    ######################
    # labelme Annotations
    def _initInnerRectangle(self):
        inner = self._findShape('inner', None)
        innerBox = min_bounding_rectangle(inner)
        return innerBox

    def _initOuterRectangle(self):
        outer = self._findShape('outer', None)
        outerBox = outer # TODO: translate to bounding box
        # TODO: Handle having more points than expected in the rectange, the first example has 5
        outerBox = outer[0:4]
        return outerBox

    def _initCracks(self):
        return self._findShape('crack', None)

    def _initBark(self):
        return self._findShape('bark', None)

    def _initCtrmid(self):
        return self._findShape('ctrmid', None)

    def _initCtrend(self):
        return self._findShape('ctrend', None)

    def _initTricky(self):
        if self._findShape('tricky', False): return True
        return False

    def _findShape(self, label, default):
        return next(
            (shape['points'] for shape in self.labelmeAnnotations['shapes'] \
            if shape["label"] == f'{self.sampleName}_{label}'), default)


    ##############################
    def _initPointLabelInfo(self):
        # Some lines return multiple values (pith), hence return all lines in an array and unpack
        pointLabelInfoDict = dict(
            [ i for x in self.header_lines for i in self._processInfoLine(x)]
        )
        return self._unpackPointLabelInfoDict(pointLabelInfoDict)
        
    def _processInfoLine(self, line):
        if '#DPI' in line:
            return [('dpi', float(self._safeRegexSearch(line, '#DPI (\d+\.\d+)')))]
        if 'Pith' in line:
            pithCoordinates = self._positionStringToFloatArray(
                self._safeRegexSearch(
                    line, 'PithCoordinates=(\d+\.\d+,\d+\.\d+)')
            )
            distanceToPith = float(self._safeRegexSearch(line, 'DistanceToPith=(\d+\.\d+)'))
            yearsToPith = float(self._safeRegexSearch(line, 'YearsToPith=(\d+)'))
            return [('pithCoordinates', pithCoordinates), ('distanceToPith', distanceToPith), ('yearsToPith', yearsToPith)]
        else:
            return [('',None)]

    def _unpackPointLabelInfoDict(self, d):
        return [ d.get('dpi'), d.get('pithCoordinates'), d.get('distanceToPith'), d.get('yearsToPith') ]


    ##############################
    def _initPointLabels(self):
        pointLabels = [self._processPositionLine(x) for x in self.label_lines]
        return pointLabels

    def _processPositionLine(self, line):
        lineSplit = line.split(' ')
        positionStrings = [ 
            self._positionStringToFloatArray(x) for x in lineSplit \
                if x not in ['','\n','.']
        ]
        return positionStrings
 
    def _positionStringToFloatArray(self, positionString):
        positionStringSplit = positionString.split(',')
        return [float(x) for x in positionStringSplit]


    ##############################
    def _initGapLabels(self):
        gapLabels = [ 
            self._processGapPositionLine(x) for x in self.gap_lines
        ]
        gapLabels = [ x for x in gapLabels if ( len(x) != 0 )] 
        # Empty array returned if no gap, hence needs to be removed, 
        # TODO: maybe maket this cleaner
        return gapLabels      

    def _processGapPositionLine(self, line):
        lineSplit = line.split(' ')
        gapStrings = [ self._processGaps(x) for x in lineSplit \
            if x.startswith("D") and "#%gap" not in x]
        return [ self._positionStringToFloatArray(x) for x in gapStrings ]
        
    def _processGaps(self, element):
        element = self._safeRegexSearch(element,'D(\d+\.\d+,\d+\.\d+)')
        return element

    ##############################
    # Helpers
    def _getLines(self):
        """ Reads pos file and splits lines into three categories,
        adding them to corresponding lists
        """
        with open(self.corePosPath) as f:
            lines = f.readlines()
        for line in lines:
            if line.startswith("#") or line.startswith("SCALE"):
                self.header_lines.append(line)
            elif line.startswith("D"): #and "#%gap" in line
                self.gap_lines.append(line)
            else:
                self.label_lines.append(line)

    def _safeRegexSearch(self,string,pattern):
        try:
            return re.search(pattern, string).group(1)
        except:
            print(f'Error: { pattern } not found in { string }')
            return None