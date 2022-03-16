from operator import truediv
from preprocessing.geometry import min_bounding_rectangle
import os
import re

class Core_Annotation:
    def __init__(self, labelmeAnnotations, sampleName, pointLabelDirectoryPath, imagePath):
        self.sampleName  = sampleName
        self.pointLabelDirectoryPath = pointLabelDirectoryPath
        self.imagePath = imagePath
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
        return next((shape['points'] for shape in self.labelmeAnnotations['shapes'] \
                                     if shape["label"] == f'{self.sampleName}_{label}'), default)


    ##############################
    def _initPointLabelInfo(self):
        lines = self._tryReadLines(f'{self.sampleName}.pos')
        # Some lines return multiple values (pith), hence return all lines in an array and unpack
        pointLabelInfoDict = dict([ i for x in lines for i in self._processInfoLine(x) \
                                                      if not self._isPositionLine(x) ])
        return self._unpackPointLabelInfoDict(pointLabelInfoDict)
        
    def _processInfoLine(self, line):
        if '#DPI' in line:
            return [('dpi', float(self._safeRegexSearch(line, '#DPI (\d+\.\d+)')))]
        if 'Pith' in line:
            pithCoordinates = float(self._safeRegexSearch(line, 'PithCoordinates=(\d+\.\d+)'))
            distanceToPith = float(self._safeRegexSearch(line, 'DistanceToPith=(\d+\.\d+)'))
            yearsToPith = float(self._safeRegexSearch(line, 'YearsToPith=(\d+)'))
            return [('pithCoordinates', pithCoordinates), ('distanceToPith', distanceToPith), ('yearsToPith', yearsToPith)]
        else:
            return [('',None)]

    def _unpackPointLabelInfoDict(self, d):
        return [ d.get('dpi'), d.get('pithCoordinates'), d.get('distanceToPith'), d.get('yearsToPith') ]


    ##############################
    def _initPointLabels(self):
        lines = self._tryReadLines(f'{self.sampleName}.pos')
        pointLabels = [ self._processPositionLine(x) for x in lines  \
                                              if self._isPositionLine(x) 
                                              and not self._isGapLine(x) ]
        return pointLabels

    def _processPositionLine(self, line):
        lineSplit = line.split(' ')
        lineSplit = [ x for x in lineSplit if ( x not in ['','\n','.'] ) ]
        return [ self._positionStringToFloatArray(x) for x in lineSplit ]
 
    def _positionStringToFloatArray(self, positionString):
        positionStringSplit = positionString.split(',')
        return [float(x) for x in positionStringSplit]


    ##############################
    def _initGapLabels(self):
        lines = self._tryReadLines(f'{self.sampleName}.pos')
        gapLabels = [ self._processGapPositionLine(x) for x in lines  \
                                                 if self._isPositionLine(x) ]
        gapLabels = [ x for x in gapLabels if ( len(x) != 0 )] # Empty array returned if no gap, hence needs to be removed, TODO: maybe maket this cleaner
        return gapLabels      

    def _processGapPositionLine(self, line):
        lineSplit = line.split(' ')
        lineSplit = [ self._processGaps(x) for x in lineSplit \
                                           if ( self._isGapLine(x)) ] # Remove unwanted elements and only process gaps
        return [ self._positionStringToFloatArray(x) for x in lineSplit ]
        
    def _processGaps(self, element):
        element = self._safeRegexSearch(element,'D(\d+\.\d+,\d+\.\d+)')
        return element

    def _isGapLine(self, label):
        # Gaps are labeled as:   D138.112,17.716 #%gap\n
        # NOTE: #%gap\n already removed by _unwantedElems 
        if ( 'D' in label ): return True
        return False


    ##############################
    # Helpers
    def _isPositionLine(self, line):
        return not line[0] in ["#", "S"]
    
    def _tryReadLines(self, file):
        f = None
        try:
            f = open(os.path.join(self.pointLabelDirectoryPath, file))
            lines = f.readlines()
        except:
            print(f'Cannot open { file }')
            lines = []
        finally:
            if f is not None:
                f.close()
        return lines

    def _safeRegexSearch(self,string,pattern):
        try:
            return re.search(pattern, string).group(1)
        except:
            print(f'Error: { pattern } not found in { string }')
            return None