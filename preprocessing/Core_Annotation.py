import os
import re

class Core_Annotation:
    def __init__(self, labelmeAnnotations, sampleName, pointLabelDirectoryPath):
        self.sampleName  = sampleName
        self.pointLabelDirectoryPath = pointLabelDirectoryPath
        self.labelmeAnnotations = labelmeAnnotations

        # Shapes: [ [x,y], ... ]
        self.innerBound  = self.initInner()
        self.outerBound  = self.initOuter()
        self.cracks      = self.initCracks()
        self.bark        = self.initBark()
        self.ctrmid      = self.initCtrmid()
        self.ctrend      = self.initCtrend()
        self.shapes      = [self.innerBound, self.outerBound, self.cracks, self.bark, self.ctrmid, self.ctrend]
        
        self.is_tricky   = self.initTricky()
        
        # Point Labels: [ [ [x,y], ... ], ... ]
        self.pointLabels = self._initPointLabels()

        # Point Label Info:
        self.dpi, self.pith, self.distToPith, self.yearsToPith = self._initPointLabelInfo()

    ######################
    # labelme Annotations
    def initInner(self):
        inner = self._findShape('inner', None)
        innerBox = inner # TODO: translate to bounding box
        return innerBox

    def initOuter(self):
        outer = self._findShape('outer', None)
        outerBox = outer # TODO: translate to bounding box
        return outerBox

    def initCracks(self):
        return self._findShape('crack', None)

    def initBark(self):
        return self._findShape('bark', None)

    def initCtrmid(self):
        return self._findShape('ctrmid', None)

    def initCtrend(self):
        return self._findShape('ctrend', None)

    def initTricky(self):
        return self._findShape('tricky', False)

    def _findShape(self, label, default):
        return next((shape['points'] for shape in self.labelmeAnnotations['shapes'] \
                                     if shape["label"] == f'{self.sampleName}_{label}'), default)


    ##############################
    def _initPointLabelInfo(self):
        lines = self._tryReadLines(f'{self.sampleName}.pos')
        # Some lines return multiple values, hence return all lines in an array and unpack
        pointLabelInfoDict = dict([ i for x in lines for i in self._processInfoLine(x) \
                                                      if not self._isPositionLine(x) ])
        return self._unpackPointLabelInfoDict(pointLabelInfoDict)
        
    def _processInfoLine(self, line):
        if '#DPI' in line:
            return [('dpi', self._safeRegexSearch(line, '#DPI (\d+\.\d+)'))]
        if 'Pith' in line:
            pithCoordinates = self._safeRegexSearch(line, 'PithCoordinates=(\d+\.\d+)')
            distanceToPith = self._safeRegexSearch(line, 'DistanceToPith=(\d+\.\d+)')
            yearsToPith = self._safeRegexSearch(line, 'YearsToPith=(\d+)')
            return [('pithCoordinates', pithCoordinates), ('distanceToPith', distanceToPith), ('yearsToPith', yearsToPith)]
        else:
            return [('',None)]

    def _unpackPointLabelInfoDict(self, d):
        return [ d.get('dpi'), d.get('pithCoordinates'), d.get('distanceToPith'), d.get('yearsToPith') ]


    ##############################
    def _initPointLabels(self):
        lines = self._tryReadLines(f'{self.sampleName}.pos')
        return [ self._processPositionLine(x) for x in lines  \
                                              if self._isPositionLine(x) ]

    def _processPositionLine(self, line):
        lineSplit = line.split(' ')
        lineSplit = [ x for x in lineSplit if ( x not in ['','\n','.'] ) ] # Remove unwanted elements
        return [ self._positionStringToFloatTuple(x) for x in lineSplit ]
 
    def _positionStringToFloatTuple(self, positionString):
        positionStringSplit = positionString.split(',')
        return tuple([float(x) for x in positionStringSplit])


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