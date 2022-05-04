import time
import numpy as np
from PIL import Image
from scipy.ndimage import label
import cv2 
from copy import deepcopy
from sklearn.linear_model import LinearRegression
import networkx as nx
from collections import Counter
from ringdetector.analysis.Ring import Ring


#####################################################################################
#                                 Main Function                                    
#####################################################################################
def findRings(imgPath, readType="grayscale", denoiseH=25, denoiseTemplateWindowSize=10,
                denoiseSearchWindowSize=21, cannyMin=50, cannyMax=75,
                rightEdgeMethod="simple", invertedEdgeWindowSize=25, 
                mergeShapes1Ball=(10,5), mergeShapes1Angle=np.pi/4,
                mergeShapes2Ball=(20,20), mergeShapes2Angle=np.pi/4, 
                filterLengthImgProportion=0.5,
                filterRegressionAnglesAngleThreshold=np.pi/4 ):

    if readType == "grayscale":
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    else:
        raise "Other read types not implemented yet"

    #####################
    # Partial Application
    def __denoise(image,context={}):
        return cv2.fastNlMeansDenoising(image,
                                        None,
                                        h=denoiseH, 
                                        templateWindowSize=denoiseTemplateWindowSize, 
                                        searchWindowSize=denoiseSearchWindowSize), context
    def __contrast_(image, context): 
        return __contrast(image), dict(context, denoisedImage=image)
    def __canny(image, context): 
        return cv2.Canny(image, cannyMin, cannyMax), context
    def __getShapes_(image, context): 
        return __getShapes(image), context
    def __keepRightEdge_(image, context): 
        return __keepRightEdge(image, method=rightEdgeMethod), context
    def __removeInvertedEdges_(shapes, context): 
        return __removeInvertedEdges(shapes,
                                    context['denoisedImage'],
                                    window=invertedEdgeWindowSize), \
                                    context
    def __collectShapes(shapes, context): 
        return __getShapes(__imgArrayFromShapes(shapes, img.shape)), context
    def __mergeShapes_(shapes, context): 
        return __mergeShapes(shapes,
                            ball=mergeShapes1Ball,
                            angleThreshold=mergeShapes1Angle),\
                            context
    def __mergeShapes2_(shapes, context): 
        return __mergeShapes(shapes,
                            ball=mergeShapes2Ball,
                            angleThreshold=mergeShapes2Angle), \
                            context
    def __filterLength(shapes, context): 
        return __filterByLength(shapes, 
                                minLength=img.shape[0]*filterLengthImgProportion), \
                                context
    def __filterRegressionAngles_(shapes, context): 
        return __filterRegressionAngles(shapes,
                                        angleThreshold=filterRegressionAnglesAngleThreshold), \
                                        context
    def __toRings(shapes, _): 
        return __shapesToRings(shapes, img.shape)

    # Unused but still in consideration
    # def __removeAfterCenter_(shapes, context): return __removeAfterCenter(shapes), context

    ######################
    # Function Composition
    processing = [  __denoise,
                    __contrast_,
                    __canny,
                    __getShapes_,
                    __keepRightEdge_, # TODO: doesn't handle shapes with tails on the right side well and shapes with steep angles
                    __removeInvertedEdges_,
                    __collectShapes,
                    __mergeShapes_,
                    __keepRightEdge_,
                    __mergeShapes2_, # TODO: Still misses some? Maybe update merge shapes and try as they have in the paper
                    __keepRightEdge_,
                    __filterLength,
                    __filterRegressionAngles_, # TODO: Dangerous, might not work well if first shape is an anomaly
                    __toRings ]

    # edges = __composeAndShowIntermediateSteps(processing, img)
    edges = __compose(processing)(img)

    return edges


#####################################################################################
#                                Processing functions                                     
#####################################################################################

##########################################
# Contrast
def __contrast(img):
    return cv2.equalizeHist(img)

##########################################
# Inverted edges
#    Remove if left side is darker than right
def __removeInvertedEdges(shapes, denoisedImg, window):
    return [shape for shape in shapes if not __isInverted(shape, denoisedImg, window)]

def __isInverted(shape, img, window):
    inverted = []
    maxX = img.shape[1]
    for (y,x) in shape:
        colorLeft = np.mean(img[y, max(0,x-window-1):max(0,x-1)])
        colorRight = np.mean(img[y, min(maxX,x+1):min(maxX,x+window+1)])
        inverted.append(colorLeft <= colorRight) # Left side of line is darker)
    return np.mean(inverted)>0.5

##########################################
# Right Edge
    # Shapes with noise e.g RueL08EE mostly have noise on the left, hence keep right side of shape
def __keepRightEdge(shapes, method):
    return [__getRightEdge(shape, method=method) for shape in shapes]

def __getRightEdge(shape, method):
    if __shouldRightEdge(shape, method=method):
        rightMost = {}
        for [y,x] in shape:
            if y not in rightMost:
                rightMost[y] = x
            elif x > rightMost[y]:
                rightMost[y] = x
        return list(rightMost.items())
    else:
        return shape

def __shouldRightEdge(shape, method):
    if method=="simple":
        return True
    if method=="angle":
        angle = __getRegressionAngle(shape)
        return angle > np.pi/4 or __magicIsVariabliltyHigh(shape)
    if method=="counts":
        return __xCounts(shape) > 3 # TODO: if the angle is too steep this is bad

def __magicIsVariabliltyHigh(shape):
    y = np.array([point[0] for point in shape])
    X = np.array([point[1] for point in shape])

    model = LinearRegression()
    model.fit(X.reshape(-1, 1),y) # Do the linear regression backward

    absDiff = np.abs(model.predict(X.reshape(-1, 1))-y)
    mae = np.mean(absDiff)
    return mae>5

def __xCounts(shape):
    # Find the average number of x values for each y value, if this is too high something is wrong with the shape
    yVals = [y for [y,_] in shape]
    yCounts = list(Counter(yVals).values())
    return np.mean(yCounts)

################################################
# Merging
def __mergeShapes(shapes, ball=(10,5), angleThreshold=2):
    shapes = __sortShapes(shapes) # Sort by x value
    coeffs = [__linearModelCoeffsForShape(shape) for shape in shapes]
    
    # Find shapes to merge in a sliding window
    mergePairs = set()
    for i in range(len(shapes)):
        for j in __getMergeWindow(i, len(shapes), windowSize=10):
            # Merge condition: tips close and line angles similar
            if __tipsCloseEnough(shapes[i], shapes[j], ball=ball) and \
                abs(np.arctan(coeffs[i]) - np.arctan(coeffs[j])) < angleThreshold:
                    mergePairs.add((min(i,j),max(i,j)))
    mergePairsGraph = nx.Graph(list(mergePairs))
    indexesToMerge = [tuple(c) for c in nx.connected_components(mergePairsGraph)] # Disjoint sets of mergePairs == sets of indexes to merge (found with graph library)
    
    # Merge shapes
    mergedShapes = [[point for i in newShapeIndexes for point in shapes[i]] # Select and flatten shapes
                    for newShapeIndexes in indexesToMerge]

    # Append shapes unnaffected by merges
    mergedIndexes = set(__flattenListOfIndexes(indexesToMerge))
    unMergedIndexes = set(range(len(shapes))) - mergedIndexes
    unMergedShapes = [shapes[i] for i in unMergedIndexes]

    return mergedShapes + unMergedShapes

def __sortShapes(shapes,shape='x',allShapes='x'): # sort by smallest x values for shapes and overall shapes
    # for shape in shapes:
    #         shape.sort(key=lambda tup: tup[1])
    shapeKey = int(shape == 'x') # if x => 1, y => 0
    allShapesKey = int(allShapes == 'x')
    shapes = [sorted(shape, key=lambda tup: tup[shapeKey]) for shape in shapes]
    shapes.sort(key=lambda shape: shape[0][allShapesKey]) 
    return shapes

def __getMergeWindow(i, lenShapes, windowSize=10):
    window = list(range(max(0,i-windowSize), min(lenShapes,i+windowSize+1)))
    window.remove(i)
    return window

def __tipsCloseEnough(shape1, shape2, ball=(10,5)):
    # Tips close AND don't start at same height
    return (abs(np.subtract(shape1[-1],shape2[0])) < ball).all() or \
           (abs(np.subtract(shape1[0],shape2[-1])) < ball).all() and \
           not abs(shape1[-1][0]-shape2[-1][0]) < 5 and \
           not abs(shape1[0][0]-shape2[0][0]) < 5

def __flattenListOfIndexes(list):
    return [item for sublist in list for item in sublist]

def __linearModelCoeffsForShape(shape):
    y = np.array([point[0] for point in shape])
    X = np.array([point[1] for point in shape])

    model = LinearRegression()
    model.fit(y.reshape(-1, 1), X) # Do the linear regression backward

    return model.coef_

##########################################
# Length filter
def __filterByLength(shapes, minLength):
    return [shape for shape in shapes if len(shape) >= minLength]

##########################################
# Filtering Regression Angles
def __filterRegressionAngles(shapes, angleThreshold=np.pi/4):
    # TODO: try to filter based on a large change in angle rather than a change in angle sign
    shapes = __sortShapes(shapes)
    angles = [__getRegressionAngle(shape) for shape in shapes]

    # TODO: this doesn't work if the first element is an anomaly!!!!!!!!
    i = 0
    while i+1 != len(angles):
        if abs(angles[i+1] - angles[i]) > angleThreshold:
            del angles[i+1]
            del shapes[i+1]
            i -= 1
        i += 1
    return shapes

def __getRegressionAngle(shape):
    yVals = np.array([y for [y,_] in shape])
    xVals = np.array([x for [_,x] in shape])
    model = LinearRegression()
    model.fit(yVals.reshape(-1, 1),xVals) # Do the linear regression backward
    angle = np.arctan(model.coef_)
    return angle


#####################################################################################
#                                Data Format Helpers                                     
#####################################################################################
def __getShapes(img):
    mask = (img > np.amax(img)/2)
    s = [[1,1,1], [1,1,1], [1,1,1]]         # Neighborhood definition
    labels, numL = label(mask, structure=s) # Label the contiguous shapes
    shapes = [[] for _ in range(numL+1)]    # Create an array per shape (including 0)
    for i in range(mask.shape[0]):          # Append index to the shapes array
        for j in range(mask.shape[1]):
            shapes[labels[i,j]].append((i,j))
    shapes = shapes[1:] # 0 label => no shape, separate here as 'if' statement in the previous line is much slower 
    return shapes

def __imgArrayFromShapes(shapes, size):
    img = np.zeros(size, dtype=np.uint8)
    for instance in shapes:
        for point in instance:
            img[point] = 255
    return img

def __imgArrayFromShapes2(shapes, size): # TODO: no duplicates
    img = np.zeros((size[0],size[1],3), np.uint8)
    col = 0
    for shape in shapes:
        if col==0:
            col = 1
            color = [255,0,0]
        elif col==1:
            col = 2
            color = [0,255,0]
        else:
            col = 0
            color = [0,0,255]
        for point in shape:
            img[point] = color
    return img

def __imgFromMask(mask):
    img = np.zeros((mask.shape[0],mask.shape[1],3), np.uint8)
    for index, point in np.ndenumerate(mask):
        if point != 0:
            img[index] = [point,point,point]
    return img

def __shapesToRings(shapes, imShape):
    edges = []
    for shape in shapes:
        edge = Ring(shape, imShape)
        edge.fitPredict(model='linear')
        edges.append(edge)
    return edges


#####################################################################################
#                                 Helpers                                      
#####################################################################################
def __compose(functions):
    def composed(*args):
        for func in functions:
            args = func(*args)
        return args    
    return composed

def __composeAndShowIntermediateSteps(functions, img):
    size = img.shape
    stepsIm = deepcopy(img)
    stepsIm = cv2.cvtColor(stepsIm,cv2.COLOR_GRAY2RGB)

    totalTime = 0

    processed = deepcopy(img)
    context = {}
    for func in functions[:-1]:
        start = time.time()
        processed, context = func(processed, context)
        end = time.time()
        print(f'Time for {func.__name__}: {end-start}')
        totalTime += end-start

        if len(processed[0])==size[1]: # Img
            toConcat = __imgFromMask(processed)
        else: # List of shapes
            toConcat = __imgArrayFromShapes2(processed, size)

        stepsIm = np.concatenate((stepsIm,toConcat), axis=0)

    Image.fromarray(stepsIm).show("Candidate edges")

    print(f'Total time: {totalTime}')

    edges = functions[-1](processed, context) # Always end in toRings

    return edges


#####################################################################################
#                                Unused Processing Functions                                     
#####################################################################################

##########################################
# Remove edges after the center
def __removeAfterCenter(shapes):
    shapes = __sortShapes(shapes, shape='y') # Sort by x value
    shapeDistances = [__shapeDistance(shape1, shape2) for shape1, shape2 in zip(shapes, shapes[1:])]
    lastShape = __identifyLastValidShape(shapeDistances)
    return shapes[0:lastShape]

def __shapeDistance(shape1, shape2):
    y1 = set([point[0] for point in shape1])
    y2 = set([point[0] for point in shape2])
    yCommon = y1.union(y2)
    common1 = [point for point in shape1 if point[0] in yCommon]
    common2 = [point for point in shape2 if point[0] in yCommon]
    common1.sort(key=lambda point: point[0])
    common2.sort(key=lambda point: point[0])
    return np.mean([np.abs(point2[1]-point1[1]) for point1,point2 in zip(common1,common2)])

def __identifyLastValidShape(shapeDistances):
    meanDiff = np.mean(shapeDistances)
    stdDiff = np.std(shapeDistances)

    maxDiff = np.max(shapeDistances)
    if maxDiff > meanDiff + 2*stdDiff:
        return np.argmax(shapeDistances) + 1
    else:
        return len(shapeDistances) + 1
