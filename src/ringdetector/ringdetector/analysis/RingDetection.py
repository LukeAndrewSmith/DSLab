import time
import numpy as np
from PIL import Image
from scipy.ndimage import label
import cv2 
from copy import deepcopy
from sklearn.linear_model import LinearRegression
import networkx as nx
from collections import Counter
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from ringdetector.analysis.Ring import Ring


#####################################################################################
#                                 Main Function                                    
#####################################################################################
def findRings(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

    #####################
    # Partial Application
    def __denoise(image,context={}): return cv2.fastNlMeansDenoising(image,
                                                    None,
                                                    h=25, 
                                                    templateWindowSize=10, 
                                                    searchWindowSize=21), context
    def __contrast_(image, context): return __contrast(image), dict(context, denoisedImage=image)
    def __canny(image, context): return cv2.Canny(image, 50, 75), context
    def __getShapes_(image, context): return __getShapes(image), context
    def __keepRightEdge_(image, context): return __keepRightEdge(image), context
    def __removeInvertedEdges_(shapes, context): return __removeInvertedEdges(shapes, context['denoisedImage'], window=25), context
    def __squashShapes_(shapes, context): return __squashShapes(shapes, img.shape), context
    def __collectShapes(shapes, context): return __getShapes(__imgArrayFromShapes(shapes, img.shape)), context
    def __mergeShapes_(shapes, context): return __mergeShapes(shapes, ball=(10,5), angleThreshold=np.pi/4), context
    def __mergeShapes2_(shapes, context): return __mergeShapes(shapes, ball=(20,20), angleThreshold=np.pi/4), context
    def __filterLength(shapes, context): return __filterByLength(shapes, minLength=img.shape[0]*(1/2)), context
    def __filterRegressionAngles_(shapes, context): return __filterRegressionAngles(shapes), context
    def __removeAfterCenter_(shapes, context): return __removeAfterCenter(shapes), context
    def __toRings(shapes, context): return __shapesToRings(shapes, img.shape)

    # Attempt at the papers implementation
    len = 5
    kernel1 = np.array([-1]*len+[0]+[1]*len) # use cv2.CV_8U and negative on left side to only keep
    def __myKernel(image): return cv2.filter2D(src=image, ddepth=cv2.CV_8U, kernel=kernel1)
    def __peaks(image): return __findPeaks(image)

    # Unused: left here to give ideas to others who want to try this
    # __mergeShapes2 = lambda shapes : __mergeShapes(shapes, ball=(10,10), coeffThreshold=10) # NOTE: example reusing a func with different parameters, it does currently improve performace if before __filterRegressionAngles but not by enough to warrant keeping for now 
    # __magicOutlier = lambda shapes : __magicOutlierDetection(shapes, img.shape) # NOTE: maybe some form of this might be useful
    # __hough = lambda shapes : __houghTransform(shapes, img.shape)
    # __magicLinear = lambda shapes : __magicLinearModel(shapes, img.shape) # NOTE: use this to plot the linear fits on top of the shapes to get an visual for what we're predicting

    ######################
    # Function Composition
    # TODO:
        # Create a smallish dataset containing all different types of images so as to
        # make it easy to update pipeline then run a regression test
        # Optimise: code could probably be faster

    # NOTE: attempt at as described in the paper, not put too much work into it yet:
    # processing = [  __denoise,
    #                 __myKernel, # TODO: currently misses some obvious ones in the lightly coloured images
    #                 __threshold,
    #                 __peaks,
    #                 __TODO,...
    #                 __toRings ] # NOTE: Always end in __toRings

    # NOTE: simple base for people to work upon:
    # processing = [  __denoise,
    #                 __canny, # TODO: currently misses some obvious ones in the lightly coloured images
    #                 __getShapes,
    #                 __squashShapes_,
    #                 __removeInvertedEdges_,
    #                 __filterLength,
    #                 __filterRegressionAngles, # TODO: currently removes too much, removes edges that are very slightly angled in the opposite direction to the mean angle direction (e.g in SomA07nn)
    #                 __toRings ] # NOTE: Always end in __toRings

    # NOTE: current best performance:
    # processing = [  __denoise,
    #                 __canny, # TODO: currently misses some obvious ones in the lightly coloured images
    #                 __getShapes,
    #                 __keepRightEdge, # TODO: doesn't handle shapes with tails well
    #                 __removeInvertedEdges_,
    #                 __squashShapes_,
    #                 __collectShapes,
    #                 __mergeShapes_,
    #                 __squashShapes_,
    #                 __filterLength,
    #                 __filterRegressionAngles, # TODO: currently removes too much, removes edges that are very slightly angled in the opposite direction to the mean angle direction (e.g in soml17ww)
    #                 __toRings ]

    # TODO: new pipeline
        # Try leaving in more shapes and filtering based on regression angles etc after

    # Best so far (I beleive, not sure tbh)
    # processing = [  __denoise, # TODO: Play with the parameters
    #                 __canny, # TODO: currently misses some obvious ones in the lightly coloured images, play with contrast
    #                 __getShapes,
    #                 __keepRightEdge, # TODO: doesn't handle shapes with tails on the right side well
    #                 __removeInvertedEdges_, # TODO: In this function we could also detect gaps
    #                 __collectShapes,
    #                 __mergeShapes_,
    #                 __filterLength,
    #                 __toRings ]

    processing = [  __denoise, # TODO: Play with the parameters
                    __contrast_,
                    __canny,
                    __getShapes_,
                    __keepRightEdge_, # TODO: doesn't handle shapes with tails on the right side well and shapes with steep angles
                    __removeInvertedEdges_, # TODO: In this function we could also detect gaps
                    __collectShapes,
                    __mergeShapes_,
                    __keepRightEdge_,
                    __mergeShapes2_, # TODO: Still misses some?
                    __keepRightEdge_,
                    __filterLength,
                    __filterRegressionAngles_, # TODO: Dangerous, might not work well if first shape is an anomaly
                    __removeAfterCenter_, # TODO
                    __toRings ]

    edges = __composeAndShowIntermediateSteps(processing, img)
    # edges = __compose(processing)(img)

    return edges




#####################################################################################
#                                Processing functions                                     
#####################################################################################

##########################################
# Inverted edges
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
# Length filter
def __filterByLength(shapes, minLength):
    return [shape for shape in shapes if len(shape) >= minLength]

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


##########################################
# Right Edge
    # Shapes with noise e.g RueL08EE mostly have noise on the left, hence keep right side of shape
def __keepRightEdge(shapes):
    return [__getRightEdge(shape) for shape in shapes]

def __getRightEdge(shape):
    if __shouldRightEdge(shape):
        rightMost = {}
        for [y,x] in shape:
            if y not in rightMost:
                rightMost[y] = x
            elif x > rightMost[y]:
                rightMost[y] = x
        return list(rightMost.items())
    else:
        return shape

def __shouldRightEdge(shape):
    return True

    # angle = __getRegressionAngle(shape)
    # return angle > np.pi/4 or __magicIsVariabliltyHigh(shape)

    # from squashShapes
    # if angle < np.pi/4:
    #     newShape = __squashHorizontal(shape, yVals, xVals, model, yToXCount, size)
    # else:
    #     newShape = __squashVertical(shape, yVals, xVals, size)

    # return __xCounts(shape) > 3 # TODO: if the angle is too steep this is bad

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

#####################################################################################
# Shape squashing
    # Make every shape have either:
    #   * 1 y value per x 
    #   * 1 x value per y
    # depending on the angle of the shape
def __squashShapes(shapes, size):
    return [__squashShape(shape, size) for shape in shapes]

def __squashShape(shape, size):
    yVals = np.array([y for [y,_] in shape])
    xVals = np.array([x for [_,x] in shape])
    yToXCount = Counter(yVals)
    model = LinearRegression()
    model.fit(yVals.reshape(-1, 1),xVals) # Do the linear regression backward (shapes are steep so there are often multiple y values for any given x, hence predicting this way round is more detailed)
    angle = abs(np.arctan(model.coef_))

    if angle < np.pi/4:
        newShape = __squashHorizontal(shape, yVals, xVals, model, yToXCount, size)
    else:
        newShape = __squashVertical(shape, yVals, xVals, size)

    return newShape

def __squashHorizontal(shape, yVals, xVals, model, yToXCount, size):
    newShape = []
    for i in range(len(yVals)):
        if yToXCount[yVals[i]] > 1:
            pred = int(model.predict([[yVals[i]]])[0])
            pred = max(0,min(size[1]-1,pred))
            xValsForY = [x for (y,x) in shape if y==yVals[i]]
            closest = min(xValsForY, key=lambda x:abs(x-pred))
            newShape.append((yVals[i],closest))
        else:
            newShape.append((yVals[i],xVals[i]))
    return newShape

def __squashVertical(shape, yVals, xVals, size):
    xToYCount = Counter(xVals)
    model = LinearRegression()
    model.fit(xVals.reshape(-1, 1),yVals) # Do the linear regression forward

    newShape = []
    for i in range(len(xVals)):
        if xToYCount[xVals[i]] > 1:
            pred = int(model.predict([[xVals[i]]])[0])
            pred = max(0,min(size[1]-1,pred))
            yValsForX = [y for (y,x) in shape if x==xVals[i]]
            closest = min(yValsForX, key=lambda x:abs(x-pred))
            newShape.append((closest,xVals[i]))
        else:
            newShape.append((yVals[i],xVals[i]))
    return newShape

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
# Filtering
def __filterRegressionAngles(shapes):
    # TODO: try to filter based on a large change in angle rather than a change in angle sign
    shapes = __sortShapes(shapes)
    angles = [__getRegressionAngle(shape) for shape in shapes]
    # plt.plot(angles)
    # plt.show()

    # TODO: this doesn't work if the first element is an anomaly!!!!!!!!
    # window = 5
    finished = False
    i = 0
    while not finished:
        # if i+window == len(angles):
        #     finished = True
        #     continue
        # sequence = angles[i:i+window]
        # mean = np.mean(sequence)
        # std = np.std(sequence)
        # # [np.abs(angle-)]

        if i+1 == len(angles):
            finished = True
            continue
        
        if abs(angles[i+1] - angles[i]) > np.pi/4:
            del angles[i+1]
            del shapes[i+1]
            i -= 1

        i += 1

    # plt.plot(angles)
    # plt.show()

    anomalies = __findAngleAnomalies(angles)
    return shapes

def __getRegressionAngle(shape):
    yVals = np.array([y for [y,_] in shape])
    xVals = np.array([x for [_,x] in shape])
    model = LinearRegression()
    model.fit(yVals.reshape(-1, 1),xVals) # Do the linear regression backward
    angle = np.arctan(model.coef_)
    return angle

def __findAngleAnomalies(angles):
    angleDiffs = np.abs(np.array(angles[1:]) - np.array(angles[:-1]))
    return [diff > np.pi/4 for diff in angleDiffs]


#####################################################################################
#                                  The Paper's approach                                      
#####################################################################################


##########################################
# Peaks
def __findPeaks(image):
    return np.array([__findColPeaks(col) for col in image])

def __findColPeaks(col):
    peakIndexes, _ = find_peaks(col)
    mask = np.zeros(col.shape, int)
    mask[peakIndexes] = 1
    return np.multiply(mask,col)

def __threshold(image):
    image[image < 255/2] = 0
    return image




#####################################################################################
#                                Unused Processing Functions                                     
#####################################################################################

##########################################
# Outliers
def __magicOutlierDetection(shapes, size):
    return [shape for shape in shapes if __magicOutlierDetectionForShape(shape, size)]

def __magicOutlierDetectionForShape(shape, size):
    y = np.array([point[0] for point in shape])
    X = np.array([point[1] for point in shape])

    model = LinearRegression()
    model.fit(y.reshape(-1, 1),X) # Do the linear regression backward

    pred = model.predict(y.reshape(-1, 1)).astype(int)
    pred = [max(0,min(size[1]-1,point)) for point in pred]

    absDiff = np.abs(model.predict(y.reshape(-1, 1))-X)
    mae = np.mean(absDiff)
    return mae<5

def __magicOutlierDetection2(shapes, size):
    return [__magicOutlierDetectionForShape(shape, size) for shape in shapes]

def __magicOutlierDetectionForShape2(shape, size):
    y = np.array([point[0] for point in shape])
    X = np.array([point[1] for point in shape])

    model = LinearRegression()
    model.fit(y.reshape(-1, 1),X) # Do the linear regression backward

    pred = model.predict(y.reshape(-1, 1)).astype(int)
    pred = [max(0,min(size[1]-1,point)) for point in pred]

    absDiff = np.abs(model.predict(y.reshape(-1, 1))-X)
    mae = np.mean(absDiff)
    keep = absDiff > mae/4
    return [ point for index, point in enumerate(shape) if keep[index]]

##########################################
# Linear Model
def __magicLinearModel(shapes, size):
    return [__magicLinearModelForShape(shape, size) for shape in shapes]

def __magicLinearModelForShape(shape, size):
    y = np.array([point[0] for point in shape])
    X = np.array([point[1] for point in shape])

    model = LinearRegression()
    model.fit(y.reshape(-1, 1),X) # Do the linear regression backward, I want to plot a point per y coordinate and this is the easiest way as i render the y coords unique in shape through magicRight

    pred = model.predict(y.reshape(-1, 1)).astype(int)
    pred = [max(0,min(size[1]-1,point)) for point in pred]

    predPoints = list(zip(y,pred))
    return predPoints + shape

##########################################
# Hough
def __houghTransform(shapes, size):
    img = __imgArrayFromShapes(shapes, size)
    Image.fromarray(img).show("Candidate edges")
    lines = cv2.HoughLines(img,1,np.pi/180,20)
    for line in lines:
        for rho,theta in line:
            if ( theta < np.pi/4 and theta > 2*np.pi - np.pi/4 ) or \
            ( theta < np.pi + np.pi/4 and theta > np.pi - np.pi/4 ):
                # print(theta)
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(img,(x1,y1),(x2,y2),255,1)
    Image.fromarray(img).show("Candidate edges")
    return __getShapes(img)




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

def _imgFromMask(mask):
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
            toConcat = _imgFromMask(processed)
        else: # List of shapes
            toConcat = __imgArrayFromShapes2(processed, size)

        stepsIm = np.concatenate((stepsIm,toConcat), axis=0)

    Image.fromarray(stepsIm).show("Candidate edges")

    print(f'Total time: {totalTime}')

    edges = functions[-1](processed, context) # Always end in toRings

    return edges
