from distutils.command.build_scripts import first_line_re
import enum
from heapq import merge
from operator import index
from tkinter import Y
import numpy as np
from PIL import Image
from scipy.ndimage import label
import cv2 
from functools import reduce
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import networkx as nx
from collections import Counter
from scipy.stats import mode

from ringdetector.analysis.Edge import Edge

# TODO: why is this here??
def scoreEdges(edges, pointLabels):
    for edge in edges:
        edge.scoreEdge(pointLabels)
    return edges

#####################################################################################
# Main function
def processImg(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)

    #####################
    # Partial application
    __denoise = lambda image : cv2.fastNlMeansDenoising(image,
                                                        None,
                                                        h=10, 
                                                        templateWindowSize=7, 
                                                        searchWindowSize=21)
    __canny = lambda image : cv2.Canny(image, 50, 100)
    __filterShapeLength = lambda shapes : __filterByLength(shapes, minLength=80)
    __hough = lambda shapes : __houghTransform(shapes, img.shape)
    __collectShapes = lambda shapes : __getShapes(__imgArrayFromShapes(shapes, img.shape))
    __filterPoints = lambda shapes : __filterByLength(shapes, minLength=5)
    __magicOutlier = lambda shapes : __magicOutlierDetection(shapes, img.shape)
    __magicLinear = lambda shapes : __magicLinearModel(shapes, img.shape)
    __filterShapes2 = lambda shapes : __filterByLength(shapes, minLength=30)
    __magicRemoveBad = lambda shapes : __magicRemoveBadCandidates(shapes, img)
    __magicSquash = lambda shapes : __magicSquashShape(shapes, img.shape)
    __mergeShapes2 = lambda shapes : __mergeShapes(shapes, ball=(10,10), coeffThreshold=5)
    ######################
    # Function Composition

    # prec 0.457, rec 0.627 on RueL08EE
    # prec 0.816, rec 0.769 on SomA07nn
    # processing = [  
    #                 __denoise,
    #                 __canny,
    #                 __getShapes,
    #                 __magicRightEdge,
    #                 __magicRemoveBad,
    #                 __filterShapes2, minLength=80
    #              ]

    processing = [  
                    __denoise,
                    __canny,
                    __getShapes,
                    # __filterShapeLength,
                    # __cutTails,
                    __magicRightEdge,
                    __magicRemoveBad,
                    __magicSquash, 
                    # __collectShapes,
                    # __collectShapes,    
                    # __mergeShapes,
                    __filterShapes2,
                    __magicOutlier,
                    __mergeShapes2,
                    # __magicLinear,
                    __filterShapeLength,
                 ]
                    # __filterPoints,
                    # __magicOutlier,
                    # __magicSquashUp,
                    # __hough,

    shapes = __composeAndShowIntermediateSteps(processing, img)

    # Cleaner compose for when we don't need the intermediate steps
    # shapes = __compose( __filterByMinEdgeLength,
    #                   __magicRightEdge,
    #                   __filterByMinEdgeLength )(img)

    #TODO: try linking edge instances like in the paper
    #TODO: houghTransform(filteredEdges,cv2.imread(impath))


    return __shapesToEdges(shapes, img.shape)

#####################################################################################
# Processing Helpers
def __magicRemoveBadCandidates(shapes, img):
    return [shape for shape in shapes if not __shouldRemove(shape, img)]

def __shouldRemove(shape, img):
    colorLeft = []
    colorRight = []
    for point in shape:
        if point[1] - 1 > 0:
            left = (point[0],point[1]-1)
            colorLeft.append(img[left])
        if point[1] + 1 < img.shape[1]:
            right = (point[0],point[1]+1)
            colorRight.append(img[right])
    return np.mean(colorLeft) < np.mean(colorRight) # Left side of line is darker

#####################################################################################
# Processing Helpers
def __filterByLength(shapes, minLength):
    return [shape for shape in shapes if len(shape) >= minLength]


#####################################################################################
# Separating points from noise
def __cutTails(shapes):
    shapes = __sortShapes(shapes, tupIndex=1) # sort by y
    cut = [__cutTail(shape) for shape in shapes]
    return [shape for shape in cut if len(shape) != 0]

def __cutTail(shape):
    uniqeY = np.unique([y for (y,_) in shape])
    tenth = int(len(uniqeY)/10)

    topTenth = uniqeY[:tenth]
    topTenthX = [x for (y,x) in shape if y in topTenth]
    topTenthXCut = __mostCommon(topTenthX)
    topTenthSplit = [(y,x) for (y,x) in shape if y in topTenth and x != topTenthXCut]

    bottomTenth = uniqeY[-tenth:]
    bottomTenthX = [x for (y,x) in shape if y in bottomTenth]
    bottomTenthXCut = __mostCommon(bottomTenthX)
    bottomTenthSplit = [(y,x) for (y,x) in shape if y in bottomTenth and x != bottomTenthXCut]
    
    return topTenthSplit + shape[tenth:-tenth] + bottomTenthSplit

def __mostCommon(list):
    mod = mode(list)[0]
    if len(mod) == 0:
        return np.median(list)
    else:
        return mod[0]

def __magicRightEdge(shapes):
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

def __linearRegressionError(shape):
    y = np.array([y for [y,_] in shape])
    X = np.array([x for [_,x] in shape])
    
    model = LinearRegression()
    model.fit(y.reshape(-1, 1),X) # Do the linear regression backward
    pred = model.predict(y.reshape(-1, 1))

    mae = np.mean(np.abs(y-pred))
    return mae

def __shouldRightEdge(shape):
    return __xCounts(shape) > 3 # TODO: if the angle is too steep this is bad

def __xCounts(shape):
    # Find the average number of x values for each y value, if this is too heigh something is wrong with the shape
    yVals = [y for [y,_] in shape]
    yCounts = list(Counter(yVals).values())
    return np.mean(yCounts)


def __magicSquashShape(shapes, size):
    return [__squashShape(shape, size) for shape in shapes]

def __squashShape(shape, size):
    yVals = np.array([y for [y,_] in shape])
    xVals = np.array([x for [_,x] in shape])
    yToXCount = Counter(yVals)
    
    model = LinearRegression()
    model.fit(yVals.reshape(-1, 1),xVals) # Do the linear regression backward

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

    # mean = np.mean(list(yToXCount.values()))
    # threashold = 2*mean
    # toRemove = set([y for (y,xCount) in yToXCount.items() if xCount>=threashold])
    # shape = [point for point in shape if point[0] not in toRemove]
    return newShape


################################################
# Merging
def __mergeShapes(shapes, ball=(10,5), coeffThreshold=2):
    shapes = __sortShapes(shapes)
    coeffs = [__magicLinearModelCoeffsForShape(shape) for shape in shapes]
    
    # Find shapes to merge in a sliding window
    mergePairs = set()
    windowSize = 10
    for i in range(len(shapes)):
        window = list(range(max(0,i-windowSize), min(len(shapes),i+windowSize+1)))
        window.remove(i)
        for j in window:
            if __tipsCloseEnough(shapes[i], shapes[j], ball=ball):
                if abs(coeffs[i] - coeffs[j]) < coeffThreshold:
                    mergePairs.add((min(i,j),max(i,j)))
    mergePairsGraph = nx.Graph(list(mergePairs)) 
    toMerge = [tuple(c) for c in nx.connected_components(mergePairsGraph)] # Disjoint sets of mergePairs == sets of indexes to merge (found with graph library)
    
    # Merge shapes
    mergedShapes = []
    for indexesToMerge in toMerge:
        mergedShape = []
        for j in indexesToMerge:
            mergedShape += shapes[j]
        mergedShapes.append(mergedShape)

    # Append shapes unnaffected by merges | NOTE: comment if you'd like only to see the merged shapes
    for i in range(len(shapes)): 
        if i not in __flattenListOfIndexes(toMerge):
            mergedShapes.append(shapes[i])

    mergedShapes = __sortShapes(mergedShapes)
    return mergedShapes

def __sortShapes(shapes, tupIndex=0): # Default: sort by smallest x values
    for shape in shapes:
            shape.sort(key=lambda tup: tup[1])
    shapes.sort(key=lambda shape: shape[0][1]) 
    return shapes

def __tipsCloseEnough(shape1, shape2, ball=(10,5)):
    # Tips close AND don't start at same height
    return (abs(np.subtract(shape1[-1],shape2[0])) < ball).all() or \
           (abs(np.subtract(shape1[0],shape2[-1])) < ball).all() and \
           not abs(shape1[-1][0]-shape2[-1][0]) < 5 and \
           not abs(shape1[0][0]-shape2[0][0]) < 5

def __flattenListOfIndexes(list):
    return [item for sublist in list for item in sublist]

def __magicLinearModelCoeffsForShape(shape):
    y = np.array([point[0] for point in shape])
    X = np.array([point[1] for point in shape])

    model = LinearRegression()
    model.fit(y.reshape(-1, 1),X) # Do the linear regression backward

    return model.coef_

################################################
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

    # predPoints = list(zip(y,pred)) 
    # return predPoints
    # concat = shape + predPoints #np.concatenate((shape,predPoints), axis=0)
    # return concat

    absDiff = np.abs(model.predict(y.reshape(-1, 1))-X)
    mae = np.mean(absDiff)
    # return mae<5
    keep = absDiff > mae/4
    return [ point for index, point in enumerate(shape) if keep[index]]
    # return [point for index, point in enumerate(shape) if not __isOutlier(index,point,shape)]

################################################
# Merging
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

################################################
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
# Data Format Helpers
def __getShapes(img):
    mask = (img > np.amax(img)/2)
    s = [[1,1,1], [1,1,1], [1,1,1]]          # Neighborhood definition
    labels, numL = label(mask, structure=s)  # Label the contiguous shapes
    shapes = [[] for _ in range(numL+1)]     # Create an array per shape
    [[shapes[labels[i,j]].append((i,j)) for i in range(mask.shape[0])]  # Append index to the shapes array
                                        for j in range(mask.shape[1])]
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

        
def __shapesToEdges(shapes, imShape):
    edges = []
    for shape in shapes:
        edge = Edge(shape, imShape)
        edge.fitPredict(model='linear')
        edges.append(edge)
    return edges


#####################################################################################
# Function Helpers
def __compose(*functions):
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def __composeAndShowIntermediateSteps(functions, img):
    size = img.shape
    stepsIm = deepcopy(img)
    stepsIm = cv2.cvtColor(stepsIm,cv2.COLOR_GRAY2RGB)

    processed = deepcopy(img)
    for func in functions:
        processed = func(processed)

        if len(processed[0])==size[1]: # Img
            toConcat = _imgFromMask(processed)
        else: # List of shapes
            toConcat = __imgArrayFromShapes2(processed, size)

        stepsIm = np.concatenate((stepsIm,toConcat), axis=0)

    Image.fromarray(stepsIm).show("Candidate edges")

    return processed
