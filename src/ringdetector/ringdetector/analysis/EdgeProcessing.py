import numpy as np
from PIL import Image
from tqdm import trange, tqdm
import time
from scipy.ndimage import label
import cv2 

from ringdetector.analysis.Edge import Edge

def getEdges(candidateEdgesImg, minEdgeLen, edgeModel):
    '''
        Gets an candidateEdges (grascale image) from one of the Edge detection
        algos and postprocesses to identify edges
    '''
    def __getEdges(candidateEdgesImg, minEdgeLen, edgeModel):
        shapes = __getShapes(candidateEdgesImg)
        return identifyEdges(
            shapes, candidateEdgesImg.shape, minEdgeLen, edgeModel
        )

    def __getShapes(candidateEdgesMask):
        # NOTE: the output of shapes has inverted the x and y axes of the image
        # s.t. each point has (y,x) as its coordinate. These are inverted
        # back to normal in Edge init.
        candidateEdgesMask = (candidateEdgesImg > np.amax(candidateEdgesImg)/2)
        # Neighborhood definition
        s = [[1,1,1], [1,1,1], [1,1,1]]  
        # Label the contiguous shapes
        labels, numL = label(candidateEdgesMask, structure=s) 
        # Create an array per shape
        shapes = [[] for _ in range(numL+1)]
        # Append index to the shapes array
        [[shapes[labels[i,j]].append((i,j)) for i in range(candidateEdgesMask.shape[0])] 
                                            for j in range(candidateEdgesMask.shape[1])]
        # 0 label => no shape, didn't but an 'if' in the previous line 
        # as it's much slower 
        shapes = shapes[1:]
        return shapes

    def identifyEdges(shapes, maskShape, minEdgeLen, edgeModel):
        #TODO: try linking edge instances like in the paper
        filteredShapes = __filterByLength(shapes, minEdgeLen)
        edges = []
        for shape in filteredShapes:
            edge = Edge(shape, maskShape)
            edge.fitPredict(model=edgeModel)
            edges.append(edge)
        return edges

    def __filterByLength(shapes, minLength):
        filteredShapes = [ instance for instance in shapes 
                                                 if len(instance) >= minLength]
        return filteredShapes

    return __getEdges(candidateEdgesImg, minEdgeLen, edgeModel)


def scoreEdges(edges, pointLabels):
    for edge in edges:
        edge.scoreEdge(pointLabels)
    return edges

# Testing:
def houghTransform(candidateEdgesImg):
    lines = cv2.HoughLines(candidateEdgesImg,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(candidateEdgesImg,(x1,y1),(x2,y2),(0,0,255),2)
    return candidateEdgesImg

# ###########################################################
# ### Plotting functions
def imageFromEdgeInstances(edges, dim1, dim2):
    im = np.zeros((dim1, dim2), dtype=np.uint8)
    for instance in edges:
        for point in instance.edge:
            im[point] = 255
    return im

def saveEdgeInstanceImage(path, edges, dim1, dim2):
    im = imageFromEdgeInstances(edges, dim1, dim2)
    gradImage = Image.fromarray(im)
    gradImage.save(path)

def showEdgeInstanceImage(edges, dim1, dim2):
    im = imageFromEdgeInstances(edges, dim1, dim2)
    gradImage = Image.fromarray(im)
    gradImage.show("Edge instances")

def showCandidateEdges(candidateEdgesImg):
    candidateEdgesMask = (candidateEdgesImg > np.amax(candidateEdgesImg)/2)
    gradImage = Image.fromarray(candidateEdgesMask)
    gradImage.show("Candidate edges")
