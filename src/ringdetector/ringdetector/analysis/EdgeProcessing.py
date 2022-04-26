import numpy as np
from PIL import Image
from tqdm import trange, tqdm
import time
from scipy.ndimage import label

from ringdetector.analysis.Edge import Edge

def getEdges(candidateEdgesImg, minEdgeLen, edgeModel):
    '''
        Gets an candidateEdges (grascale image) from one of the Edge detection
        algos and postprocesses to identify edges
    '''
    def __getEdges(candidateEdgesImg, minEdgeLen, edgeModel):
        candidateEdgesMask = (candidateEdgesImg > np.amax(candidateEdgesImg)/2)
        shapes = __getShapes(candidateEdgesMask)
        return identifyEdges(
            shapes, candidateEdgesMask.shape, minEdgeLen, edgeModel
        )

    def __getShapes(candidateEdgesMask):
        # Neighborhood definition
        s = [[1,1,1], [1,1,1], [1,1,1]]  
        # Label the contiguous shapes
        labels, numL = label(candidateEdgesMask, structure=s) 
        # Create an array per shape
        shapes = [[] for _ in range(numL+1)]
        # Append index to the shapes array
        [[shapes[labels[i,j]].append((i,j)) for i in range(candidateEdgesMask.shape[0])] for j in range(candidateEdgesMask.shape[1])]
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

    # ###########################################################
    # ### Plotting functions
    # def ImageFromEdgeInstances(self, edgeType="all"):
    #     """ Create black and white image displaying edge instances. 
    #     -- edgeType: "all" for all edgeInstances, "processed" for 
    #     processedEdgeInstances.
    #     """
    #     im = np.zeros((self.dim1, self.dim2), dtype=np.uint8)
    #     if edgeType == "all": 
    #         edges = self.edgeInstances
    #     else: 
    #         edges = self.processedEdges
    #     for instance in edges:
    #         for point in instance:
    #             im[point] = 255
    #     return im

    # def saveEdgeInstanceImage(self, path):
    #     im = self.ImageFromEdgeInstances()
    #     gradImage = Image.fromarray(im)
    #     gradImage.save(path)
