import numpy as np
from PIL import Image
from tqdm import trange, tqdm
import time
from scipy.ndimage import label

from ringdetector.analysis.Edge import Edge

def getEdges(candidateEdgesImg, cfg):
    '''
        Gets an candidateEdges (grascale image) from one of the Edge detection algos 
        and postprocesses to identify edges
    '''
    def __getEdges(candidateEdgesImg, cfg):
        candidateEdgesMask = (candidateEdgesImg > np.amax(candidateEdgesImg)/2)
        shapes = __getShapes(candidateEdgesMask)
        return identifyEdges(shapes, candidateEdgesMask.shape, cfg)

    def __getShapes(candidateEdgesMask):
        s = [[1,1,1], [1,1,1], [1,1,1]]                       # Neighborhood definition
        labels, numL = label(candidateEdgesMask, structure=s) # Label the contiguous shapes
        shapes = [[] for _ in range(numL+1)]                  # Create an array per shape
        [[shapes[labels[i,j]].append((i,j)) for i in range(candidateEdgesMask.shape[0])] 
                                            for j in range(candidateEdgesMask.shape[1])] # Append index to the shapes array
        shapes = shapes[1:]                                   # 0 label => no shape, didn't but an 'if' in the previous line as it's much slower
        return shapes

    def identifyEdges(shapes, maskShape, cfg):
        #TODO: try linking edge instances like in the paper
        filteredShapes = __filterByLength(shapes, cfg.minedgelen)
        edges = []
        for shape in filteredShapes:
            edge = Edge(shape, maskShape)
            edge.fitPredict(model=cfg.edgemodel)
            edges.append(edge)
        return edges

    def __filterByLength(shapes, minLength):
        filteredShapes = [ instance for instance in shapes 
                                                 if len(instance) >= minLength]
        return filteredShapes

    return __getEdges(candidateEdgesImg, cfg)


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
