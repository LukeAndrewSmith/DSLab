import numpy as np
from PIL import Image
from tqdm import trange, tqdm
import time
from scipy.ndimage import label

from ringdetector.analysis.Edge import Edge

class EdgeProcessor():
    '''
        Gets an candidate edges from one of the Edge detection algos and postprocesses 
        to find edges
    '''
    def __init__(self, candidateEdges, cfg):
        self.edges = self.__getEdges(candidateEdges, cfg)

        
    def __getEdges(self, candidateEdges, cfg):
        binaryEdgeMatrix = self.__getBinaryEdgeMatrix(candidateEdges)
        
        s = [[1,1,1],                                       # neighborhood definition
            [1,1,1],
            [1,1,1]]
        labels, numL = label(binaryEdgeMatrix, structure=s) # Label the contiguous shapes
        shapes = [[] for _ in range(numL+1)]                # Create an array per shape
        [[shapes[labels[i,j]].append((i,j)) for i in range(candidateEdges.shape[0])] 
                                            for j in range(candidateEdges.shape[1])] # Append index to the shapes array
        shapes = shapes[1:]                            # 0 label => no shape, didn't but an 'if' in the previous line as it's much slower

        return self.processEdgeInstances(binaryEdgeMatrix, shapes, cfg)
    


    def __getBinaryEdgeMatrix(self, edges):
        # converts to binary for further processing:
        # assumes edges to be min=0, max=? prob 255
        binaryEdgeMatrix = np.around(
            edges/np.max(edges), decimals=0).astype(int)
        return binaryEdgeMatrix

    ### Process shapes into Edges
    def processEdgeInstances(self, binaryEdgeMatrix, shapes, cfg):
        #TODO: try linking edge instances like in the paper
        filteredShapes = self.__filterByLength(shapes, cfg.minedgelen)
        edges = []
        for shape in filteredShapes:
            edge = Edge(shape, binaryEdgeMatrix.shape)
            edge.fitPredict(model=cfg.edgemodel)
            edges.append(edge)
        return edges

    def __filterByLength(self, shapes, minLength):
        filteredShapes = [
            instance for instance in shapes 
                if len(instance) >= minLength
        ]
        return filteredShapes

    def scoreEdges(self, pointLabels):
        for edge in self.edges:
            edge.scoreEdge(pointLabels)

    ### Plotting functions
    def ImageFromEdgeInstances(self, edgeType="all"):
        """ Create black and white image displaying edge instances. 
        -- edgeType: "all" for all edgeInstances, "processed" for 
        processedEdgeInstances.
        """
        im = np.zeros((self.dim1, self.dim2), dtype=np.uint8)
        if edgeType == "all": 
            edges = self.edgeInstances
        else: 
            edges = self.processedEdges
        for instance in edges:
            for point in instance:
                im[point] = 255
        return im

    def saveEdgeInstanceImage(self, path):
        im = self.ImageFromEdgeInstances()
        gradImage = Image.fromarray(im)
        gradImage.save(path)
