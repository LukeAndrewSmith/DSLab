import numpy as np
from PIL import Image
from tqdm import trange, tqdm
import time
from scipy.ndimage import label

from ringdetector.analysis.Edge import Edge

class EdgeProcessor():
    def __init__(self, edges, cfg):
        # gets an ouput from one of the Edge detection algos and postprocesses to find edges, e.g. the output of the cv2.Canny function
        dim1 = np.shape(edges)[0]
        dim2 = np.shape(edges)[1]

        self.binaryEdgeMatrix = self.__getBinaryEdgeMatrix(edges)
        
        self.cfg = cfg

        s = [[1,1,1],                                                # neighborhood definition
            [1,1,1],
            [1,1,1]]
        labels, numL = label(self.binaryEdgeMatrix, structure=s)     # Label the contiguous shapes
        shapes = [[] for _ in range(numL+1)]                         # Create an array per shape
        [[shapes[labels[i,j]].append((i,j)) for i in range(dim1)] for j in range(dim2)] # Append index to the shapes array
        self.shapes = shapes[1:]                                     # 0 label => no shape, didn't but an 'if' in the previous line as it's much slower

        self.filteredShapes = None
        self.edges = None


    def __getBinaryEdgeMatrix(self, edges):
        # converts to binary for further processing:
        # assumes edges to be min=0, max=? prob 255
        binaryEdgeMatrix = np.around(
            edges/np.max(edges), decimals=0).astype(int)
        return binaryEdgeMatrix

    ### Process shapes into Edges
    def processEdgeInstances(self):
        # we can filter and do further processing on the edgeinstances here
        self.filteredShapes = self.__filterByLength(self.cfg.minedgelen)
        #TODO: try linking edge instances like in the paper
        self.edges = []
        for shape in self.filteredShapes:
            edge = Edge(shape, self.binaryEdgeMatrix.shape)
            edge.fitPredict(model=self.cfg.edgemodel)
            self.edges.append(edge)

    def __filterByLength(self, minLength):
        filteredShapes = [
            instance for instance in self.shapes 
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

    ### Utils
    def __flatten(self, list):
        return [item for sublist in list for item in sublist]
