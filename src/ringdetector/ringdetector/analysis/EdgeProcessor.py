import numpy as np
from PIL import Image
from tqdm import trange, tqdm

from ringdetector.analysis.Edge import Edge

class EdgeProcessor():
    def __init__(self, edges, cfg):
        # gets an ouput from one of the Edge detection algos and postprocesses to find edges, e.g. the output of the cv2.Canny function
        self.dim1 = np.shape(edges)[0]
        self.dim2 = np.shape(edges)[1]

        self.binaryEdgeMatrix = self.__getBinaryEdgeMatrix(edges)
        
        self.cfg = cfg

        self.shapes = self.__collectShapeInstances()
        self.filteredShapes = None
        self.edges = None

    ### Identify shapes
    def __collectShapeInstances(self):
        """ Collects continuous shapes
        """
        detectedShapes = list()
        # definitely not the fastest
        for i in trange(self.dim1, desc= "Collecting edge instances"):
            for j in range(self.dim2):
                if (self.binaryEdgeMatrix[i, j] == 1 and 
                    (i, j) not in self.__flatten(detectedShapes)):
                    shape = [(i, j)]
                    newShape = self.__findShape(i, j,shape)
                    detectedShapes.append(newShape)
        return detectedShapes
    
    def __findShape(self, i, j, shape):
        # explore the 3x3 grid (i,j) as well but it will be ignored anyway:
        for addi in [-1, 0, 1]:
            for addj in [-1, 0, 1]:
                # need to check in bounds first:
                if 0 <= i+addi < self.dim1 and 0 <= j+addj < self.dim2:
                    # if still a 1 => add to the shape!
                    if (self.binaryEdgeMatrix[i+addi, j+addj] == 1 and 
                        (i+addi, j+addj) not in shape):
                        shape.append((i+addi, j+addj))
                        shape = self.__findShape(i+addi, j+addj, shape)
        return shape

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
        for shape in tqdm(self.filteredShapes, desc="Fitting edges"):
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
        for edge in tqdm(self.edges, desc="Scoring edges"):
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
