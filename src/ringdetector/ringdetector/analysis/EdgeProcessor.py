import numpy as np
from PIL import Image
from tqdm import trange

class EdgeProcessor():
    def __init__(self, edges):
        # gets an ouput from one of the Edge detection algos and postprocesses to find edges, e.g. the output of the cv2.Canny function
        self.dim1 = np.shape(edges)[0]
        self.dim2 = np.shape(edges)[1]

        self.binaryEdgeMatrix = self.__getBinaryEdgeMatrix(edges)

        self.edgeInstances = self.__collectEdgeInstances()
        self.processedEdges = None

    def __collectEdgeInstances(self):
        # loop through the entire matrix and look for a signal
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

    def processEdgeInstances(self, minLength):
        # we can filter and do further processing on the edgeinstances here
        self.processedEdges = self.__filterByLength(minLength)
        # her we can have :
        # self.linkEdgeInstances
        # self.fitEdgeInstances or whatever...

    def __filterByLength(self, minLength):
        #TODO: either this needs to be made private or ProcessEdgeInstances 
        # has to be made private (and called with default args in init)
        filteredEdges = [
            instance for instance in self.edgeInstances 
                if len(instance) >= minLength
        ]
        return filteredEdges

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

    def __flatten(self, list):
        return [item for sublist in list for item in sublist]
