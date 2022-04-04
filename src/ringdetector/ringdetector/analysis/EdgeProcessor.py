import numpy as np
from PIL import Image
from tqdm import trange

class EdgeProcessor():
    def __init__(self, edges):
        # gets an ouput from one of the Edge detection algos and postprocesses to find edges, e.g. the output of the cv2.Canny function
        self.edges = self.__getEdges(edges)
        self.dim1 = np.shape(edges)[0]
        self.dim2 = np.shape(edges)[1]
        self.edgeInstances = self.__collectEdgeInstances()
        self.processedEdges = None

    def __collectEdgeInstances(self):
        # loop through the entire matrix and look for a signal
        detectedShapes = list()
        # definitely not the fastest
        for i in trange(self.dim1, desc= "Collecting edge instances"):
            for j in range(self.dim2):
                if (self.edges[i, j] == 1 and 
                    (i, j) not in self.__flatten(detectedShapes)):
                    shape = [(i, j)]
                    newShape = self.__findShape(i, j,shape)
                    detectedShapes.append(newShape)
        return detectedShapes

    def processEdgeInstances(self, minLength):
        # we can filter and do further processing on the edgeinstances here
        self.processedEdges = self.filterEdgeInstances(minLength)
        # her we can have :
        # self.linkEdgeInstances
        # self.fitEdgeInstances or whatever...

    def filterEdgeInstances(self, minLength):
        filteredEdges = [
            instance for instance in self.edgeInstances 
                if len(instance) >= minLength
        ]
        return filteredEdges

    def ImageFromEdgeInstances(self):
        im = np.zeros((self.dim1, self.dim2))
        for instance in self.edgeInstances:
            for point in instance:
                im[point] = 255
        return im

    def saveEdgeInstanceImage(self, path):
        im = self.ImageFromEdgeInstances()
        gradImage = Image.fromarray(im.astype(np.uint8))
        gradImage.save(path)

    def __findShape(self, i, j, shape):
        # explore the 3x3 grid (i,j) as well but it will be ignored anyway:
        for addi in [-1, 0, 1]:
            for addj in [-1, 0, 1]:
                # need to check in bounds first:
                if 0 <= i+addi < self.dim1 and 0 <= j+addj < self.dim2:
                    # if still a 1 => add to the shape!
                    if (self.edges[i+addi, j+addj] == 1 and 
                        (i+addi, j+addj) not in shape):
                        shape.append((i+addi, j+addj))
                        shape = self.__findShape(i+addi, j+addj, shape)
        return shape

    def __getEdges(self, edges):
        # converts to binary for further processing:
        # assumes edges to be min=0, max=? prob 255
        binaryEdges = np.around(edges/np.max(edges), decimals=0).astype(int)
        return binaryEdges

    def __flatten(self, list):
        return [item for sublist in list for item in sublist]
