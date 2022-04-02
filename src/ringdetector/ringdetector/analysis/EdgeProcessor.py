import numpy as np

class EdgeProcessor():
    def __init__(self, edges):
        # gets an ouput from one of the Edge detection algos and postprocesses to find edges
        self.edges = self.__getEdges(edges)
        self.dim1 = np.shape(edges)[0]
        self.dim2 = np.shape(edges)[1]
        self.edgeInstances = self.collectEdgeInstances()

    def collectEdgeInstances(self):
        # loop through the entire matrix and look for a signal
        detectedShapes = list()
        # definitely not the fastest
        for i in range(self.dim1):
            for j in range(self.dim2):
                if self.edges[i, j] == 1 and (i, j) not in self.__flatten(detectedShapes):
                    shape = [(i, j)]
                    newShape = self.__findShape(i, j,shape)
                    detectedShapes.append(newShape)
        return detectedShapes

    def __findShape(self, i, j, shape):
        # explore the 3x3 grid (i,j as well but it will be ignored anyway:
        for addi in [-1, 0, 1]:
            for addj in [-1, 0, 1]:
                # need to check in bounds first:
                if 0 <= i+addi < self.dim1 and 0 <= j+addj < self.dim2:
                    # if still a 1 => add to the shape!
                    if self.edges[i+addi, j+addj] == 1 and (i+addi, j+addj) not in shape:
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
