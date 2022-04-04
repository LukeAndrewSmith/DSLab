from ringdetector.analysis.EdgeProcessor import EdgeProcessor

def testEdgeInstances():
    matrix = [[0, 1, 0, 1],
              [0, 1, 0, 1],
              [0, 1, 0, 1]]
    edgeprocess = EdgeProcessor(matrix)
    assert edgeprocess.edgeInstances == [[(0, 1), (1, 1), (2, 1)], [(0, 3), (1, 3), (2, 3)]]


if __name__ == "__main__":
    testEdgeInstances()
