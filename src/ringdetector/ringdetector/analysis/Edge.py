import numpy as np
import cv2
import math
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class Edge():
    
    def __init__(self, edgeCoords, imgDims):
        """ Input is a list of pixel coordinates that constitute an edge in 
        an image. 
        """
        self.edge = edgeCoords

        self.y = np.array([point[1] for point in self.edge])
        self.X = np.array([point[0] for point in self.edge])
        self.horiz_min = min(self.y)
        self.horiz_max = max(self.y)

        # TODO: don't need to give denoised image, just dims of img
        self.imgheight, self.imgwidth = imgDims
        
        self.model = None
        self.predCoords = None
        self.minDist = None
        self.closestLabelPoint = None
    
    def scoreEdge(self, pointLabels):
        flattened_points = [item for sublist in pointLabels for item in sublist]
        #TODO: should make closestpoints a list of points with min dist
        self.minDist, self.closestLabelPoint = self.__findClosestLabel(
            flattened_points
        )

    # TODO: no need to pass self, probs take this out of class
    def __pixelDist(self, a, b):
        dx2 = (a[0]-b[0])**2
        dy2 = (a[1]-b[1])**2
        return math.sqrt(dx2 + dy2)

    # TODO: this flipped point handling kinda sucks, need to pick an order of
    # of coordinates. i think the point label order is correct wrt cv2 plotting.
    def __findClosestLabel(self, flattened_points):
        min_dist = 100000
        min_label_point = (0,0)
        for point in flattened_points:
            flipped_point = [point[1], point[0]]
            for pred in self.predCoords:
                dist = self.__pixelDist(flipped_point, pred)
                if dist > 1000 and dist > min_dist:
                    break
                else:
                    if dist < min_dist:
                        min_dist = dist
                        min_label_point = flipped_point
        return min_dist, min_label_point

    def fitPredict(self, model="linear"):
        pred_domain = np.arange(-10, self.imgheight+10, 1)
        
        if model == "linear":
            self.model = LinearRegression()
            self.model.fit(self.X.reshape(-1, 1),self.y)
            
            height_pred = np.rint(
                self.model.predict(pred_domain.reshape(-1, 1))
            ).astype(np.int64)
            mse_pred = np.rint(
                self.model.predict(self.X.reshape(-1, 1))
            ).astype(np.int64)

        self.predCoords = list(zip(pred_domain, height_pred))

        self.mse = mean_squared_error(self.y, mse_pred)

    def showEdge(self):
        self.edgeim_gbr = np.zeros(
            (self.imgheight, self.imgwidth, 3), 
            dtype=np.uint8
        )
        for point in self.edge:
            self.edgeim_gbr[point] = (255,255,255)

        for coord in self.predCoords:
            self.edgeim_gbr[coord[0], coord[1],:] = (0,0,255)

        cv2.imshow(
            'EdgeGBR', self.edgeim_gbr[:,self.horiz_min-60:self.horiz_max+60,:]
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()