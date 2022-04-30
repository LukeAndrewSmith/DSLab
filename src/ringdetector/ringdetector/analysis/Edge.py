import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from ringdetector.preprocessing.GeometryUtils import pixelDist

class Edge():
    
    def __init__(self, edgeCoords, imgDims):
        """ Input is a list of pixel coordinates that constitute an edge in 
        an image. 
        """
        # NOTE: shape finding inverts our coordinates. Here we set them back
        # to (x,y)
        self.edge = [(coord[1],coord[0]) for coord in edgeCoords]

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


    def __findClosestLabel(self, flattened_points):
        min_dist = 100000
        min_label_point = [0,0]
        for point in flattened_points:
            for pred in self.predCoords:
                dist = pixelDist(point, pred)
                if dist > 1000 and dist > min_dist:
                    break
                else:
                    if dist < min_dist:
                        min_dist = dist
                        min_label_point = point
        return min_dist, min_label_point

    def fitPredict(self, model="linear"):
        # NOTE: our domain for prediction is the y axis of the image
        # because we assume that our rings our vertical, so we want to predict
        # an x axis value at each y position in the image.
        # Below, we use the traditional X, y notation for features and label
        # however keep in mind that X is actually image y values, and y
        # is image x values.
        y = np.array([point[0] for point in self.edge])
        X = np.array([point[1] for point in self.edge])

        pred_domain = np.arange(-10, self.imgheight+10, 1)
        
        if model == "linear":
            self.model = LinearRegression()
            self.model.fit(X.reshape(-1, 1), y)
            
            height_pred = np.rint(
                self.model.predict(pred_domain.reshape(-1, 1))
            ).astype(np.int64)
            mse_pred = np.rint(
                self.model.predict(X.reshape(-1, 1))
            ).astype(np.int64)

        self.predCoords = list(zip(height_pred, pred_domain))

        self.mse = mean_squared_error(y, mse_pred)

    # TODO: fix this to something that makes sense, I want to keep 
    # plotting capability within the class
    def showEdge(self, img, pred=False):
        y = np.array([point[0] for point in self.edge])
        horiz_min = min(y)
        horiz_max = max(y)

        self.edgeim_gbr = np.zeros(
            (self.imgheight, self.imgwidth, 3), 
            dtype=np.uint8
        )
        for point in self.edge:
            img[point] = (255,255,255)

        if pred:
            for coord in self.predCoords:
                img[coord[0], coord[1],:] = (0,0,255)

        cv2.imshow(
            'EdgeGBR', self.edgeim_gbr[:,horiz_min-60:horiz_max+60,:]
        )
        cv2.waitKey(0)
        cv2.destroyAllWindows()