# instance of one core detection with meta info
import cv2
import torch
import numpy as np
from ringdetector.preprocessing.GeometryUtils import min_bounding_rectangle
import rasterio.features
import shapely.geometry

class CoreDetection:
    def __init__(self, pred_box, score, pred_class, pred_mask):
        self.box = pred_box
        self.score = score
        # name class not possible
        self.predClass = pred_class
        self.mask = pred_mask
        # get more info:
        self.maskSize = self.__getSizeFromMask(self.mask)
        self.maskRectangle = self.__getRectangleFromMask(self.mask)
        self.maskRectangleSize = self.__getSizeFromAngledRectangle(self.maskRectangle)


    def __getSizeFromMask(self, mask):
        sizeTensor = torch.count_nonzero(mask * 1)
        return int(sizeTensor.cpu().detach().numpy())

    def __getRectangleFromMask(self, mask):
        # convert to polygon and then apply min bounding rectangle
        polygons = self.__getPolygonFromMask(mask)
        if len(polygons) == 1:
            boundingRect = self.__convertToRectangle(polygons)
            return boundingRect
        elif len(polygons) > 1:
            # flatten out in case of multiple polygons in one box
            polygons = list(np.concatenate(polygons).flat) # TODO List comprehesnion
            boundingRect = self.__convertToRectangle(polygons)
            return boundingRect
        else:
            return None
        # The bounding rectangle is in a polygon interpretable format

    def __convertToRectangle(self, shape):
        # in case there are multiple polygons: connect them in one shape and do a bounding rect for the union
        boundingRect = min_bounding_rectangle(shape)
        return boundingRect

    def __getPolygonFromMask(self, mask):
        mask_n = (mask.cpu().detach().numpy() * 1).astype(np.uint8)
        shapes = rasterio.features.shapes(mask_n)
        polygons = [shape[0]["coordinates"][0] for shape in shapes if shape[1] == 1]

        return polygons

    def __getSizeFromAngledRectangle(self, rect):
        # NOTE: Rectangle points are stored clockwise from the bottom left:
        # 1 ----- 2
        # |       |
        # 0 ----- 3
        if rect is not None:
            height = np.linalg.norm(rect[1] - rect[0])
            width = np.linalg.norm(rect[2] - rect[1])
            return height * width
        else:
            return 0
