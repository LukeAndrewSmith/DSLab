# class that takes the outputs of a model and nCores
# the instances get filtered and a labelme json format can be produced
from CoreDetection import CoreDetection
import json, os

class DetectionProcessor:
    def __init__(self, outputs, imgPath, nCores=20):
        self.instances = outputs['instances']
        self.coreDetections = self.__collectDetections()
        self.nCores = nCores
        self.imgHeight = self.instances.image_size[0]
        self.imgWidth = self.instances.image_size[1]
        self.imgPath = imgPath

    def filterDetections(self):
        # filter until target of cores is reached if nCores is given:
        if self.nCores != None:
            # order by size of mask and take ncores highest:
            # taking detected mask size for now not rectangle
            maskSizesSorted = sorted(self.coreDetections, key=lambda x: x.maskSize, reverse=True)
            if self.nCores < len(maskSizesSorted):
                self.coreDetections = maskSizesSorted[:self.nCores]
        else:
            # need to set a custom threshold for minSize
            minSize = 100000
            self.coreDetections = [d for d in self.coreDetections if d.maskSize > minSize]


    def mergeDetection(self):
        # if two detewction overlap on the y ccordinate put them both in a polygon and get the minbounding rect
        # then create a new detection instead of the old one
        pass


    def exportDetections(self):
        coreList = list()
        for i,core in enumerate(self.coreDetections):
            if core.maskRectangle is not None:
                coreDict = {
                    "label": str(i),
                    "points": core.maskRectangle.tolist(),
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                coreList.append(coreDict)
        labelmeJson = {
            "version": "5.0.1",
            "flags": {},
            "shapes": coreList,
            "imagePath": self.imgPath,
            "imageData": None,
            "imageHeight": self.imgHeight,
            "imageWidth": self.imgWidth
        }

        # write to json:
        #dir = os.path.dirname(self.imgPath)
        filename = self.imgPath.split('.')[:-1][0] + '.json'

        with open(filename, 'w') as json_file:
            json.dump(labelmeJson, json_file)



    def __collectDetections(self):
        detections = list()
        for inst in range(len(self.instances)):
            box = self.instances.get('pred_boxes').tensor[inst,:]
            score = self.instances.get('scores')[inst]
            classes = self.instances.get('pred_classes')[inst]
            mask = self.instances.get('pred_masks')[inst, :,:]
            detection = CoreDetection(box, score, classes, mask)
            detections.append(detection)
        return detections

