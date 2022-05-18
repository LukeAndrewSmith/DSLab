# class that takes the outputs of a model and nCores
# the instances get filtered and a labelme json format can be produced
from math import isnan
from CoreDetection import CoreDetection
import json, os, csv

class DetectionProcessor:
    def __init__(self, outputs, imgPath, csvPath=None):
        self.instances = outputs['instances']
        self.coreDetections = self.__collectDetections()
        self.imgHeight = self.instances.image_size[0]
        self.imgWidth = self.instances.image_size[1]
        self.imgPath = imgPath
        self.csvPath = csvPath
        if self.csvPath is not None:
            self.core_names, self.start_years = self.__getCoreInfo()
            self.nCores = len(self.core_names)


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
        # TODO sort list by top to bottom based on y coordinate and then assign label of csv if it is given
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
            "imagePath": self.imgPath.split('/')[-1],
            "imageData": None,
            "imageHeight": self.imgHeight,
            "imageWidth": self.imgWidth
        }

        # write to json:
        #dir = os.path.dirname(self.imgPath)
        filename = self.imgPath.split('.')[:-1][0] + '.json'

        with open(filename, 'w') as json_file:
            json.dump(labelmeJson, json_file)

        return filename


    # TODO request: @freddy you need to handle the FNs in the prediction so that we can correctly align the names w/ the cores (I think it can be a part of the crop detection heuristic ticket)
    # NOTE: you need to (write code somewhere else to) assert that the csvPath matches the imgPath before calling the func.
    def __getCoreInfo(self):
        core_names = []
        start_years = []
        correct_header = ['CORE_NAMES', 'START_YEAR']
        
        with open(self.csvPath, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            
            header = next(csv_reader)
            assert header == correct_header, f'CSV file header assertion failed. Did you name and order your header properly with {correct_header}?'

            for index, row in enumerate(csv_reader):
                core_name, start_year = row

                assert (core_name and start_year), f'NaN values is not allowed! Check row {index + 1}.'

                core_names.append(core_name.strip())
                start_years.append(int(start_year.strip()))
        
        return core_names, start_years

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

