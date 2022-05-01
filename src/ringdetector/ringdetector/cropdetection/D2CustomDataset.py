from argparse import ArgumentError
import os

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from ringdetector.preprocessing.ImageAnnotation import ImageAnnotation
from ringdetector.preprocessing.GeometryUtils import transform_to_xywha,\
    transform_to_xyxy

class D2CustomDataset():
    def __init__(self, json_path, pos_path, ) -> None:
        self.json_path = os.path.join(json_path)
        self.pos_path = os.path.join(pos_path)
    
    def __getLabelMePoly(self, core, annoType):
        if annoType == "inner":
            return core.origInnerRectangle
        elif annoType == "outer": 
            return core.origOuterRectangle
        else:
            raise ArgumentError("Unsupported annoType")

    def __getBbox(self, poly, angle):
        if angle:
            return transform_to_xywha(poly)
        else: 
            return transform_to_xyxy(poly)

    def __getSegPoly(self, poly):
        segPoly = []
        for point in poly:
            segPoly.append(float(point[0])+.5)
            segPoly.append(float(point[1])+.5)
        return segPoly

    def __generator(self, annoType, angle):
        #TODO: type = "inner", "outer", "gap", "center"

        dataset = []

        for file in os.listdir(self.json_path):
            if file.endswith(".json"):
                img = ImageAnnotation(
                    os.path.join(self.json_path, file), 
                    self.pos_path
                )
                
                annos = []
                for core in img.core_annotations:
                    poly = self.__getLabelMePoly(core, annoType)
                    annos.append(
                        {
                        "bbox": self.__getBbox(poly, angle),
                        # TODO: make this change based on angle
                        "bbox_mode": BoxMode.XYXY_ABS, #BoxMode.XYWHA_ABS
                        "segmentation": [self.__getSegPoly(poly)],
                        "category_id": 0})

                dataset.append({
                    "file_name": os.path.join(
                        self.json_path, img.image_path),
                    "height": img.height,#TODO(3): integer
                    "width": img.width,#TODO(3): integer
                    "image_id": os.path.basename(img.image_path),
                    "annotations":annos})
        
        return dataset

    def generate_dataset(self, split, annoType, angle):
        #TODO: implement train test datasplit, or use multiple datasets

        dataset = self.__generator(annoType, angle)
        
        #data = []## TODO(3): multiple datasets, use list.extend()
        #metadata = [] ## TODO(3): multiple datasets, metadata instance
        
        #name = "train"
        #DatasetCatalog.register(name, lambda )
        #data = DatasetCatalog.get(name)
        #metadata = MetadataCatalog.get(name)
        #metadata.set(thing_classes=[f"{annoType}_{angle}"])
        
        #return data, metadata
        return dataset