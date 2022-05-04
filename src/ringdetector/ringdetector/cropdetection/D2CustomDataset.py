from argparse import ArgumentError
import os

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from ringdetector.preprocessing.ImageAnnotation import ImageAnnotation
from ringdetector.preprocessing.GeometryUtils import transform_to_xywha,\
    transform_to_xyxy, transform_to_xywh, min_bounding_rectangle

class D2CustomDataset():
    def __init__(self,  json_path, pos_path, ) -> None:
        self.json_path = os.path.join(json_path)
        self.pos_path = os.path.join(pos_path)
    
    def __getLabelMePoly(self, core, annoType):
        if annoType == "inner":
            return core.origInnerRectangle
        elif annoType == "outer" or annoType == "outerInner":
            return core.origOuterRectangle
        elif annoType == "cracks":
            # append cracks and gaps and return them
            return core.cracks + core.gaps
        else:
            raise ArgumentError("Unsupported annoType")

    def __getBbox(self, poly, angle):
        if angle:
            return transform_to_xywha(poly)
        else: 
            return transform_to_xyxy(poly)

    def __convertToRectangle(self, shape):
        boundingRect = min_bounding_rectangle(shape)
        return boundingRect

    def __getSegPoly(self, poly, annoType, core):
        if annoType == "outerInner":
            # need to overwrite the poly for this use case
            poly = self.__getLabelMePoly(core, 'inner')

        segPoly = []
        for point in poly:
            segPoly.append(float(point[0])+.5)
            segPoly.append(float(point[1])+.5)
        return segPoly

    def __generator(self, annoType, angle, split, cracks):
        dataset = []

        for file in os.listdir(os.path.join(self.json_path, split)):
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
                        "segmentation": [self.__getSegPoly(poly, annoType, core)],
                        "category_id": 0})

                    if cracks:
                        # list of crack annotations for the core as polygon shape
                        polyCracks = self.__getLabelMePoly(core, "cracks")
                        for poly in polyCracks:
                            annos.append({
                                # need to convert to rect before here:
                                "bbox": self.__getBbox(self.__convertToRectangle(poly), angle),
                                # TODO: make this change based on angle
                                "bbox_mode": BoxMode.XYXY_ABS,  # BoxMode.XYWHA_ABS
                                "segmentation": [self.__getSegPoly(poly, "cracks", core)],
                                "category_id": 1
                            })

                dataset.append({
                    "file_name": os.path.join(
                        self.json_path, img.image_path),
                    "height": img.height,
                    "width": img.width,
                    "image_id": os.path.basename(img.image_path),
                    "annotations":annos})
        
        return dataset

    def generate_dataset(self, split, annoType, angle, cracks):

        dataset = self.__generator(annoType, angle, split, cracks)

        return dataset