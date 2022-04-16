import os

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from ringdetector.preprocessing.ImageAnnotation import ImageAnnotation

LABELME_JSONS = './json_train/'## TODO(2): original_path
POINT_LABELS = './pos_train/'## TODO(2): original_path

def generate_dataset():
    dataset = []
    
    for file in os.listdir(LABELME_JSONS):
        if file.endswith(".json"):
            img = ImageAnnotation(LABELME_JSONS+file, POINT_LABELS)## TODO(1): modify preprocessing pipeline
            
            annos = []
            for core in img.core_annotations:
                annos.append({
                    "bbox": core.innerRectangle,
                    "bbox_mode": BoxMode.XYWHA_ABS,
                    "category_id": 0})

            dataset.append({
                "file_name":LABELME_JSONS+img.image_path,
                "height": img.height,#TODO(3): integer
                "width": img.width,#TODO(3): integer
                "image_id": img.image_path,
                "annotations":annos})
    
    return dataset

DatasetCatalog.register("obj_detection_train", generate_dataset)
MetadataCatalog.get("obj_detection_train").set(thing_classes=["inner_core"])

obj_detection_train = generate_dataset()
metadata_train = MetadataCatalog.get("obj_detection_train")