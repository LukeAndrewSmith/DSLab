import os

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from ringdetector.preprocessing.ImageAnnotation import ImageAnnotation

LABELME_JSONS = './json_train/'## TODO(2): original_path
POINT_LABELS = './pos_train/'## TODO(2): original_path

class CropDataset():
    def __init__(self, name, cfg, is_train) -> None:
        self.name = name
        self.cfg = cfg
        self.is_train = is_train

    def __generator_train(self):
        dataset = []
        
        for file in os.listdir(LABELME_JSONS):
            if file.endswith(".json"):
                img = ImageAnnotation(LABELME_JSONS+file, POINT_LABELS)
                
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

    def generate_dataset(self):## TODO(1):validation
        if self.is_train:
            generator = self.__generator_train
        else:
            generator = self.__generator_test
        ## Register the dataset
        DatasetCatalog.register(self.name, generator)
        
        data = DatasetCatalog.get(self.name)
        
        metadata = MetadataCatalog.get(self.name)
        metadata.set(thing_classes=["inner_core"])
        
        return data, metadata