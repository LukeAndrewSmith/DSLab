import os

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper

from augmentation import RatioResize
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
    
    def __generator_test(self):
        pass

    def __register_dataset(self, generator):
        DatasetCatalog.register(self.name, generator)
        MetadataCatalog.get(self.name).set(thing_classes=["inner_core"])

    def generate_dataset(self):## TODO(2):validation
        if self.is_train:
            generator = self.__generator_train
        else:
            generator = self.__generator_test
        
        self.__register_dataset(generator)
        
        data = generator()
        metadata = MetadataCatalog.get(self.name)

        dataset_mapper = DatasetMapper(self.cfg, is_train=self.is_train, augmentations=[RatioResize(0.15)])## TODO(2): augs
        
        return data, metadata, dataset_mapper