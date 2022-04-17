import os

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from ringdetector.preprocessing.ImageAnnotation import ImageAnnotation

class CropDataset():
    def __init__(self, is_train, json_path, pos_path) -> None:
        sub_dir = 'train/' if is_train else 'val/'
        self.json_path = os.path.join(json_path, sub_dir)
        self.pos_path = os.path.join(pos_path, sub_dir)
    
    def __generator(self):
        dataset = []

        for file in os.listdir(self.json_path):
            if file.endswith(".json"):
                img = ImageAnnotation(self.json_path+file, self.pos_path)
                
                annos = []
                for core in img.core_annotations:
                    annos.append({
                        "bbox": core.innerRectangle,
                        "bbox_mode": BoxMode.XYWHA_ABS,
                        "category_id": 0})

                dataset.append({
                    "file_name":self.json_path+img.image_path,
                    "height": img.height,#TODO(3): integer
                    "width": img.width,#TODO(3): integer
                    "image_id": img.image_path,
                    "annotations":annos})
        
        return dataset

    def generate_dataset(self, names):
        generator = self.__generator
        
        data = []## TODO(3): multiple datasets, use list.extnnd()
        metadata = [] ## TODO(3): multiple datasets, metadata instance
        
        for name in names:
            DatasetCatalog.register(name, generator)
            data = DatasetCatalog.get(name)
            metadata = MetadataCatalog.get(name)
            metadata.set(thing_classes=["inner_core"])
        
        return data, metadata