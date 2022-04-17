import os

from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from ringdetector.preprocessing.ImageAnnotation import ImageAnnotation

LABELME_JSONS = 'src/json_files/'
POINT_LABELS = 'src/pos_files/'

class CropDataset():
    def __init__(self, is_train) -> None:
        if is_train:
            self.json_path = os.path.join(LABELME_JSONS, 'train/')
            self.pos_path = os.path.join(POINT_LABELS, 'train/')
        else:
            self.json_path = os.path.join(LABELME_JSONS, 'val/')
            self.pos_path = os.path.join(POINT_LABELS, 'val/')
    
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