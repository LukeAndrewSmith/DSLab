from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader

from augmentation import RatioResize

class CustomizedTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        train_loader  = build_detection_train_loader(cfg, 
                                                     mapper=DatasetMapper(cfg, 
                                                                          is_train=True, 
                                                                          augmentations=[
                                                                              RatioResize(0.15)]))
        return train_loader