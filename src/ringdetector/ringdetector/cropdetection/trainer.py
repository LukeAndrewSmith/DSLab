from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader
from augmentation import RatioResize

class CustomizedTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper_train = DatasetMapper(cfg, is_train=True, augmentations=[RatioResize(0.15)])## TODO(2): augs
        train_loader  = build_detection_train_loader(cfg, mapper=mapper_train)
        
        return train_loader
    ## TODO(1): validator
    ## TODO(2): testor