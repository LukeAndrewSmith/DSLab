from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from augmentation import RatioResize

##TODO(1): evaluator
# https://detectron2.readthedocs.io/en/latest/modules/engine.html#detectron2.engine.defaults.DefaultTrainer
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py
# https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py

class CustomizedTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper_train = DatasetMapper(cfg, is_train=True, augmentations=[RatioResize(0.15)])## TODO(2): augs
        train_loader  = build_detection_train_loader(cfg, mapper=mapper_train)
        
        return train_loader

class CustomizedEvaluator(DefaultTrainer):
    # classmethod build_evaluator(cfg, dataset_name):
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name): # classmethod build_test_loader(cfg, dataset_name)
        mapper_test = DatasetMapper(cfg, is_train=False, augmentations=[RatioResize(0.15)])## TODO(2): augs
        test_loader  = build_detection_test_loader(dataset_name, mapper=mapper_test)
        
        return test_loader
    
    # classmethod test(cfg, model, evaluators=None)