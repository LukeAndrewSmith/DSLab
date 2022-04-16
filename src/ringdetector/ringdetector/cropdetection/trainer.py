from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_train_loader

from model_config import generate_config
from dataset import CropDataset

class CustomizedTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg, train_mapper):
        train_loader  = build_detection_train_loader(cfg, mapper=train_mapper)
        return train_loader

## TODO(2): validator

## TODO(1): trainer refactoring
config = generate_config()## TODO(2): model_config refactoring
data_train, metadata_train, mapper_train = CropDataset(name="crop_detection_train", cfg=config, is_train=True)

trainer = CustomizedTrainer(config, mapper_train)

## TODO(2): validator