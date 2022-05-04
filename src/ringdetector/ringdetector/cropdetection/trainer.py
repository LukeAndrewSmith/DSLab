from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator
from augmentation import RatioResize
import detectron2.data.transforms as T


class CustomizedTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(self, cfg, dataset_name):
        # NOTE: RotatedCOCOEvaluator uses IOU only and does not consider angle differences.
        # TODO(1): RotatedCOCOEvaluator has internal bugs(?) that produce nonsensical evaluation results rn (see colab sample). Customization is needed.
        # RotatedCOCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

    @classmethod
    def build_train_loader(self, cfg):
        mapper_train = DatasetMapper(cfg, is_train=True, augmentations=[RatioResize(0.15), T.RandomBrightness(0.9,1.1),
                                                                        T.RandomContrast(0.9, 1.1),
                                                                        T.RandomFlip(horizontal=False, vertical=True),
                                                                        T.RandomCrop(crop_type="relative_range", crop_size=(0.5,0.9))])  #keeps 90-100% width and 50-100% height
        train_loader = build_detection_train_loader(cfg, mapper=mapper_train)
        return train_loader
    
    @classmethod
    def build_test_loader(self, cfg, dataset_name):
        mapper_test = DatasetMapper(cfg, is_train=False, augmentations=[RatioResize(0.15)])
        test_loader = build_detection_test_loader(cfg, dataset_name, mapper=mapper_test)
        
        return test_loader



