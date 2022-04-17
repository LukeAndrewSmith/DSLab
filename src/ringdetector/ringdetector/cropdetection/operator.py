from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import RotatedCOCOEvaluator

from augmentation import RatioResize

class CustomizedTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        # 1. ”instances_predictions.pth” a file that can be loaded with torch.load 
        # and contains all the results in the format they are produced by the model.
        # 2. ”coco_instances_results.json” a json file in COCO’s result format.
        
        ## NOTE: RotatedCOCOEvaluator uses IOU only and does not consider angle differences.
        ## TODO(1): RotatedCOCOEvaluator has internal bugs that produce nonsensical evaluation results rn (see colab sample). Customization is needed.
        return RotatedCOCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper_train = DatasetMapper(cfg, is_train=True, augmentations=[RatioResize(0.15)])## TODO(2): augs
        train_loader  = build_detection_train_loader(cfg, mapper=mapper_train)
        
        return train_loader
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper_test = DatasetMapper(cfg, is_train=False, augmentations=[RatioResize(0.15)])## TODO(2): augs
        test_loader  = build_detection_test_loader(cfg, dataset_name, mapper=mapper_test)
        
        return test_loader
    
    # In the end of training, run an evaluation with TTA
    # Only support some R-CNN models.
    # NOTE: do we need test-time augmentation? diff from build_test_loader?
    # import logging
    # from detectron2.modeling import DatasetMapperTTA, GeneralizedRCNNWithTTA
    # @classmethod
    # def test_with_TTA(cls, cfg, model):
    #     logger = logging.getLogger("detectron2.trainer")
    #     logger.info("Running inference with test-time augmentation ...")
    #     model = GeneralizedRCNNWithTTA(cfg, model)
    #     evaluators = [
    #         cls.build_evaluator(
    #             cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
    #         )
    #         for name in cfg.DATASETS.TEST
    #     ]
    #     res = cls.test(cfg, model, evaluators)
    #     res = OrderedDict({k + "_TTA": v for k, v in res.items()})
    #     return res