from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from augmentation import RatioResize

##TODO(1): evaluator
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py
# https://github.com/facebookresearch/detectron2/blob/main/tools/train_net.py
# https://github.com/facebookresearch/detectron2/blob/main/tools/plain_train_net.py

class CustomizedTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        mapper_train = DatasetMapper(cfg, is_train=True, augmentations=[RatioResize(0.15)])## TODO(2): augs
        train_loader  = build_detection_train_loader(cfg, mapper=mapper_train)
        
        return train_loader

    # @classmethod
    # def build_evaluator(cls, cfg, dataset_name, output_folder=None):
    #     return build_evaluator(cfg, dataset_name, output_folder)

    # @classmethod
    # def test_with_TTA(cls, cfg, model):
    #     logger = logging.getLogger("detectron2.trainer")
    #     # In the end of training, run an evaluation with TTA
    #     # Only support some R-CNN models.
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


# class CustomizedEvaluator(DefaultTrainer):
#     @classmethod
#     def build_evaluator(cfg, dataset_name):
#         pass
    
#     @classmethod
#     def build_test_loader(cls, cfg, dataset_name): # classmethod build_test_loader(cfg, dataset_name)
#         mapper_test = DatasetMapper(cfg, is_train=False, augmentations=[RatioResize(0.15)])## TODO(2): augs
#         test_loader  = build_detection_test_loader(dataset_name, mapper=mapper_test)
        
#         return test_loader
    
    # classmethod test(cfg, model, evaluators=None)