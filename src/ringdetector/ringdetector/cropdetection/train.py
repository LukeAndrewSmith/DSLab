import os

from detectron2.utils.logger import setup_logger

from utils import get_cuda_info
from dataset import CropDataset
from model_config import generate_config
from ringdetector.cropdetection.operator import CustomizedTrainer

LABELME_JSONS = 'src/json_files/'
POINT_LABELS = 'src/pos_files/'
OUTPUT_DIR = 'src/output/'

##NOTE: current implementation does not support returning concatenated dataset (see the TODO in `dataset.py`), so this should only contain one element each!
DATASET_TRAIN = ("crop_detection_train",)
DATASET_VAL = ("crop_detection_evaluate",)

def training(is_resume):
      # TODO(2): logging of val metrics upon training. see https://github.com/facebookresearch/detectron2/issues/810 
      # and https://eidos-ai.medium.com/training-on-detectron2-with-a-validation-set-and-plot-loss-on-it-to-avoid-overfitting-6449418fbf4e

      ##setup config
      ##TODO(2): model_config refactoring using lazy config: https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html#lazy-configs
      ##TODO(2): config argparse
      cfg = generate_config(OUTPUT_DIR, DATASET_TRAIN, DATASET_VAL)

      ##create and register dataset, the returned dataset is for visualization purpose. type(dataset) is list[dicts], type(metatdata) is metatdata instance.
      ##NOTE: metadata_train.evaluator_type is None cuz currently using rotated coco eval as default
      data_train, metadata_train = CropDataset(is_train=True, json_path=LABELME_JSONS, pos_path=POINT_LABELS).generate_dataset(DATASET_TRAIN)
      data_evaluate, metadata_evaluate = CropDataset(is_train=False, json_path=LABELME_JSONS, pos_path=POINT_LABELS).generate_dataset(DATASET_VAL)

      trainer = CustomizedTrainer(cfg)
      trainer.resume_or_load(resume=is_resume)
      # NOTE: do we need test-time augmentation? diff from build_test_loader?
      #     if config.TEST.AUG.ENABLED:
      #         trainer.register_hooks(
      #             [hooks.EvalHook(0, lambda: trainer.test_with_TTA(config, trainer.model))]
      #         )
      
      ##train
      trainer.train()

      ##TODO(2): visualization

if __name__ == "__main__":
      print(get_cuda_info())
      ##setup logger
      ##TODO(2): use wandb logger instead: https://github.com/wandb/artifacts-examples/blob/master/detectron2/wandb_train_net.py
      setup_logger()

      os.makedirs(OUTPUT_DIR, exist_ok=True)
      training(is_resume=True)