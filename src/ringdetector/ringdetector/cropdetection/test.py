from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer

from utils import get_cuda_info
from model_config import generate_config
from ringdetector.cropdetection.operator import CustomizedTrainer

CKPT_DIR = 'src/output/'

##NOTE: current implementation does not support returning concatenated dataset (see the TODO in `dataset.py`), so this should only contain one element each!
DATASET_TEST = ("crop_detection_evaluate",)

def test():
      ##TODO(2): load test dataset

      ##setup config
      ##TODO(2): model_config refactoring using lazy config: https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html#lazy-configs
      ##TODO(2): config argparse
      cfg = generate_config(CKPT_DIR, (), DATASET_TEST)

      ##load model
      ##TODO(2): use a class for testing purpose only
      model = CustomizedTrainer.build_model(cfg)
      ckpt = DetectionCheckpointer(model, save_dir=CKPT_DIR)
      ckpt.resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
      # NOTE: do we need test-time augmentation? diff from build_test_loader?
      # if config.TEST.AUG.ENABLED:
      #       result.update(CustomizedTrainer.test_with_TTA(config, model))##
      # if comm.is_main_process(): ##TODO(2): Distributed Training 
      #       verify_results(config, result)

      ##test
      CustomizedTrainer.test(cfg, model)

      ##TODO(2): visualization

if __name__ == "__main__":
      print(get_cuda_info())
      ##setup logger
      ##TODO(2): use wandb logger instead: https://github.com/wandb/artifacts-examples/blob/master/detectron2/wandb_train_net.py
      setup_logger()

      test()