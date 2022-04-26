from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer

from utils import get_cuda_info
from dataset import CropDataset
from model_config import generate_config
from ringdetector.cropdetection.operator import CustomizedTrainer

LABELME_JSONS = '/home/leona/Documents/ds_lab/dslabtreering/src/json_files/'
POINT_LABELS = '/home/leona/Documents/ds_lab/dslabtreering/src/pos_files/'
CKPT_DIR = '/home/leona/Documents/ds_lab/dslabtreering/src/output/'

##NOTE: current implementation does not support returning concatenated dataset (see the TODO in `dataset.py`), so this should only contain one element each!
DATASET_VAL = ("crop_detection_evaluate",)

def validate():
    ##create and register dataset, the returned dataset is for visualization purpose. type(dataset) is list[dicts], type(metatdata) is metatdata instance.
    ##NOTE: metadata_train.evaluator_type is None cuz currently using rotated coco eval as default
    data_evaluate, metadata_evaluate = CropDataset(is_train=False, json_path=LABELME_JSONS, pos_path=POINT_LABELS).generate_dataset(DATASET_VAL)

    ##setup config
    ##TODO(2): model_config refactoring using lazy config: https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html#lazy-configs
    ##TODO(2): config argparse
    cfg = generate_config(CKPT_DIR, (), DATASET_VAL)

    ##load model
    model = CustomizedTrainer.build_model(cfg)
    ckpt = DetectionCheckpointer(model, save_dir=CKPT_DIR)
    ckpt.resume_or_load(cfg.MODEL.WEIGHTS, resume=True)
    # NOTE: do we need test-time augmentation? diff from build_test_loader?
    # if config.TEST.AUG.ENABLED:
    #       result.update(CustomizedTrainer.test_with_TTA(config, model))##
    # if comm.is_main_process(): ##TODO(2): Distributed Training 
    #       verify_results(config, result)

    ##validate
    CustomizedTrainer.test(cfg, model)

    ##TODO(2): visualization

if __name__ == "__main__":
    print(get_cuda_info())
    ##setup logger
    ##TODO(2): use wandb logger instead: https://github.com/wandb/artifacts-examples/blob/master/detectron2/wandb_train_net.py
    setup_logger()

    validate()