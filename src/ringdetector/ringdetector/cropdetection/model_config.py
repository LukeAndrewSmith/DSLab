import os

from detectron2 import model_zoo
from detectron2.config import get_cfg

## TODO(2): model_config refactoring
def generate_config():
    cfg = get_cfg()

    ## ====Model====
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") 

    ## ====Dataset====
    cfg.DATASETS.TRAIN = ("crop_detection_train")## train_name
    cfg.DATASETS.TEST = ("crop_detection_evaluate")## test_name

    ## ====Dataloader====
    cfg.DATALOADER.NUM_WORKERS = 2

    ## ====Model:Hyperparam====
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # GPU Sensitive (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class. (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    ## ====Solver====
    cfg.SOLVER.IMS_PER_BATCH = 2 #GPU Sensitive 
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 100
    cfg.SOLVER.STEPS = [] # do not decay learning rate

    ## ===Misc===
    cfg.OUTPUT_DIR = "./output_core"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.SEED = 12

    # Set seed to negative to fully randomize everything.
    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    # Disabling all parallelism further increases reproducibility.

    return cfg