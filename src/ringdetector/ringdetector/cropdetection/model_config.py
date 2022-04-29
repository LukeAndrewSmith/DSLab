from detectron2 import model_zoo
from detectron2.config import get_cfg

##TODO(2): model_config refactoring
##TODO(2): config argparse
def generate_config(output_dir, dataset_train, dataset_test):
    cfg = get_cfg()

    ## ====Model====
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml") 

    ## ====Dataset====
    cfg.DATASETS.TRAIN = dataset_train
    cfg.DATASETS.TEST = dataset_test

    ## ====Dataloader====
    cfg.DATALOADER.NUM_WORKERS = 4

    ## ====Model:Hyperparam====
    ## TODO(2): rotated rect prediction: https://github.com/facebookresearch/detectron2/issues/21, 
    ## see https://colab.research.google.com/drive/1JXKl48u1fxC35bBryKlQVyQf8tp-DUpE?usp=sharing for a possible solution
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # GPU Sensitive (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set a custom testing threshold
    
    ## ====Solver====
    cfg.SOLVER.IMS_PER_BATCH = 2 #GPU Sensitive 
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 1500
    cfg.SOLVER.STEPS = [] # do not decay learning rate

    cfg.TEST.EVAL_PERIOD = 10

    ## ===Misc===
    cfg.OUTPUT_DIR = output_dir
    cfg.SEED = 12

    # Set seed to negative to fully randomize everything.
    # Set seed to positive to use a fixed seed. Note that a fixed seed increases
    # reproducibility but does not guarantee fully deterministic behavior.
    # Disabling all parallelism further increases reproducibility.

    return cfg