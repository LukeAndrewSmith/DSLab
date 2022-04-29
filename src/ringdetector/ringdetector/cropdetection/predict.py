import os

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor

from utils import get_cuda_info
from model_config import generate_config
from dataset import CropDataset
from visualizer import visualize_pred

LABELME_JSONS = '/home/leona/Documents/ds_lab/dslabtreering/src/json_files/'
POINT_LABELS = '/home/leona/Documents/ds_lab/dslabtreering/src/pos_files/'
CKPT_DIR = '/home/leona/Documents/ds_lab/dslabtreering/src/output/'

##NOTE: current implementation does not support returning concatenated dataset (see the TODO in `dataset.py`), so this should only contain one element each!
DATASET_TRAIN = ("crop_detection_train",)
DATASET_VAL = ("crop_detection_evaluate",)
DATASET_TEST = ("crop_detection_evaluate",)

def predicting():
    ##TODO(3): model_config refactoring using lazy config: https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html#lazy-configs
    ##TODO(3): config argparse
    cfg = generate_config(CKPT_DIR, (), DATASET_VAL)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained

    ##TODO(2): load actual test dataset so that predict and visualize can be decoupled, this is only a temporary resort!
    ##NOTE: the returned dataset is for visualization purpose. type(dataset) is list[dicts], type(metatdata) is metatdata instance.
    ##NOTE: metadata_train.evaluator_type is None cuz currently using rotated coco eval as default
    data_train, metadata_train = CropDataset(is_train=True, json_path=LABELME_JSONS, pos_path=POINT_LABELS).generate_dataset(DATASET_TRAIN)
    data_evaluate, metadata_evaluate = CropDataset(is_train=False, json_path=LABELME_JSONS, pos_path=POINT_LABELS).generate_dataset(DATASET_VAL)
    
    predictor = DefaultPredictor(cfg)

    visualize_pred(data_train, metadata_train, predictor, k=6)

if __name__ == "__main__":
    print(get_cuda_info())

    ##TODO(2): use wandb logger instead: https://github.com/wandb/artifacts-examples/blob/master/detectron2/wandb_train_net.py
    setup_logger()

    predicting()