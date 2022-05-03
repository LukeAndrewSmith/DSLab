import os

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from predictor import CustomizedPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from utils import get_cuda_info
from model_config import generate_config
from D2CustomDataset import D2CustomDataset
from visualizer import visualize_pred
from trainer import CustomizedTrainer
from ringdetector.Paths import D2_RESULTS, LABELME_JSONS, POINT_LABELS


CKPT_DIR = os.path.join(D2_RESULTS, "2022-05-02_18-45-19")

##NOTE: current implementation does not support returning concatenated dataset (see the TODO in `dataset.py`), so this should only contain one element each!
DATASET_TRAIN = ("train",)
DATASET_VAL = ("val",)
DATASET_TEST = ("test",)

def predicting():
    ##TODO(3): model_config refactoring using lazy config: https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html#lazy-configs
    ##TODO(3): config argparse
    ds = D2CustomDataset(
        json_path=LABELME_JSONS, pos_path=POINT_LABELS
    )
    DatasetCatalog.register("train", lambda d="train": ds.generate_dataset(d, "inner", False))
    metadata_val = MetadataCatalog.get("train")
    metadata_val.set(thing_classes=[f"inner_crop"])

    DatasetCatalog.register("val", lambda d="val": ds.generate_dataset(d, "inner", False))
    metadata_val = MetadataCatalog.get("val")
    metadata_val.set(thing_classes=[f"inner_crop"])
    dataset_val = DatasetCatalog.get("val")

    DatasetCatalog.register("test", lambda d="test": ds.generate_dataset(d, "inner", False))
    metadata_val = MetadataCatalog.get("test")
    metadata_val.set(thing_classes=[f"inner_crop"])


    cfg = generate_config(CKPT_DIR, DATASET_TRAIN, DATASET_VAL)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.DEVICE = "cpu"
    #trainer = CustomizedTrainer(cfg)
    #val_loader = trainer.build_test_loader(cfg, DATASET_VAL)
    ##TODO(2): ze can be decouplload actual test dataset so that predict and visualied, this is only a temporary resort!
    ##NOTE: the returned dataset is for visualization purpose. type(dataset) is list[dicts], type(metatdata) is metatdata instance.
    ##NOTE: metadata_train.evaluator_type is None cuz currently using rotated coco eval as default

    #data_evaluate, metadata_evaluate = CropDataset(is_train=False, json_path=LABELME_JSONS, pos_path=POINT_LABELS).generate_dataset(DATASET_VAL)
    
    predictor = CustomizedPredictor(cfg)

    visualize_pred(dataset_val, metadata_val, predictor, k=6)

if __name__ == "__main__":
    #print(get_cuda_info())

    ##TODO(2): use wandb logger instead: https://github.com/wandb/artifacts-examples/blob/master/detectron2/wandb_train_net.py
    setup_logger()

    predicting()