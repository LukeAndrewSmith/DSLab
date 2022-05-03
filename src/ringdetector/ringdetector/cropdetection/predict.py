import os
from detectron2.utils.logger import setup_logger
from predictor import CustomizedPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from utils import get_cuda_info
from model_config import generate_config
from D2CustomDataset import D2CustomDataset
from visualizer import visualize_pred
from ringdetector.Paths import D2_RESULTS, LABELME_JSONS, POINT_LABELS
import argparse


def predicting(dataMode, modelPath, split="test", k=6):
    ## TODO(3): model_config refactoring using lazy config: https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html#lazy-configs
    ## TODO(3): config argparse
    # register dataset to predict on:
    ds = D2CustomDataset(
        json_path=LABELME_JSONS, pos_path=POINT_LABELS
    )
    DatasetCatalog.register(split, lambda d=split: ds.generate_dataset(d, dataMode, False))
    dataset = DatasetCatalog.get(split)
    metadata = MetadataCatalog.get(split)
    metadata.set(thing_classes=[f"inner_crop"])

    resDir = os.path.join(modelPath.split('/')[:-1])
    cfg = generate_config(resDir, (), (split,))
    cfg.MODEL.WEIGHTS = modelPath  # path to the model we just trained
    cfg.MODEL.DEVICE = "cpu"

    ## TODO(2): ze can be decouplload actual test dataset so that predict and visualied, this is only a temporary resort!
    ## NOTE: the returned dataset is for visualization purpose. type(dataset) is list[dicts], type(metatdata) is metatdata instance.
    ## NOTE: metadata_train.evaluator_type is None cuz currently using rotated coco eval as default

    # use custom predictor to get the ratioresize 0.15
    predictor = CustomizedPredictor(cfg)

    # visualize the predictions
    visualize_pred(dataset, metadata, predictor, k=k)

if __name__ == "__main__":
    #print(get_cuda_info())
    parser = argparse.ArgumentParser(description='Arguments what to predict and visualize')
    parser.add_argument("--datamode", "-d", dest="dataMode", default="inner", choices=['outer', 'inner', 'outerInner'],
                        type=str)
    parser.add_argument("--split", dest="split", default="test", choices=['train', 'val', 'test'], type=str)
    parser.add_argument("--modelpath", "-m", dest="modelPath", type=str)
    setup_logger()

    predicting()