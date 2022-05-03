import os, datetime
from detectron2.utils.logger import setup_logger
from PIL import Image
from detectron2.data import MetadataCatalog, DatasetCatalog
from ringdetector.cropdetection.trainer import CustomizedTrainer
from detectron2.checkpoint import DetectionCheckpointer
from ringdetector.cropdetection.D2CustomDataset import D2CustomDataset
from ringdetector.cropdetection.utils import get_cuda_info
from ringdetector.Paths import LABELME_JSONS, POINT_LABELS, D2_RESULTS
from ringdetector.cropdetection.model_config import generate_config
from ringdetector.cropdetection.predictor import CustomizedPredictor
from ringdetector.cropdetection.visualizer import visualizePred
import argparse
import warnings
import logging
import coloredlogs
import wandb

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")


def registerDatasets(split, dataMode):
    ds = D2CustomDataset(
        json_path=LABELME_JSONS, pos_path=POINT_LABELS
    )
    DatasetCatalog.register(split, lambda d=split: ds.generate_dataset(d, dataMode, False))
    metadata = MetadataCatalog.get(split)
    metadata.set(thing_classes=[f"inner_crop"])


def createOutputDirectory():
    outputDir = os.path.join(D2_RESULTS, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(outputDir, exist_ok=True)
    print(f'results will be stored here: {outputDir}')
    return outputDir


def getModelDirectory(modelPath):
    # gets directory model is in from the full path by clipping the filename
    modelDir = os.path.join(modelPath.spli("/")[:-1])
    return modelDir


def wandbLog(cfg, args):
    wandb.init(project='cropdetection', sync_tensorboard=True,
               settings=wandb.Settings(start_method="thread", console="off"), config=cfg)
    wandb.config.update({"dataMode": args.dataMode})


def train(cfg, is_resume):
    trainer = CustomizedTrainer(cfg)
    trainer.resume_or_load(resume=is_resume)
    trainer.train()


def validate(cfg, modelPath, saveDir):
    model = CustomizedTrainer.build_model(cfg)
    ckpt = DetectionCheckpointer(model, save_dir=saveDir)
    ckpt.resume_or_load(modelPath, resume=True)
    CustomizedTrainer.test(cfg, model)


def main(args, is_resume):
    # register all the datasets:
    splits = ["train", "val", "test"]
    for split in splits:
        registerDatasets(split, args.dataMode)

    if args.mode == "train":
        outputDir = createOutputDirectory()
        cfg = generate_config(outputDir, ("train",), ("val",))
        wandbLog(cfg, args)
        train(cfg, is_resume=is_resume)

    elif args.mode == "eval":
        modelDir = getModelDirectory(args.modelPath)
        cfg = generate_config(modelDir, (), (args.split,))

        validate(cfg=cfg, modelPath=args.modelPath, saveDir=modelDir)

    elif args.mode == "pred":
        modelDir = getModelDirectory(args.modelPath)
        cfg = generate_config(modelDir, (), (args.split,))
        cfg.MODEL.WEIGHTS = args.modelPath

        # get data and metadata to predict on
        dataset = DatasetCatalog.get(args.split)
        metadataset = MetadataCatalog.get(args.split)

        # initialize custom predictor
        predictor = CustomizedPredictor(cfg)

        visualizePred(dataset, metadataset, predictor, k)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments what mode to run in and training configurations')
    parser.add_argument("--mode", "-m", dest="mode", default="train", choices=["train", "eval", "pred"],
                        type=str)
    parser.add_argument("--dataMode", "-tm", dest="dataMode", choices=["inner", "outer", "outerInner"],
                        default="inner", type=str)
    parser.add_argument("--modelPath", dest="modelPath", type=str)
    parser.add_argument("-split", dest="split", help="What split to predict on if mode pred or eval is chosen",
                        choices=["train", "val", "test"], type=str)
    parser.add_argument("--k-pred", "-k", dest="k", type=int)
    parser.add_argument("--num-gpus", dest="num-gpus", type=int)
    args = parser.parse_args()

    ##NOTE: memory issues
    Image.MAX_IMAGE_PIXELS = None

    logging.info(get_cuda_info())
    setup_logger()

    main(args=args, is_resume=True)
