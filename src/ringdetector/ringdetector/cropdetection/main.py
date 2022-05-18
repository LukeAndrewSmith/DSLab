import os, datetime
from PIL import Image
import argparse
import warnings
import logging
import coloredlogs
import wandb
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
import cv2

from ringdetector.Paths import DATA
from ringdetector.cropdetection.DetectionProcessor import DetectionProcessor
from ringdetector.cropdetection.trainer import CustomizedTrainer
from ringdetector.cropdetection.D2CustomDataset import D2CustomDataset
from ringdetector.cropdetection.utils import get_cuda_info
from ringdetector.Paths import LABELME_JSONS, POINT_LABELS, D2_RESULTS
from ringdetector.cropdetection.model_config import generate_config
from ringdetector.cropdetection.predictor import CustomizedPredictor
from ringdetector.cropdetection.visualizer import visualizePred, wandbVisualizePred
from ringdetector.utils.configArgs import getCropDetectionArgs

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
args = getCropDetectionArgs(parser)


def registerDatasets(split, dataMode, cracks):
    ds = D2CustomDataset(
        json_path=LABELME_JSONS, pos_path=POINT_LABELS
    )
    DatasetCatalog.register(split, lambda d=split: ds.generate_dataset(d, dataMode, False, cracks))
    metadata = MetadataCatalog.get(split)
    if cracks:
        metadata.set(thing_classes=["inner_crop", "cracks/gaps"])
    else:
        metadata.set(thing_classes=["inner_crop"])


def createOutputDirectory():
    outputDir = os.path.join(D2_RESULTS, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(outputDir, exist_ok=True)
    print(f'results will be stored here: {outputDir}')
    return outputDir


def getModelDirectory(modelPath):
    # gets directory model is in from the full path by clipping the filename
    modelDir = os.path.dirname(modelPath)
    return modelDir


def wandbLog(cfg, args):
    wandb.init(project='cropdetection', sync_tensorboard=True,
               settings=wandb.Settings(start_method="thread", console="off"), config=cfg)
    wandb.config.update({"dataMode": args.dataMode})
    wandb.config.update({"crackDetection": args.cracks})


def wandbLogPredictions(cfg, modelPath, split="val"):
    # predict on the val set
    dataset = DatasetCatalog.get(split)
    metadataset = MetadataCatalog.get(split)
    cfg.MODEL.WEIGHTS = modelPath
    predictor = CustomizedPredictor(cfg)
    results = wandbVisualizePred(dataset, metadataset, predictor, k=3)
    images = [Image.fromarray(image) for image in results]
    wandb.log({"predictions": [wandb.Image(image) for image in images]})


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
        registerDatasets(split, args.dataMode, args.cracks)

    if args.mode == "train":
        outputDir = createOutputDirectory()
        cfg = generate_config(outputDir, ("train",), ("val",))
        wandbLog(cfg, args)

        if args.cracks:
            assert cfg.MODEL.ROI_HEADS.NUM_CLASSES == 2
        else:
            assert cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1

        train(cfg, is_resume=is_resume)

        # log the actual predictions on the val set in wandb:
        # after training is finished there is a model_final.pth in the output dir
        modelPath = os.path.join(outputDir, "model_final.pth")
        wandbLogPredictions(cfg, modelPath)

    elif args.mode == "eval":
        modelDir = getModelDirectory(args.modelPath)
        cfg = generate_config(modelDir, (), (args.split,))

        validate(cfg=cfg, modelPath=args.modelPath, saveDir=modelDir)

    elif args.mode == "pred":
        modelDir = getModelDirectory(args.modelPath)
        cfg = generate_config(modelDir, (), ())
        cfg.MODEL.WEIGHTS = args.modelPath
        # initialize custom predictor
        predictor = CustomizedPredictor(cfg)
        imgPath = os.path.join(DATA, args.imgPath)
        img = cv2.imread(imgPath)
        #NOTE: is this the final pipeline where you input the image one-by-one? I think we should make batch processing possible. 
        #Also, looping the default predictor would be slow as it does not support parallel computing. We may need to rewrite the __call__()
        outputs = predictor(img)
        processor = DetectionProcessor(outputs, args.imgPath, args.csvPath)
        processor.filterDetections()
        processor.exportDetections()


if __name__ == "__main__":
    ##NOTE: memory issues
    Image.MAX_IMAGE_PIXELS = None
    #logging.info(get_cuda_info())
    setup_logger()

    main(args=args, is_resume=True)
