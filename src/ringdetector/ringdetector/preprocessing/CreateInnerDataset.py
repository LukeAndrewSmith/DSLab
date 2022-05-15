from pip import main
import logging
import coloredlogs
import warnings

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

from ringdetector.preprocessing.InnerCropExtraction import extractInnerCrops

if __name__ == "__main__":
    logging.info("Creating inner dataset")
    extractInnerCrops(saveDataset=True)
    logging.info("Inner dataset successfully created")
