import logging
import coloredlogs
import warnings

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

from ringdetector.preprocessing.DatasetExtractor import DatasetExtractor

if __name__ == "__main__":
    logging.info("Creating inner dataset")
    de = DatasetExtractor()
    de.createInnerDataset()
    logging.info("Inner dataset successfully created")