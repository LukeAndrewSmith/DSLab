import os
import argparse
import warnings
import logging
import coloredlogs
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from ringdetector.Paths import GENERATED_DATASETS_INNER, \
    GENERATED_DATASETS_INNER_PICKLES
from ringdetector.utils.configArgs import getArgs
from ringdetector.analysis.CoreProcessor import CoreProcessor

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
cfg = getArgs(parser)

if __name__ == "__main__":

    logging.info("Processing Cores")
    
    resultDir = os.path.join(GENERATED_DATASETS_INNER, "results")
    if not os.path.exists(resultDir):
        os.mkdir(resultDir)
        logging.info(
            f"Created directory {resultDir} for inner dataset results."
        )
    else:
        logging.info(
            f"Directory {resultDir} already exists, overwriting "
            "existing result files.")

    samples = []
    if cfg.sample:
        samples.append(cfg.sample)
    else:
        for fname in os.listdir(GENERATED_DATASETS_INNER_PICKLES):
            samples.append(fname[:-4])

    cores = []    
    for sample in tqdm(samples, "Cores:"):
        cp = CoreProcessor(sample, cfg)
        cp.scoreCore()
        cp.exportCoreImg(resultDir)
        # TODO: could also pickle the CoreProcessor object
        cores.append(cp)

    prec = np.array([cp.precision for cp in cores])
    rec = np.array([cp.recall for cp in cores])

    for name, data in [("Precision", prec), ("Recall", rec)]:
        summary = (f"{name}: mean {np.mean(data)}, median: {np.median(data)}"
            f"std {np.std(data)}, min: {np.min(data)}, max: {np.max(data)}")
        logging.info(summary)
        plt.hist(data, bins=15)
        plt.title(f"{name} across samples")
        plt.savefig(os.path.join(resultDir, f'{name}.png'))
    