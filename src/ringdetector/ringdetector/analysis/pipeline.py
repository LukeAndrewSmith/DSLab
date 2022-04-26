import os
import argparse
import warnings
import logging
import coloredlogs
from tqdm import tqdm
import numpy as np
import wandb

from ringdetector.Paths import GENERATED_DATASETS_INNER_PICKLES,\
    RESULTS, RESULTS_PKL, RESULTS_POS, RESULTS_DIAG
from ringdetector.utils.configArgs import getArgs
from ringdetector.analysis.CoreProcessor import CoreProcessor

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
cfg = getArgs(parser)

if __name__ == "__main__":

    if cfg.wb:
        wandb.init(
                entity='treering', project="analysis", name=cfg.wbname
            )
        wandb.config.update(cfg)
    
    paths = [RESULTS, RESULTS_PKL, RESULTS_POS, RESULTS_DIAG]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
            logging.info(f"Created directory {path}")
        else:
            logging.info(
                f"Directory {path} already exists, overwriting any "
                "existing files.")

    samples = []
    if cfg.sample:
        samples.append(cfg.sample)
    else:
        for fname in os.listdir(GENERATED_DATASETS_INNER_PICKLES):
            samples.append(fname[:-4])

    wbMetrics = []
    for sample in tqdm(samples[:5], "Cores"):
        cp = CoreProcessor(sample, 
                            readType=cfg.ipread,
                            denoiseH=cfg.denoiseh, 
                            denoiseTemplateWindowSize=cfg.denoisetempwind, 
                            searchWindowSize=cfg.denoisesearchwind,
                            gradMethod=cfg.ipgrad,
                            cannyMin=cfg.cannymin,
                            cannyMax=cfg.cannymax,
                            minEdgeLen=cfg.minedgelen,
                            edgeModel=cfg.edgemodel)
        cp.scoreCore()
        logging.info(f"Sample {sample}: prec {round(cp.precision,3)}, "
            f"rec {round(cp.recall, 3)}")
        if cfg.wb:
            cp.reportCore()
        wbMetrics.append([cp.sampleName, cp.precision, cp.recall])
        cp.exportCoreImg(RESULTS_DIAG)
        cp.exportCoreShapeImg(RESULTS_DIAG)
        cp.exportPos(RESULTS_POS, sanityCheck=True)
        cp.toPickle(RESULTS_PKL)

    if cfg.wb:
        wbTable = wandb.Table(
            data=wbMetrics, columns=["core", "precision", "recall"]
        )
        precHist = wandb.plot.histogram(
            wbTable, value='precision', title='Precision')
        recHist = wandb.plot.histogram(
            wbTable, value='recall', title='Recall')
        scatter = wandb.plot.scatter(
            wbTable, x='recall', y='precision', title='Precision vs. Recall')
    
        wandb.log({'precision_hist': precHist, 
                'recall_hist': recHist, 
                'scatter': scatter})

    wbMetrics = np.array(wbMetrics)
    prec = wbMetrics[:,1].astype(np.double)
    rec = wbMetrics[:,2].astype(np.double)
    
    for name, data in [("precision", prec), ("recall", rec)]:
        summary = (f"{name}: mean {round(np.mean(data), 3)}, "
                f"median: {round(np.median(data), 3)}, "
                f"std {round(np.std(data), 4)}, "
                f"min: {round(np.min(data),3)}, max: {round(np.max(data),3)}")
        logging.info(summary)
        if cfg.wb:
            wandb.run.summary[f"{name}_mean"] = round(np.mean(data), 4)
            wandb.run.summary[f"{name}_std"] = round(np.median(data), 4)
