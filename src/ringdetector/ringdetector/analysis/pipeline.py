import os
import argparse
import warnings
import logging
import coloredlogs
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import datetime
import wandb

from ringdetector.Paths import GENERATED_DATASETS_INNER, \
    GENERATED_DATASETS_INNER_PICKLES
from ringdetector.utils.configArgs import getArgs
from ringdetector.analysis.CoreProcessor import CoreProcessor

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")

parser = argparse.ArgumentParser()
cfg = getArgs(parser)

now = datetime.datetime.now()

if __name__ == "__main__":

    if cfg.wb:
        wandb.init(
                entity='treering', project="analysis", name=cfg.wbname
            )
        wandb.config.update(cfg)

    logging.info("Processing Cores")
    
    #TODO: remove test here
    resultDir = os.path.join(GENERATED_DATASETS_INNER, "test_results")
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
        cp.exportCoreImg(resultDir)
        cp.exportCoreShapeImg(resultDir)
        cp.toPickle(resultDir)

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

    # Histograms of precision and recall
    fig, axes = plt.subplots(1, 2)
    fig.suptitle("Precision and Recall for all Samples")
    sns.histplot(data=prec, ax=axes[0], bins=30, kde=True)
    sns.histplot(data=rec, ax=axes[1], bins=30, kde=True)

    axes[0].set_xlabel("Precision")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("")

    plt.savefig(os.path.join(resultDir, f'diagnostics_{now}.png'))
    