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

    if cfg.n:
        samples = samples[:cfg.n]

    wbMetrics = []
    wbMetricsTricky = []
    wbMetricsEasy = []
    for sample in tqdm(samples, "Cores"):
        cp = CoreProcessor(sample,readType=cfg.imReadType, denoiseH=cfg.denoiseH,
                denoiseTemplateWindowSize=cfg.denoiseTemplateWindowSize,
                denoiseSearchWindowSize=cfg.denoiseSearchWindowSize, cannyMin=cfg.cannyMin, cannyMax=cfg.cannyMax,
                rightEdgeMethod=cfg.rightEdgeMethod, invertedEdgeWindowSize=cfg.invertedEdgeWindowSize, 
                mergeShapes1Ball=cfg.mergeShapes1Ball, mergeShapes1Angle=cfg.mergeShapes1Angle,
                mergeShapes2Ball=cfg.mergeShapes2Ball, mergeShapes2Angle=cfg.mergeShapes2Angle, 
                filterLengthImgProportion=cfg.filterLengthImgProportion,
                filterRegressionAnglesAngleThreshold=cfg.filterRegressionAnglesAngleThreshold)
        cp.scoreCore()
        logging.info(f"Sample {sample}: prec {round(cp.precision,3)}, "
            f"rec {round(cp.recall, 3)}")
        if cfg.wb:
            cp.reportCore()
        wbMetrics.append([cp.sampleName, cp.precision, cp.recall])
        if cp.core.tricky:
            wbMetricsTricky.append([cp.sampleName, cp.precision, cp.recall])
        else:
            wbMetricsEasy.append([cp.sampleName, cp.precision, cp.recall])
        cp.exportLinePlot(RESULTS_DIAG)
        cp.exportShapePlot(RESULTS_DIAG)
        cp.exportPos(RESULTS_POS, sanityCheck=False)
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
    

        wbTableTricky = wandb.Table(
            data=wbMetricsTricky, columns=["core", "precision", "recall"]
        )
        precHistTricky = wandb.plot.histogram(
            wbTableTricky, value='precision', title='Precision Tricky')
        recHistTricky = wandb.plot.histogram(
            wbTableTricky, value='recall', title='Recall Tricky')
        scatterTricky = wandb.plot.scatter(
            wbTableTricky, x='recall', y='precision', title='Precision vs. Recall Tricky')
    
    
        wbTableEasy = wandb.Table(
            data=wbMetricsEasy, columns=["core", "precision", "recall"]
        )
        precHistEasy = wandb.plot.histogram(
            wbTableEasy, value='precision', title='Precision Easy')
        recHistEasy = wandb.plot.histogram(
            wbTableEasy, value='recall', title='Recall Easy')
        scatterEasy = wandb.plot.scatter(
            wbTableEasy, x='recall', y='precision', title='Precision vs. Recall Easy')


        wandb.log({'precision_hist': precHist, 
                'recall_hist': recHist, 
                'scatter': scatter,
                'precision_hist_tricky': precHistTricky, 
                'recall_hist_tricky': recHistTricky, 
                'scatter_tricky': scatterTricky,
                'precision_hist_easy': precHistEasy, 
                'recall_hist_easy': recHistEasy, 
                'scatter_easy': scatterEasy,
                })

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
