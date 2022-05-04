# DSLabTreeRing

Ring Detector!

## Installation steps:

### Anaconda

Create venv using yaml config: `conda env create -n venv python=3.9.12 -f env_config.yaml`

You will receive a pip error stating that detectron2 can't be installed. After creation, activate the env and run `python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html` (documented here: https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

### Python

TBC

### After environment creation

Navigate to `src` folder and run:

`pip install -e ringdetector/`

Then, open `src/ringdetector/ringdetector/Paths.py` and set absolute path to the folder where you have the data saved. Subfolder structure as outlined in Paths.py will be created during execution.

## Pipeline

### Crop Detection
TBC

### Dataset Creation
Execute `python preprocessing/CreateInnerDataset.py`

### Core Prediction and Scoring
Execute `python analysis/pipeline.py` with optional args documented in `utils/ConfigArgs.py`. Quite slow for the moment, consider running for a single core first with `python analysis/pipeline.py -sample KunA01SS`, replacing the sample name with a sample in your inner dataset directory.

