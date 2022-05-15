# DSLabTreeRing

Ring Detector! Installation steps:

Navigate to `src` folder and run:

`pip install -e ringdetector/`

Then, open Paths.py and set absolute path to the folder where you have the data saved. Please make sure that folders have appropriate names, as defined in Paths.py.

### Dataset Creation
Execute `python preprocessing/CreateInnerDataset.py`

### Core Prediction and Scoring
Execute `python analysis/pipeline.py` with optional args documented in `utils/ConfigArgs.py`. Quite slow for the moment, consider running for a single core first with `python analysis/pipeline.py -sample KunA01SS`, replacing the sample name with a sample in your inner dataset directory.
