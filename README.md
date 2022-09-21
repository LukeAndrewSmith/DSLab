# DSLabTreeRing

Project for the Data Science Lab @ ETHZ. We investigated the automatic detection of tree rings from images, please see the file 'DSLab_TreeRings.pdf' for more details.

Warning: the final CNN used is too large to fit on github, it should be found in 'models/model_final.pth', hence if you'd actually like to run this code yourself please contact me and I'll happily share the file with you.

# HOW TO RUN THE FULL PIPELINE AS A USER

## STEP 1: Install Docker: 
Please follow the steps [here](https://docs.docker.com/get-docker/) to install docker for your operating system

## STEP 2: Build the docker image: 
Once docker is installed, you need to build the docker image of this application. 
For this open a terminal and navigate to the dslabtreering folder
then run the following command: (this can take a couple of minutes) 
`docker build -t dslabtreering .`
NOTE: You have to build this image only once, if you use the application multiple times, this has not to be repeated

## STEP 3: Give resources
The application is computationally quite expensive. To make sure docker doesn't stop the process, pleae open 
Docker Desktop and go to Settings > Resources > Advanced and give generous resource levels for Memory, SWAP and CPUs.
The Docker container will need access to your data folder which should also be located inside the dslabtreering folder. 
For this, please go to Settings > Resources > File Sharing and make sure the path to the data folder is included, if not add it. 

## STEP 4: Check folder structure
The application is assuming a folder structure, where there is a data folder (commonly located inside the dslabtreering directory)
which has the following subfolders: 
-images : put the images to analyze here
-results: you can expect the output here
-core_lists: put the csv file with the core names here. The csv file should have the same name as the corresponding .jpg 

example: 
```bash
└── dslabtreering
    ├── data
    │   ├── images            # put the images to analyze here
    │   ├── results           # the results will be stored here
    │   └── core_lists        # put the csvs with the core names here
    │    # everything below is part of the repository already
    ├── models  
    │   └── model_final.pth  
    └── src
        └── ...
```

## STEP 5: Run the docker container:
Once the image is created you can start up the environment with all your needs: 
`docker run -it --volume {ABSOLUTE PATH TO THE DATA FOLDER ON YOUR MACHINE}:/dslabtreering/data:rw dslabtreering python3 finalPipeline.py -imagePath /dslabtreering/data/images/{PUT THE IMAGE NAME HERE} -savePath /dslabtreering/data/results`

# Further documentation to use this repository

## Installation steps:

### Anaconda

Create venv using yaml config: `conda env create -n venv python=3.9.12 -f env_config.yaml`

You will receive a pip error stating that detectron2 can't be installed. After creation, activate the env and run `python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html` (documented here: https://detectron2.readthedocs.io/en/latest/tutorials/install.html)


### After environment creation

Navigate to `src` folder and run:

`pip install -e ringdetector/`

Then, open `src/ringdetector/ringdetector/Paths.py` and set absolute path to the folder where you have the data saved. Subfolder structure as outlined in Paths.py will be created during execution.

## Pipeline

### Dataset Creation
Execute `python preprocessing/CreateInnerDataset.py`

### Crop Detection Training: 
#### Crop Detection  toggle cracks if you want to also train a crack detector
#### dataMode specifies whether the outer or inner rectangle is taken as a bounding box + segmentation mask 
#### dataMode outerInner has the outer rectangle and inner segmentation mask
Execute `python cropdetection/main.py --mode=train --dataMode=inner --num-gpus=2 --cracks`

### Core Prediction and Scoring
Execute `python analysis/pipeline.py` with optional args documented in `utils/ConfigArgs.py`. Quite slow for the moment, consider running for a single core first with `python analysis/pipeline.py -sample KunA01SS`, replacing the sample name with a sample in your inner dataset directory.


