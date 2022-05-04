#%% splits the labelme json into train val test folders
import os
import shutil
import logging
import coloredlogs
import warnings

from sklearn.model_selection import train_test_split

from ringdetector.Paths import LABELME_JSONS, POINT_LABELS
from ringdetector.preprocessing.ImageAnnotation import ImageAnnotation 

coloredlogs.install(level=logging.INFO)
warnings.filterwarnings("ignore")

#%%
seed = 42

files = os.listdir(LABELME_JSONS)

# split test 0.3 into 0.15 val 0.15 test
train, test = train_test_split(
    files, train_size=0.7, test_size=0.3, random_state= seed
)
val, test = train_test_split(
    test, train_size=0.5, test_size=0.5, random_state=seed
)

splits = [train, val, test]
names = ["train", "val", "test"]
coreCounts = []
totalCount = 0
for name, split in zip(names,splits):
    splitCounts = []
    for file in split:
        if file.endswith(".json"):
            imgAnno = ImageAnnotation(
                os.path.join(LABELME_JSONS, file), 
                POINT_LABELS
            )
            ncores = len(imgAnno.core_annotations)
            splitCounts.append(ncores)
            totalCount += ncores
    coreCounts.append(splitCounts)

logging.info(f"Train total cores: {sum(coreCounts[0])/totalCount}, "
    f"Val: {sum(coreCounts[1])/totalCount}, "
    f"Test: {sum(coreCounts[2])/totalCount}")
logging.info(f"Image sizes val: f{coreCounts[1]}")
logging.info(f"Image sizes test: f{coreCounts[2]}")

#%% need to put all fil
for idx in range(3):
    os.makedirs(os.path.join(LABELME_JSONS, names[idx]))
    for file in splits[idx]:
        shutil.copy(os.path.join(LABELME_JSONS, file), os.path.join(LABELME_JSONS, names[idx], file))
