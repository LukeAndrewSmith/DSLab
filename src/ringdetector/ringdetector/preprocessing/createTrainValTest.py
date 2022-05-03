# splits the labelme json into train val test folders
import os
from sklearn.model_selection import train_test_split
import random
import shutil
from ringdetector.Paths import LABELME_JSONS
seed = 42

files = os.listdir(LABELME_JSONS)

train, test = train_test_split(files, train_size=0.7, test_size=0.3, seed=seed)
# split test 0.3 into 0.15 val 0.15 test
val, test = train_test_split(test, train_size=0.5, test_size=0.5, seed=seed)

# need to put all fil
split = [train, val, test]
name = ["train", "val", "test"]
for idx in range(3):
    os.makedirs(os.path.join(LABELME_JSONS, name[idx]))
    for file in split[idx]:
        shutil.copy(os.path.join(LABELME_JSONS, file), os.path.join(LABELME_JSONS, name[idx], file))
