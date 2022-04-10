import os

# Set absolute path to your own data folder.
DATA = '/Users/lukeasmi/Documents/ETHZ/dslabtreering/data'

GENERATED_DATASETS = os.path.join(DATA, "generated_datasets")
IMAGES = os.path.join(DATA, 'images/')
LABELME_JSONS = os.path.join(DATA, 'labelme_jsons/')
POINT_LABELS = os.path.join(DATA, 'point_labels/')
WIDTH_MEASUREMENTS = os.path.join(DATA, '/width_measurements')

# inner dataset
GENERATED_DATASETS_INNER = os.path.join(DATA, 'generated_datasets/inner/')
GENERATED_DATASETS_INNER_CROPS = os.path.join(
    GENERATED_DATASETS_INNER, "cropped_core_images")
GENERATED_DATASETS_INNER_PICKLES = os.path.join(
    GENERATED_DATASETS_INNER, "pickled_cores")
GENERATED_DATASETS_TEST_INNER = os.path.join(
    DATA, 'generated_datasets/test_inner')