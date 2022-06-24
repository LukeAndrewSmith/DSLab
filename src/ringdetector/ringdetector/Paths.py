import os

# Set absolute path to your own data folder.
# the data folder should be relative from dslabtreering/src/ringdetector/ringdetector
DATA = '../../../data'

# Raw data paths
IMAGES = os.path.join(DATA, 'images/')
LABELME_JSONS = os.path.join(DATA, 'labelme_jsons/')
POINT_LABELS = os.path.join(DATA, 'point_labels/')
WIDTH_MEASUREMENTS = os.path.join(DATA, 'width_measurements')
CORE_LISTS = os.path.join(DATA, 'core_lists') # directory for prediction csvs

# saved models
MODELS = os.path.join(DATA, '..', 'models')
CROP_MODEL = os.path.join(MODELS, 'model_final.pth')

# inner dataset
GENERATED_DATASETS = os.path.join(DATA, "generated_datasets")
GENERATED_DATASETS_INNER = os.path.join(DATA, 'generated_datasets/inner')
GENERATED_DATASETS_INNER_CROPS = os.path.join(
    GENERATED_DATASETS_INNER, "cropped_core_images")
GENERATED_DATASETS_INNER_PICKLES = os.path.join(
    GENERATED_DATASETS_INNER, "pickled_cores")
GENERATED_DATASETS_TEST_INNER = os.path.join(
    DATA, 'generated_datasets/test_inner')

# ring results
RESULTS = os.path.abspath(os.path.join(DATA, 'results'))
RESULTS_PKL = os.path.join(RESULTS, "processed_cores")
RESULTS_POS = os.path.join(RESULTS, "pos")
RESULTS_DIAG = os.path.join(RESULTS, "diag")

# crop results
D2_RESULTS = os.path.abspath(os.path.join(DATA, '..', 'd2_results'))
