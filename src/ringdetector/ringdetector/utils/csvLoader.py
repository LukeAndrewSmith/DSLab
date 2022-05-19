import os, csv

from ringdetector.Paths import CORE_LISTS

# TODO request: @freddy you need to handle the FNs in the prediction so that we can correctly align the names w/ the cores (I think it can be a part of the crop detection heuristic ticket)
# NOTE: you need to (write code somewhere else to) assert that the csvPath matches the imgPath before calling the func.
def loadImageCSV(imageName):
    """ Loads info from manual input json at the start of the inference workflow

    :param imageName: name of the original scanned image e.g. KunA10_11_14
    :return core_names:list of core name strings
    :return start_years: list of start year integers
    """

    core_names = []
    start_years = []
    correct_header = ['CORE_NAMES', 'START_YEAR']
    csv_path = os.path.join(CORE_LISTS, imageName + ".csv")

    with open(csv_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        
        header = next(csv_reader)
        assert header == correct_header, f'CSV file header assertion failed. Did you name and order your header properly with {correct_header}?'

        for index, row in enumerate(csv_reader):
            core_name, start_year = row

            assert (core_name and start_year), f'NaN values is not allowed! Check row {index + 1}.'

            core_names.append(core_name.strip())
            start_years.append(int(start_year.strip()))
    
    return core_names, start_years