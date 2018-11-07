import os

# some standard values that we don't want to have to copy and paste in multiple places
URL_PATH_SEPARATORS = ['/', '?', '.', '=', '-', '_']
URI_SCHEMES = [
    'http://',
    'https://',
    'ftp://',
    'file:',
    'mailto:',
    'telnet://'
]

# some directory wrangling for our other files to use (so that we don't have to do this over and over)
this_dir = os.path.split(os.path.abspath(__file__))[0]
parent_dir = this_dir.rsplit(os.path.sep, 1)[0]

ARTIFACT_DIR = os.path.join(parent_dir, 'artifacts')
DATA_DIR = os.path.join(parent_dir, 'data')

PATH_DATA_TRAIN = os.path.join(DATA_DIR, 'data_train.csv')
PATH_DATA_TEST = os.path.join(DATA_DIR, 'data_test.csv')

MODEL_PATH = os.path.join(ARTIFACT_DIR, 'model.pickle.gzip')
