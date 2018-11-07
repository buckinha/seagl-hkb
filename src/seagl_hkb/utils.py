import pickle
import gzip
import os
import seagl_hkb.constants as constants


def write_model_to_compressed_pickle(model):

    with gzip.open(constants.MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)


def read_model_from_compressed_pickle():

    with gzip.open(constants.MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

