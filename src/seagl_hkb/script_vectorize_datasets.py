"""script_vectroize_datasets.py: This is my data preparation script."""

__author__ = "Hailey Buckingham"
__email__ = "hailey.k.buckingham@gmail.com"

import pandas as pd
import os
import sklearn.utils
import seagl_hkb.vectorization as vectorization
import scipy.sparse
import time
import pickle
import seagl_hkb.constants as constants


def main():
    """
    This function loads the raw data, builds the vectorizer that we'll be using based on what's in the data file, and
    then saves all the vectors and pandas dataframes back to disk for later use. It's nice to save the dataframes
    because if we do anything to the raw data in this function (like shuffling), then we'd have to make sure to
    replicate those steps every time we loaded. By saving the dataframes AFTER those modifications, we don't have to
    duplicate the loading steps, and get consistency.
    :return:
    """

    start_time = time.time()

    # first we'll start by building a vectorizer object. In this library, it was amusing to me to build my own, instead
    # of using one of the excellent options available in scikit-learn. As part of that exercise, I ran into a bunch of
    # interesting caveats that I had not thought about before. I highly recommend doing these sorts of exercises, as
    # they'll really help you understand and appreciate the frameworks like sklearn, and even more so, will make you
    # more productive when using those frameworks, since you'll go in with a much deeper understanding of what's going
    # on under the covers.

    # In any case, this vectorizer object is responsible for taking raw urls as strings, and turning them into some
    # kind of array of numbers (usually a numpy array, or scipy.sparse array), which the ML tools can work with.
    print('building vectorizer')
    vectorizer = vectorization.build_vectorizer_from_file(constants.PATH_DATA_TRAIN)

    # save the vectorizer for later use
    pickle.dump(vectorizer,
                open(os.path.join(constants.ARTIFACT_DIR, 'vectorizer.pickle'), 'wb')
                )

    print('loading data frames')
    # here, i'm shuffling the data after I load it. This is because the first part of the data files are all the bad
    # urls, and then all the benign urls afterward. If we don't shuffle, and then take a slice of the data, we'd be
    # likely to get all of one kind of url or all of the other. Shuffling insures that we get a nice mix.
    df_train = sklearn.utils.shuffle(pd.read_csv(constants.PATH_DATA_TRAIN))
    df_test = sklearn.utils.shuffle(pd.read_csv(constants.PATH_DATA_TEST))

    # now we're actually using the vectorizer do make vectors from the loaded data. The n_jobs parameter is asking the
    # vectorizer to run in parallel, to save some time. Watch your CPU temp rise :D
    print('vectorizing data')
    X_train = vectorizer.vectorize(df_train.url, n_jobs=6)
    X_test = vectorizer.vectorize(df_test.url, n_jobs=6)

    print('Saving vector and dataframes')
    # save the vectors for later use when we build the model, so that we don't have to wait for vectorization again
    scipy.sparse.save_npz(file=os.path.join(constants.DATA_DIR, 'train_vectors.npz'), matrix=X_train)
    scipy.sparse.save_npz(file=os.path.join(constants.DATA_DIR, 'test_vectors.npz'), matrix=X_test)

    # save the dataframes, for ease of use later on (and to make sure we don't have to keep shuffling them)
    pd.to_pickle(df_train, os.path.join(constants.DATA_DIR, 'train_dataframe.pickle'))
    pd.to_pickle(df_test, os.path.join(constants.DATA_DIR, 'test_dataframe.pickle'))

    elapsed = int(time.time() - start_time)
    print('finished in {} seconds'.format(elapsed))


if __name__ == '__main__':
    main()