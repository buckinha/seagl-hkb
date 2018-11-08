"""script_model_training.py:  My main script for loading and training the good/bad URL classifier"""

__author__ = "Hailey Buckingham"
__email__ = "hailey.k.buckingham@gmail.com"

import os
import scipy.sparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seagl_hkb.constants as constants
import seagl_hkb.utils as utils


PATH_TRAIN_NPZ = os.path.join(constants.DATA_DIR, 'train_vectors.npz')
PATH_TEST_NPZ = os.path.join(constants.DATA_DIR, 'test_vectors.npz')
PATH_TRAIN_DF = os.path.join(constants.DATA_DIR, 'train_dataframe.pickle')
PATH_TEST_DF = os.path.join(constants.DATA_DIR, 'test_dataframe.pickle')


def train_model(save_model=True):
    """ This function trains a random forest classifier on our data, and prints some statistics about its accuracy on
    the testing data. The task at hand is to train a classifier that can tell the difference between good and bad web
    URLs, based on the examples from the dataset provided (from www.kaggle.com/antonyj453/urldataset)

    In this case, the training/testing vectors have already been processed (See script_vectorize_datasets.py), and
    saved as .npz files.  This is a standard format for numpy arrays, and scipy.sparse arrays.

    The original urls, and the labels for each one, are saved in pandas dataframes (also by the vectorization script).
    These were saved using pandas to_pickle() method, and are being loaded by pandas read_pickle() method.

    The RandomForest Classifier is an easy and quick classifier to begin playing around with. Another good option for
    starting with would be a LogisticRegression classifier, which can be imported from the sklearn.linear_model module
    """

    # load the training and testing data.
    print('loading data')
    n = -1  # set this to -1 to use all the data. I use something small, like 500, to speed things up when i'm testing
    train_vectors = scipy.sparse.load_npz(PATH_TRAIN_NPZ)[:n]
    test_vectors = scipy.sparse.load_npz(PATH_TEST_NPZ)[:n]
    train_df = pd.read_pickle(PATH_TRAIN_DF)[:n]
    test_df = pd.read_pickle(PATH_TEST_DF)[:n]

    # instantiate a new classifier, and fit it to the data, using the .fit() method
    print('training classifier')
    clf = RandomForestClassifier(n_estimators=15)
    clf.fit(X=train_vectors, y=train_df.label)

    # get some predictions from the classifier. We'd hope that these are mostly
    # correct. If not, the classifier is having a hard time getting anywhere with it's current settings and the current
    # data
    predictions = clf.predict(X=train_vectors)

    # Use a handy pandas function to print out how our classifier did. The crosstab method will show us how many
    # times the classifier got each label right, and how many times it got each label wrong.
    print('Accuracy on training samples')
    print(pd.crosstab(train_df.label, predictions, rownames=['True'], colnames=['Predicted'], margins=True))

    # now lets see how the classifier does on the test data. Remember that the classifier has only seen the training
    # data so far, so it's never used any of the information in the test_vectors to learn. This is an excellent way to
    # see how good our classifier really is
    predictions = clf.predict(X=test_vectors)

    # lets print out another crosstab report
    print()
    print('Accuracy on Test Data')
    print(pd.crosstab(test_df.label, predictions, rownames=['True'], colnames=['Predicted'], margins=True))

    # if we want to use the model again, we can just pickle it. Later on, we can unpickle it, and continue calling its
    # .predict() function for new data. This is handy way to give a model to other researchers and engineers.
    if save_model:
        utils.write_model_to_compressed_pickle(clf)


if __name__ == '__main__':
    train_model()


