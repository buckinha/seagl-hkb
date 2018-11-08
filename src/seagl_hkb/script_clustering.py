"""script_clustering.py:  My main script for playing with clustering algorithms"""

__author__ = "Hailey Buckingham"
__email__ = "hailey.k.buckingham@gmail.com"

import os
import scipy.sparse
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import DBSCAN, KMeans
import seagl_hkb.constants as constants
import seagl_hkb.clustering_tools as clustering_tools

base_dir = constants.DATA_DIR
PATH_TRAIN_NPZ = os.path.join(base_dir, 'train_vectors.npz')
PATH_TEST_NPZ = os.path.join(base_dir, 'test_vectors.npz')
PATH_TRAIN_DF = os.path.join(base_dir, 'train_dataframe.pickle')
PATH_TEST_DF = os.path.join(base_dir, 'test_dataframe.pickle')


def cluster() -> None:
    """
    Clustering on sparse text data is somewhat difficult, and as such, this is the roughest portion of this demo.
    Still, it's easy to see how to swap in different scikit clustering algorithms to see how they work
    :return:
    """

    # load the vectors and dataframes. We'll use the vectors for clustering but can check the corresponding urls and
    # labels in the dataframes, if we like.

    n = 5000  # set this to -1 to use all the data. It'll take a while, if you do that though.
    train_vectors = scipy.sparse.load_npz(PATH_TRAIN_NPZ)[:n]
    test_vectors = scipy.sparse.load_npz(PATH_TEST_NPZ)[:n]
    train_df = pd.read_pickle(PATH_TRAIN_DF)[:n]
    test_df = pd.read_pickle(PATH_TEST_DF)[:n]

    # kmeans clustering
    #cluster_alg = KMeans(n_clusters=40)

    # DBSCAN clustering
    cluster_alg = DBSCAN(eps=4)

    cluster_alg.fit(train_vectors)

    cluster_ids = clustering_tools.get_cluster_ids(cluster_alg)

    examples_per_cluster = 10
    for i in cluster_ids:
        print('{} Examples from cluster: {}'.format(examples_per_cluster, i))
        examples = clustering_tools.get_examples_from_cluster(cluster_id=i,
                                                              clustering_obj=cluster_alg,
                                                              data=train_df.url[:n],
                                                              n_examples=examples_per_cluster)
        for e in examples:
            print("  {}".format(e))


if __name__ == "__main__":
    cluster()




