import os
import scipy.sparse
import numpy as np
import pandas as pd
import pickle
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
import seagl_hkb.constants as constants

base_dir = constants.DATA_DIR
PATH_TRAIN_NPZ = os.path.join(base_dir, 'train_vectors.npz')
PATH_TEST_NPZ = os.path.join(base_dir, 'test_vectors.npz')
PATH_TRAIN_DF = os.path.join(base_dir, 'train_dataframe.pickle')
PATH_TEST_DF = os.path.join(base_dir, 'test_dataframe.pickle')


def cluster() -> None:
    train_vectors = scipy.sparse.load_npz(PATH_TRAIN_NPZ)
    test_vectors = scipy.sparse.load_npz(PATH_TEST_NPZ)
    train_df = pd.read_pickle(PATH_TRAIN_DF)
    test_df = pd.read_pickle(PATH_TEST_DF)

    kmeans_clusters = 20
    #cluster_alg = KMeans(n_clusters=kmeans_clusters)
    cluster_alg = DBSCAN(eps=4)
    #cluster_alg = AgglomerativeClustering(n_clusters=kmeans_clusters)  # note, requires a dense matrix

    data_size = -1
    # clusters = cluster_alg.fit(train_vectors[:data_size]
    clusters = cluster_alg.fit(train_vectors[:data_size].todense())

    #print(cluster_alg.labels_)
    examples_per_cluster = 10
    for i in range(kmeans_clusters):
        print('{} Examples from cluster: {}'.format(examples_per_cluster, i))
        examples = train_df.url[:data_size][cluster_alg.labels_==i][:examples_per_cluster].tolist()
        for e in examples:
            print("  {}".format(e))


if __name__ == "__main__":
    cluster()




