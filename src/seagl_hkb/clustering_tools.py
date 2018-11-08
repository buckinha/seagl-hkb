"""
clustering_tools.py: just a few helper functions for working with clusters. I might add a few more, for things like
sorting clusters by size, etc...

"""

__author__ = "Hailey Buckingham"
__email__ = "hailey.k.buckingham@gmail.com"

def get_cluster_count(clustering_obj):
    cluster_ids = set(clustering_obj.labels_)
    return len(cluster_ids)


def get_cluster_ids(clustering_obj):
    return set(clustering_obj.labels_)


def get_examples_from_cluster(cluster_id, clustering_obj, data, n_examples=10):
    examples = data[clustering_obj.labels_ == cluster_id][:n_examples].tolist()
    return [str(e) for e in examples]


