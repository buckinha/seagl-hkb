import sklearn.cluster


def get_cluster_count(clustering_obj):
    cluster_ids = set(clustering_obj.labels_)
    return len(cluster_ids)


def get_cluster_ids(clustering_obj):
    return set(clustering_obj.labels_)


def get_examples_from_cluster(cluster_id, clustering_obj, data, n_examples=10):
    examples = data[clustering_obj.labels_ == cluster_id][:n_examples].tolist()
    return [str(e) for e in examples]


