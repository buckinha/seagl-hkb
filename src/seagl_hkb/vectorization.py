"""vectorization.py: My primary URL-to-vector tools, and the URLVectorizer class"""

__author__ = "Hailey Buckingham"
__email__ = "hailey.k.buckingham@gmail.com"

import numpy as np
import pandas as pd
import os
import ipaddress
import scipy.sparse
from joblib import Parallel, delayed

URI_SCHEMES = [
        'http://',
        'https://',
        'ftp://',
        'file:',
        'mailto:',
        'telnet://'
    ]

URL_PATH_SEPARATORS = ['/', '?', '.', '=', '-', '_']

# some directory wrangling
this_dir = os.path.split(os.path.abspath(__file__))[0]
parent_dir = this_dir.rsplit(os.path.sep, 1)[0]

ARTIFACT_DIR = os.path.join(parent_dir, 'artifacts')
DATA_DIR = os.path.join(parent_dir, 'data')

PATH_DATA_TRAIN = os.path.join(DATA_DIR, 'data_train.csv')
PATH_DATA_TEST = os.path.join(DATA_DIR, 'data_test.csv')


def remove_scheme(url):
    """ Take the scheme section off of a url.
    I'm mainly doing this to allow easier splitting of path and host, by the first slash in the url"""
    for scheme in URI_SCHEMES:
        if url.startswith(scheme):
            url = url[len(scheme):]
    return url


def replace_path_seperators(path):
    """ Replaces a bunch of characters in the path section of a url, with spaces, to allow easy splitting"""
    for sep in URL_PATH_SEPARATORS:
        path = path.replace(sep, ' ')

    return path


def split_url(url):
    """ Removes the scheme, and then splits a url at its first slash. If there's no slash, then the whole url is assumed
    to be the host, and path is assigned as ''
    """
    clean_url = remove_scheme(url)
    if '/' in clean_url:
        host, path = clean_url.split('/', 1)
    else:
        host = clean_url
        path = ''
    return host, replace_path_seperators(path)


def host_is_like_ip_address(host: str) -> int:
    """
    Returns 1 if the provided string appears to be a valid IP address, 0 otherwise
    :return: 0 or 1
    """
    try:
        ip = ipaddress.ip_address(host)
        # no error, so that seems to be a valid ip address
        return 1
    except ValueError as e:
        return 0


def host_port_or_none(host: str) -> int:
    """
    Returns the integer of the host port, if one is present and -1 otherwise.  This was causing problems in my
    clustering experiments, since IP addresses are very large numbers compared to all the other features. Because of
    this, this function is not currently being used. A better way might be to binarize all the ports that are seen,
    and have one boolean feature for each one.

    :param host: string
        the host section of a url
    :return: 0 or 1
    """
    port = None
    if ':' in host:
        port_string = host.split(':')[-1]
        try:
            port = int(port_string)
        except Exception as e:
            pass

    return port


def get_constant_features(host: str, path: str):
    """
    This function does several counts, and a few other features.  In general, it is also easy to use a scikit
    CountVectorizer to get counts of characters and words, etc...

    Because I might add or subtract features from this function, it gets a little funny knowing how many
    features it will produce. For this reason, I made the get_constant_feature_count() function which just tries
    out this function, and reports how many features it sees.

    :param host: string
        the host portion of a url
    :param path: string
        the path portion of a url
    :return: np.array
    """
    features = [
        len(host.split('.')),                           # number of parts in host
        len(host),                                      # length of the host
        len(path),                                      # length of the path
        len(replace_path_seperators(path).split(' ')),  # number of path separators
        host_is_like_ip_address(host),                  # host appears to be an IP address
        # host_port_or_none(host) or -1,                  # host port as an integer, or a -1, if there isn't a port
    ]
    features.extend([len(path.split(sep)) for sep in URL_PATH_SEPARATORS])  # how many of each separator

    return np.asarray(features)


def get_constant_feature_count() -> int:
    """
    Returns the number of features that are produced by the get_constant_features() function
    :return: intege
    """
    v = get_constant_features('', '')
    return v.shape[0]


def vectorize_url(url: str, host_words_dict: dict, path_words_dict: dict,
                  n_host_words: int, n_path_words: int) -> scipy.sparse.csr_matrix:
    """

    :param url: string
        the url to be turned into a vector

    :param host_words_dict: dictionary
        a dictionary produced by build_word_dicts, representing the words found in the host portion of the urls

    :param path_words_dict: dictionary
        a dictionary produced by build_word_dicts, representing the words found in the path portion of the urls

    :param n_host_words: integer
        the length of the host_words_dictionary. Since this function is called a lot of times, passing the length once
        saves a lot of time, in lieu of having this function measure that length on every call

    :param n_path_words: integer
        the length of the path_words_dictionary.

    :return: sparse.csr_matrix of shape (1, n)
        where n is the number of features in the vector. N will be equal to the number of terms in both dictionaries,
        plus the count of features coming from the get_constant_features() function.
    """

    host, path = split_url(url)

    host_word_array = np.zeros((n_host_words, ))
    path_word_array = np.zeros((n_path_words, ))

    for word in host.split('.'):
        if word in host_words_dict:
            host_word_array[host_words_dict[word]] = 1

    for word in path.split(' '):
        if word in path_words_dict:
            path_word_array[path_words_dict[word]] = 1

    const_features = get_constant_features(host, path)
    v = np.hstack([const_features, host_word_array, path_word_array])
    return scipy.sparse.csr_matrix(v)


def vectorize_url_batch(url_list: [str], host_words_dict: dict, path_words_dict: dict,
                        n_host_words: int, n_path_words: int) -> scipy.sparse.csr_matrix:
    """
    This function takes a batch of urls as a list, and vectorizes each of them.

    :param url_list: list of strings

    :param host_words_dict: dictionary
        a dictionary produced by build_word_dicts, representing the words found in the host portion of the urls

    :param path_words_dict: dictionary
        a dictionary produced by build_word_dicts, representing the words found in the path portion of the urls

    :param n_host_words: integer
        the length of the host_words_dictionary. Since this function is called a lot of times, passing the length once
        saves a lot of time, in lieu of having this function measure that length on every call

    :param n_path_words: integer
        the length of the path_words_dictionary.

    :return: sparse.csr_matrix of shape (m, n)
        where m is the number of urls in url_list, and n is the number of features in each vector. N will be equal to
        the number of terms in both dictionaries, plus the count of features coming from the get_constant_features()
        function.
    """
    sparse_arrays = [vectorize_url(u,
                                   host_words_dict,
                                   path_words_dict,
                                   n_host_words,
                                   n_path_words) for u in url_list]
    return scipy.sparse.vstack(sparse_arrays)


def reverse_dict(d):
    """
    This function flip-flops a dictionary, so that the result has all the values from the first dict as keys, and the
    keys of the first dict are values. In this case, since our values are all unique (just indices), this does not
    cause any loss of data. I'm sure there are better ways of doing this, but it was expedient, and makes it really
    easy to look up the name of each feature, by index, in URLVectorizer.get_feature_name_for_idx()

    :param d: dictionary
    :return: dictionary, with keys/values reversed
    """
    r_d = {}
    for i, v in d.items():
        r_d[v] = i
    return r_d


class URLVectorizer:
    """
    This is my primary vectorizer class. It can load a dataset and build the host and path words dictionaries, and then
    holds onto those dictionaries to allow consistent vectorization. To use this class, do the following:

        import seagl_hkb.vectorization as vec
        my_vectorizer = vec.URLVectorizer()
        my_vectorizer.build_from_url_list(list_of_urls)

    Once this is done, you can call the vectorize() and vectorize_url() functions, etc...

    Alternatively, if you want to build a vectorizer directly from a csv of urls, you can use the helper functions
    that follow, namely:

        build_vectorizer_from_file(path_to_csv_file)

    """

    def __init__(self):
        self.host_words_dict = {}
        self.path_words_dict = {}
        self.n_host_words = 0
        self.n_path_words = 0
        self.n_constant_features = get_constant_feature_count()

    def _set_host_words_dict(self, d):
        """
        Ingests the host words dictionary output of the _build_word_dicts() function
        """
        self.host_words_dict = d
        self.host_reverse_dict = reverse_dict(d)
        self.n_host_words = len(d)

    def _set_path_words_dict(self, d):
        """
        Ingests the path words dictionary output of the _build_word_dicts() function
        """
        self.path_words_dict = d
        self.path_reverse_dict = reverse_dict(d)
        self.n_path_words = len(d)

    def build_from_url_list(self, url_list: [str]):
        """
        Takes a list of urls, and builds the vectorization scheme.

        :param url_list: list of strings, where each string should be a url of some kind
        :return: None
        """
        host_d, path_d = _build_word_dicts(url_list)
        self._set_host_words_dict(host_d)
        self._set_path_words_dict(path_d)

    @property
    def n_features(self):
        """
        The number of features that the vectors will have, when calling this object's vectorize and vectorize_url
        functions.
        :return:
        """
        return self.n_constant_features + self.n_host_words + self.n_path_words

    def vectorize_url(self, url) -> scipy.sparse.csr_matrix:
        """ Vectorize a single url
        This kust calls the global, vectorize_url() function, using the vectorizer's own host and path word dictionaries

        :param url: string, a url to vectorize

        :return: scipy.sparse.csr_matrix
        """
        return vectorize_url(url,
                             host_words_dict=self.host_words_dict,
                             path_words_dict=self.path_words_dict,
                             n_host_words=self.n_host_words,
                             n_path_words=self.n_path_words)

    def vectorize(self, x, n_jobs=4) -> scipy.sparse.csr_matrix:
        """
        Take a list of urls, and run the vectorize_url function on each of them

        :param x: a list or array of urls to vectorize

        :param n_jobs: integer, or None
            If set to None or 1, then this function will run single-threaded.
            If set to an integer greater than 1, then this function will invoke joblib to parallelize the vectorization
            With less then 500 or so urls, its faster to run single-threaded, due to the overhead of copying necessary
            host and path word dictionaries, etc... for parallelization

        :return: scipy.sparse.csr_matrix
        """

        if isinstance(x, str):
            return self.vectorize_url(x)

        if n_jobs is None or n_jobs == 1:
            sparse_arrays = [self.vectorize_url(u) for u in x]
        else:
            # split up the dataset into batches, based on the number of jobs, and run them in parallel. Note that
            # the large host and path word dictionaries need to be copied for each job, so there's non-trivial overhead
            # here. However, with the size of the dataset, this still helps speed things up.
            chunks = np.array_split(x, n_jobs)
            sparse_arrays = Parallel(n_jobs=n_jobs)(
                delayed(vectorize_url_batch)(
                    url_list=chunks[i],
                    host_words_dict=self.host_words_dict,
                    path_words_dict=self.path_words_dict,
                    n_host_words=self.n_host_words,
                    n_path_words=self.n_path_words
                ) for i in range(n_jobs)
            )

        return scipy.sparse.vstack(sparse_arrays)

    def get_feature_name_for_idx(self, idx):
        """
        Returns the name of the feature at a particular index of the vectorizer. Names will start with one of:
            ConstantFeature__N     for features that come from get_constant_features() and N is an integer
            HostWord__<word>       for words found in the host portion of the url and <word> is replaced by the word
            PathWord__<word>       for words found in the path portion of the url and <word> is replaced by the word
        :param idx:
        :return:
        """

        if idx >= self.n_features:
            raise ValueError('Index {} out of range for vectorizer with only {} features'.format(idx, self.n_features))
        if idx < self.n_constant_features:
            return 'ConstantFeature__{}'.format(idx)
        elif (idx - self.n_constant_features) < self.n_host_words:
            return 'HostWord__{}'.format(self.host_reverse_dict[idx - self.n_constant_features])
        elif idx - self.n_constant_features - self.n_host_words < self.n_path_words:
            return 'PathWord__{}'.format(self.path_reverse_dict[idx - self.n_constant_features - self.n_host_words])

    def print_feature_report_for_url(self, url):
        """
        Prints all the features that are 'active', that is, not just a zero, for the given url.
        Even though there are over 400k possible features in this dataset, any one url will likely only have a few of
        them.
        :param url:
        :return:
        """
        v = self.vectorize(url)
        for i in range(v.shape[0]):
            if v[i] != 0:
                print('index: {}, feature: {}, val: {}'.format(i, self.get_feature_name_for_idx(i), v[i]))


def _build_word_dicts(urls):
    """
    This function looks at a list of urls, and constructs a dictionary of all the 'words' in both the host and the path
    sections of the url.

    For the host section, a word is defined as one of the parts between '.' characters.
    For the path section, we split the path by any of a number of characters listed in constants.py, and treat anything
    in between those symbols as 'words'

    :param urls: a list of strings, where each string is a url
    :return: two dictionaries, where the keys are the words, and the values are an integer. If there are N words in one
    of the dictionaries, then the values of each of those words will be a unique integer from 0 to 99. This is useful
    later on when we want to vectorize a url using these dictionaries. The integer value of each key determines where
    in the vector this feature should go.
    """
    host_words = set()
    path_words = set()

    for u in urls:
        host, path = split_url(u)
        host_words.update(host.split(':')[0].split('.'))
        path_words.update(path.split(' '))

    host_d = {}
    count = 0
    for word in host_words:
        host_d[word] = count
        count += 1

    path_d = {}
    count = 0
    for word in path_words:
        path_d[word] = count
        count += 1

    return host_d, path_d


def build_vectorizer_from_file(filepath: str) -> URLVectorizer:
    """
    Constructs a vectorizer object by looking a the urls in a csv file
    The file should have a column labeled 'url'. Other columns will be ignored

    :param filepath: string; the path to a csv file
    :return: vectorizer object
    """
    df = pd.read_csv(filepath)
    my_vec = URLVectorizer()
    my_vec.build_from_url_list(df.url.tolist())
    print('Built Vectorizer with {} features'.format(my_vec.n_features))
    return my_vec


def build_vectorizer_from_file_list(filelist: [str]) -> URLVectorizer:
    """
    The same as build_vectorizer_from_file() except that it will form a vectorizer based on the urls found in several
    files. The urls in each file will simply be concatenated, so this method does not help save memory; it is only for
    convenience

    :param filelist: a list of strings, where each string is a filepath to a csv file
    :return: vectorizer object
    """

    df_list = []
    for f in filelist:
        df_list.append(pd.read_csv(f))
    df = pd.concat(df_list)

    my_vec = URLVectorizer()
    my_vec.build_from_url_list(df.url.tolist())
    print('Built Vectorizer with {} features'.format(my_vec.n_features))
    return my_vec
