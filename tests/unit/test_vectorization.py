import unittest
import scipy.sparse
import numpy as np
import seagl_hkb.constants as constants
import seagl_hkb.vectorization as vectorization


class TestVectorizationHelpers(unittest.TestCase):

    def setUp(self):
        self.path = '/yay.html?SOMETHING/idontknow_whatever-this=nothing_i_guess'
        self.base_url = 'something.whereever.com' + self.path

    def test_remove_scheme(self):

        for scheme in constants.URI_SCHEMES:
            self.assertNotIn(scheme, vectorization.remove_scheme(scheme + self.base_url))

        for scheme in ['not', 'real', 'schemes://']:
            self.assertIn(scheme, vectorization.remove_scheme(scheme + self.base_url))

    def test_replace_path_seperators(self):

        for seperator in constants.URL_PATH_SEPARATORS:
            self.assertNotIn(seperator, vectorization.replace_path_seperators(self.path))


class TestVectorize(unittest.TestCase):

    def setUp(self):
        self.path = '/yay.html?SOMETHING/idontknow_whatever-this=nothing_i_guess'
        self.base_url = 'something.whereever.com:5' + self.path
        self.url = 'http://' + self.base_url

        self.host_words_dict = {
            'something': 0,
            'whereever': 1,
            'com': 2,
            'net': 3
        }
        self.path_words_dict = {
            'yay': 0,
            'idontknow': 1,
            'blahblahblah': 2
        }

    def test_constant_feature_count(self):
        n = vectorization.get_constant_feature_count()
        self.assertGreater(n, 0)
        print(n)

    def test_vectorize(self):
        v = vectorization.vectorize_url(self.url,
                                        self.host_words_dict,
                                        self.path_words_dict,
                                        len(self.host_words_dict),
                                        len(self.path_words_dict))

        n_features = vectorization.get_constant_feature_count() + len(self.host_words_dict) + len(self.path_words_dict)

        #self.assertIsInstance(v, np.ndarray)
        self.assertTrue(scipy.sparse.issparse(v))
        self.assertEqual(v.shape[1], n_features)
        print(v)

    def test_vectorizer_with_one_url(self):
        my_vec = vectorization.URLVectorizer()
        my_vec._set_host_words_dict(self.host_words_dict)
        my_vec._set_path_words_dict(self.path_words_dict)
        self.assertEqual(len(self.host_words_dict), my_vec.n_host_words)
        self.assertEqual(len(self.path_words_dict), my_vec.n_path_words)
        v = my_vec.vectorize_url(self.url)
        self.assertEqual(v.shape[1], my_vec.n_features)
        print(v)

    def test_vectorizer_with_many_urls(self):
        my_vec = vectorization.URLVectorizer()
        my_vec._set_host_words_dict(self.host_words_dict)
        my_vec._set_path_words_dict(self.path_words_dict)
        self.assertEqual(len(self.host_words_dict), my_vec.n_host_words)
        self.assertEqual(len(self.path_words_dict), my_vec.n_path_words)

        n_copies = 10
        v = my_vec.vectorize([self.url for i in range(n_copies)])
        self.assertEqual(v.shape[1], my_vec.n_features)
        self.assertEqual(v.shape[0], n_copies)
        print(v)


class TestDictBuilding(unittest.TestCase):

    def setUp(self):
        self.urls = [
            'www.google.com/home',
            'http://bad.site.com/username?something'
        ]

    def test_build_dicts(self):
        host_words, path_words = vectorization._build_word_dicts(self.urls)

        expected_host_words = ['www', 'google', 'com', 'bad', 'site']
        for w in expected_host_words:
            self.assertIn(w, host_words)

        expected_path_words = ['username', 'something', 'home']
        for w in expected_path_words:
            self.assertIn(w, path_words)

    def test_vectorization_from_build_dicts(self):
        host_words, path_words = vectorization._build_word_dicts(self.urls)
        my_vec = vectorization.URLVectorizer()
        my_vec._set_host_words_dict(host_words)
        my_vec._set_path_words_dict(path_words)
        v = my_vec.vectorize_url(self.urls[0])
        print(v.shape)
        self.assertEqual(v.shape[1], my_vec.n_features)
        print(v)