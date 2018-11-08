"""test_app_ml.py
This file is mainly to test whether our model microservice is running. In particular, this lets us test the route that
accepts json in the GET, since that's hard to do in a browser.

These functions will fail if the service is not running locally. See the local urls in each or the request.get()
functions.
"""

__author__ = "Hailey Buckingham"
__email__ = "hailey.k.buckingham@gmail.com"

import requests


def test_classification_with_json():
    """
    form up a json GET query, and see what the response looks like for a url that should be okay
    :return:
    """
    response = requests.get(url='http://127.0.0.1:5000/app_ml/v1/classify_url_from_json/',
                            json={
                                "url": "www.yahoo.com",
                                'request_id': 1234}
                            )
    if response.ok:
        print(response.json())
    else:
        print('failed')


def test_classification_with_json_bad():
    """
    form up a json GET query and see what the response looks like for what should be a malicious url
    :return:
    """
    response = requests.get(url='http://127.0.0.1:5000/app_ml/v1/classify_url_from_json/',
                            json={
                                "url": "botnet:6000",
                                'request_id': 5678}
                            )
    if response.ok:
        print(response.json())
    else:
        print('failed')


if __name__ == '__main__':
    print('GOOD URL RESPONSE:')
    test_classification_with_json()

    print()
    print('BAD URL RESPONSE:')
    test_classification_with_json_bad()