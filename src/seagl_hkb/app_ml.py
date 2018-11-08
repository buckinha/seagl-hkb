"""
app_ml.py: This is the Flask application that will be running in Docker. In general, it's a LOT better to use NGINX or gunicorn, or
other such webserver applications instead of the test server that comes bundled with Flask. However, each of those
will have different Docker setups, depending on what you're after. For simplicity, I decided to go with the flask server
here.
"""

__author__ = "Hailey Buckingham"
__email__ = "hailey.k.buckingham@gmail.com"

from flask import Flask, request, jsonify
import pickle
import time
import os
import seagl_hkb.constants as constants
import seagl_hkb.utils as utils

# make the flask application object
app_ml = Flask(__name__)

# load the ML model and vectorizer. We only want to do this once, since they are big objects, and take a bit to load.
ml_model = utils.read_model_from_compressed_pickle()

vectorizer_path = os.path.join(constants.ARTIFACT_DIR, 'vectorizer.pickle')
with open(vectorizer_path, 'rb') as vec_in:
    ml_vectorizer = pickle.load(vec_in)


# Below are all the routes into our service. Each one links a decorator, which describes the url route and variables, to
# a python function with the same variables. Whatever the python function returns will be returned by the webserver to
# the caller. In this case, i'm using Flask's jsonify() function, to make sure everything is formatted consistently on
# the return

@app_ml.route('/', methods=['GET'])
@app_ml.route('/app_ml/v1/', methods=['GET'])
def list_routes():
    """
    There are two routes into this function. One is just the '/', and the other at 'app_ml/v1'  They both return a json
    dictionary showing the rest of the routes I'm defining in this file. Note that these are not automatic, I have to
    update this return dict myself, to reflect any changes, additions, or deletions to the other routes in the file
    :return:
    """
    return jsonify({
        'health check':                        '/app_ml/v1/health_check/',
        'url classification':                  '/app_ml/v1/classify_url/<url>/',
        'ul classification with json payload': '/app_ml/v1/classify_url_from_json/'
    })


@app_ml.route('/app_ml/v1/health_check/', methods=['GET'])
def health_check():
    """
    Health checks are an important way to make sure your microservices are available when you think they are. Monitoring
    systems can be pointed at these health check routes to periodically ping them. If they ever fail to respond, you'll
    get alerts! It's handy!
    :return:
    """
    return jsonify(True)


@app_ml.route('/app_ml/v1/classify_url/<url>/', methods=['GET'])
def classify_url(url):
    """
    This route will take a variable right in the url itself. Whatever is typed in the <url> section, will be handed in
    to the python function as the url variable. However, and this is really important: Urls often have '/' characters in
    them. But if you add more slashes to your url, flask will interpret that as a differnt route entirely, and will
    return a 404 Not Found error. That's bad, and illustrates the limits of passing some types of data as variables in
    the url as I've done here.

    :param url: string
        the url to be classified
    :return: dictionary (json)
        of the form:
            {'url': whatever-you-typed,
             'classification': either 'good' or 'bad', depending on what the model says
             'timestamp': the current system time from the server, for logging or whatever.
             }
    """
    # get the classification
    vector = ml_vectorizer.vectorize_url(url)
    classification = str(ml_model.predict(vector)[0])

    # send it back
    return jsonify({
        'url': url,
        'classification': classification,
        'timestamp': time.time()
    })


@app_ml.route('/app_ml/v1/classify_url_from_json/', methods=['GET', 'POST'])
def classify_url_from_json():
    """
    This function expects that the GET or POST request will have a json payload from which to extract the URL to be
    classified. THis is handy, since we will not be limited as to what characters the url can have, as we are in the
    function above.

    Since we're taking a json payload anyway, why not ask for a request ID as well. This will make it much easier to
    track down problems in our microservice architecture, since the request ID can theoretically be passed to and from
    each service, and allow you to recreate the sequence of events during debugging.

    :return:
    """
    content = request.json
    url = content.get('url', None)
    request_id = content.get('request_id', None)

    # get the classification
    vector = ml_vectorizer.vectorize_url(url)
    classification = str(ml_model.predict(vector)[0])

    # a little logging
    print('Request ID: {}, URL: {}, Classification: {}'.format(request_id, url, classification))

    # return a json response
    return jsonify({
        'url': url,
        'classification': classification,
        'request_id': request_id,
        'timestamp': time.time()
    })


if __name__ == '__main__':
    app_ml.run(debug=True, host='0.0.0.0')


