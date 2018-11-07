# seagl-hkb

Welcome to my SeaGL 2018 talk codebase. This code encompasses two talks, entitled:
* Intro to Machine Learning with Scikit-Learn
* Easy Microservices with Python, Flask, and Docker

Let me know if you have questions or comments!

My name is Hailey BuckinghamI'm available
* twitter: @HKBuckingham
* slack: PDX Women In Tech, Women Who Code Portland


## Overview

This code encompasses two tasks. The first is a machine learning exercise that attempts to build a classifier that can
tell the difference between web URLs that lead to regular websites and those that lead to malicious websites.

The second, is a microservice that hosts the model we build in the first task. The microservice will open several REST
routes that can accept URLs, and will respond with whether the ML classifier things the URL is good or bad.



## Guide

#### Scikit-learn Talk

The main files to look at, relating to my talk on learning to use scikit-learn, are:
* src/seagl_hkb/script_model_training.py - this is the main script I use to demonstrate model training
* src/seagl_hkb/script_clustering.py - this is the script I use to demonstrate clustering with DBSCAN, etc..
* src/seagl_hkb/script_vectorize_datasets.py - this is the script I used before my talk to prep the data

#### Microservices Talk

The microservice I demo in the talk uses the products of the machine learning talk, so I decided to keep them together
for ease. In general, I would probably separate these out into separate python packages, if I were doing this in
production

The main files to look at are:
* src/seagl_hkb/app_hello.py - this is the basic hello-world flask app
* src/seagl_hkb/app_ml.py - this is the microservice I demo during the talk
* src/seagl_hkb/tst_app_ml.py - this is a quick test script to show that the microservice is live, locally
* Dockerfile - this is the docker file that does the magic to create a reusable docker image


#### Installation

These instructions are for linux and mac OSX environments. I don't have a Windows machine to test this out with.

Once you've cloned the github repo, you can use the included Makefile to build the python virtualenv environment. If
you're unfamiliar with virtualenv, definitely take a look; it's a really handy way to set up python dependencies
without touching your system python, which can be pretty helpful

To run the make file, got to the repo root directory and run this command:

```
make env
```

To activate the virtualenv, run this command:

```
source env/bin/activate
```

You can then run the various python apps directly, by moving into the src/ folder and issuing commands like:

```
python -m seagl_hkb.app_ml
```

```
python -m seagl_hkb.script_vectorize_datasets
```

etc...


#### Docker Operations

First, you'll need to install docker. Head on over to https://www.docker.com/ to find out more.

Once docker is installed, you should be able to run the following commands, to build and run the machine learning microservice.

Build the Docker Image:

```
docker build -t seagl-hkb:latest .
```

Run the microservice (assuming the image built successfully)

```
docker run -p 5000:5000 seagl_hkb
```

You can kill the service by pressing ctrl-c

To see if the service is working, use a web browser to navigate to 127.0.0.1:5000

Alternatively, you can use the shortcuts I build in to my Makefile to run these commands:

```
make docker-build
make docker-run
```


## Dataset

The data set I'm using is from https://www.kaggle.com/antonyj453/urldataset, and appears to be  licensed under the
Open Database Licence here: https://opendatacommons.org/licenses/odbl/1.0/

I've sliced it up a bit for training and testing, etc...


