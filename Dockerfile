# start with a general ubuntu image, and then add python 3. That will give us a baseline from which to run all else.
FROM ubuntu:latest
FROM python:3.6

MAINTAINER Hailey Buckingham "hailey.k.buckingham@gmail.com"

# update ubuntu things, and add more python tools
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

# install our python dependencies
ADD requirements.txt /
RUN pip install -r requirements.txt

# copy our code and artifacts
COPY src /src

# add our project to the python path, so that when we unpickle objects and such, they can find the code they need.
ENV PYTHONPATH "${PYTHONPATH}:/src"

# open the port that flask will use
EXPOSE 5000

# define how to launch the microservice
WORKDIR /src/seagl_hkb
ENTRYPOINT ["python"]
CMD ["app_ml.py"]