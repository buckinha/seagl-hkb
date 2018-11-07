#FROM python:3.6-alpine
FROM ubuntu:latest
FROM python:3.6

MAINTAINER Hailey Buckingham "hailey.k.buckingham@gmail.com"

# update ubuntu things
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

# install our python dependencies
ADD requirements.txt /
RUN pip install -r requirements.txt

# copy our code and artifacts
COPY src /src
ENV PYTHONPATH "${PYTHONPATH}:/src"

# open the port that flask will use
EXPOSE 5000

WORKDIR /src/seagl_hkb
ENTRYPOINT ["python"]
CMD ["app_ml.py"]