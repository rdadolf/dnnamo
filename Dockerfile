FROM ubuntu:16.04

MAINTAINER Bob Adolf <rdadolf@gmail.com>

RUN apt-get update && apt-get install -y \
  python \
  python-pip \
  git

RUN pip install --upgrade pip

# Install dependencies
ADD requirements.txt /tmp
RUN pip install --upgrade -r /tmp/requirements.txt

# Install model dependencies
ADD models/fathom_requirements.txt /tmp
RUN pip install --upgrade -r /tmp/fathom_requirements.txt

ADD models/dl_models_requirements.txt /tmp
RUN pip install --upgrade -r /tmp/dl_models_requirements.txt

# Add Dnnamo
ADD . /dnnamo
RUN chmod -R a+rw /dnnamo
