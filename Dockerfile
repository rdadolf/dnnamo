# For Harvard folks, import from this:
#FROM hubris.int.seas.harvard.edu/harvardacc/clusterbase:latest
#  or a local copy ("./build.sh -m local" from a repo clone)
FROM harvardacc/clusterbase:latest
# For everyone else, import from this:
#FROM ubuntu:16.04

MAINTAINER Bob Adolf <rdadolf@gmail.com>

RUN apt-get update && apt-get install -y \
  python \
  python-pip \
  git

# Make pip happy about itself
RUN pip install --upgrade pip
# Unlike apt-get, upgrating pip does not change which package gets installed,
# (since it checks pypi everytime regardless) so it's okay to cache pip.

# Get TensorFlow
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl

# Get a few support libraries
RUN pip install --upgrade scipy
RUN pip install --upgrade matplotlib
RUN pip install --upgrade mkdocs
RUN pip install --upgrade pylint
RUN pip install --upgrade nose
RUN pip install --upgrade pytest

