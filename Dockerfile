FROM tensorflow/tensorflow:r0.9rc0-devel
MAINTAINER Bob Adolf <rdadolf@gmail.com>

# Extra support software
RUN pip install --upgrade pip
RUN pip install --upgrade scipy
RUN pip install --upgrade nose
RUN pip install --upgrade mkdocs
RUN pip install --upgrade pylint

# Let's create a non-root user now
RUN useradd -ms /bin/bash nnmodel

# Grab and build nnmodel
WORKDIR /nnmodel
ADD . /nnmodel/
RUN chown -R nnmodel /nnmodel

USER nnmodel
ENV PYTHONPATH=/nnmodel/
