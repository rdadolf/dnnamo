FROM tensorflow/tensorflow:r0.9rc0-devel-gpu
MAINTAINER Bob Adolf <rdadolf@gmail.com>

# Workaround for Tensorflow soname problems (tf #2865)
RUN ln -s /usr/local/nvidia/lib64/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so

# Extra support software
RUN pip install --upgrade pip
RUN pip install --upgrade nose
RUN pip install --upgrade mkdocs
RUN pip install --upgrade pylint

# Let's create a non-root user now
RUN useradd -ms /bin/bash nnmodel
USER nnmodel

# Grab and build nnmodel
WORKDIR /nnmodel
ADD . /nnmodel/
