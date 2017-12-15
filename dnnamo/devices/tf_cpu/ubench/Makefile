DEVICE_NAME:=tf_cpu
UBENCH_COMMON:=../../ubench_common
include $(UBENCH_COMMON)/common_defs.mk

SYS_TF_DIR=$(shell python -c 'import tensorflow as tf;import os.path;print os.path.split(tf.__file__)[0]')
EIGEN_ARCHIVE=$(SYS_TF_DIR)/include/external/eigen_archive/*/
#EIGEN_ARCHIVE=../../../../install/tensorflow/external/eigen_archive/*/

EIGEN_HEADERS:=-I$(wildcard $(EIGEN_ARCHIVE))
ifeq ("$(EIGEN_HEADERS)","-I")
$(error No eigen headers found in $(shell pwd)/$(EIGEN_ARCHIVE))
endif
override CPPFLAGS+=-O3 -fpic -std=c++11 $(EIGEN_HEADERS) -Wno-deprecated


# Default make rule (amongst other things) defined in here:
include $(UBENCH_COMMON)/common_rules.mk


$(DEVICE_NAME).o: $(DEVICE_NAME).cpp
