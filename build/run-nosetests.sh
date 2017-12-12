#!/bin/bash

make -C nnmodel/devices
nosetests -v test/
