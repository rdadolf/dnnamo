#!/bin/bash
git clone git://github.com/rdadolf/fathom-lite.git \
  && cd fathom-lite/data \
  && python ./download.py
