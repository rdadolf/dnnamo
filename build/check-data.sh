#!/bin/bash

die() { echo "ERROR: $@"; exit 1; }

[ -d /data ]  \
  || die '/data does not exist.'

[ -f /data/DATA_DIRECTORY_IS_VALID ]  \
  || die '/data is missing a sentinel file; is it really a data directory?'

