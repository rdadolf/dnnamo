#!/bin/bash

docker build . -f Dockerfile -t rdadolf/dnnamo
docker build . -f Dockerfile.dev -t rdadolf/dnnamo:dev
