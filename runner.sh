#!/bin/bash
if [[ "$(docker images -q geliosphere:v1 2> /dev/null)" == "" ] || [$1 = '-f']]; then
  docker build -t geliosphere:v1 .
fi

if [[$1 = '-f']]; then
  docker run -v $(pwd)/output/:/cuda_implementation/output/ --gpus all geliosphere:v1 ./cuda_implementation/Geliosphere $@
fi