#!/bin/bash
if [[ "$(docker images -q geliosphere:v1 2> /dev/null)" == "" ]]; then
  docker build -t geliosphere:v1 .
fi

docker run -v $(pwd)/output/:/cuda_implementation/output/ --gpus all geliosphere:v1 ./cuda_implementation/Geliosphere $@