#!/bin/bash
if [[ "$(docker images -q geliosphere:v1 2> /dev/null)" == "" || $1 = '-f' ]]; then
  docker build -t geliosphere:v1 .
fi

if [[ $1 != '-f' ]]; then
  docker run -v $(pwd)/output/:/geliosphere/results/ --gpus all geliosphere:v1 ./geliosphere/Geliosphere $@
fi