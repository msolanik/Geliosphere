#!/bin/bash
if [[ "$(docker images -q geliosphere-cpu:v1 2> /dev/null)" == "" || $1 = '-f' ]]; then
  docker build -t geliosphere-cpu:v1 -f Dockerfile.CPU .
fi

if [[ $1 != '-f' ]]; then
  docker run -v $(pwd)/output/:/results/ --gpus all geliosphere-cpu:v1 ./geliosphere/Geliosphere $@
fi