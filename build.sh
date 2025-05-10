#!/bin/bash

set -e
rm -rf build
mkdir build
#1. spustenie, original
cmake -B build -DCPU_VERSION_ONLY=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -g"


#cmake -B build -DCPU_VERSION_ONLY=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3"



#cmake -B build -DCPU_VERSION_ONLY=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -g -march=native -flto"
cmake --build build

