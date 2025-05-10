#!/bin/bash

set -e
rm -rf build
mkdir build
#cmake -B build -DCPU_VERSION_ONLY=1
#cmake -B build -DCPU_VERSION_ONLY=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto -funroll-loops -ffast-math"
cmake -B build -DCPU_VERSION_ONLY=1 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -flto -funroll-loops"
cmake --build build

