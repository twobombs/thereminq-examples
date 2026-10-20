#!/bin/bash

mkdir build && cd build
# Enable OpenCL for the GPU support
cmake -DENABLE_OPENCL=ON -DCMAKE_BUILD_TYPE=Release .. 
make -j$(nproc)
