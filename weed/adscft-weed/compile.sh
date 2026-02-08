#!/bin/bash
g++ -std=c++11 -O3 -DWEED_OPENCL -I./include -I./build/include/common holographic_ingest.cpp -o holographic_ingest -L/usr/local/lib/weed -lweed -lOpenCL