#!/bin/bash
g++ -std=c++11 -O3 -DWEED_OPENCL -I./include -I./build/include/common holographic_ingest_check.cpp -o holographic_ingest_check -L/usr/local/lib/weed -lweed -lOpenCL