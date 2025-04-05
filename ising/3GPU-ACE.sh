#!/bin/bash

export QRACK_QPAGER_DEVICES=10.2
export QRACK_QPAGER_DEVICES_HOST_POINTER=10.0,1.1
export QRACK_OCL_DEFAULT_DEVICE=0

python3 alt_ising_ace.py
