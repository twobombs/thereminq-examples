#!/bin/bash

export QRACK_OCL_DEFAULT_DEVICE=0
export QRACK_MAX_PAGING_QB=29
export QRACK_QTENSORNETWORK_THRESHOLD_QB=1

python3 alt_ising_ace.py
