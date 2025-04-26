#!/bin/bash

export QRACK_OCL_DEFAULT_DEVICE=0
export QRACK_MAX_PAGING_QB=28
export QRACK_QTENSORNETWORK_THRESHOLD_QB=1
export QRACK_QBDT_MAX_ALLOC_MB=10000

cd /notebooks/qrack/pyqrack-examples/ising
python3 ising_ace.py
