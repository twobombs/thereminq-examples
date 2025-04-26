#!/bin/bash

# default for largest device
export QRACK_OCL_DEFAULT_DEVICE=0
# max paging 
export QRACK_MAX_PAGING_QB=28
# lightcone on
export QRACK_QTENSORNETWORK_THRESHOLD_QB=1
# new var for max frambuffer
export QRACK_QBDT_MAX_ALLOC_MB=10000
# Suggested 'WD40' value for SDRP
export QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.024

cd /notebooks/qrack/pyqrack-examples/ising
python3 ising_ace.py
