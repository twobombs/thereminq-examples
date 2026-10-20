#!/bin/bash

python3 import_clifford_vqe_entangled.py > import_clifford_vqe_entangled.log &

tail -f import_clifford_vqe_entangled.log | grep Difference
