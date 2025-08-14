#!/bin/bash

python3 import_clifford_vqe_min-glocke.py > import_clifford_vqe_min-glocke.log &

tail -f import_clifford_vqe_min-glocke.log | grep Difference