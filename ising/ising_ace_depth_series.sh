#!/bin/bash

# experimental highly paralellized script for the ising_ace_depth_series.py on https://github.com/vm6502q/pyqrack-examples/tree/main/ising by Dan Strano

# below are advised and tested settings for running highly compressed ising models on the qrack ace tfim mitiq backend whilst enabling GPU, tensorized and SDRP settings
# a machine with 24 threads, 32GB ram and 16 GB vram should be able to run this up to 50 qubits at the same depth in parallel at runtime 

# code and docker images avaliable on dockerhub and github
# https://github.com/twobombs
# https://hub.docker.com/r/twobombs/thereminq-tensors


# disable fid guard so the run isn't interrupted and data will be produced even with low prob.
export QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1

# set here at 28 yet we will be able to go to 50+ in parallel and it fits all on just 32 GB (!!)
export QRACK_MAX_PAGING_QB=28

# tensor on = 0 / off is -1 
export QRACK_QTENSORNETWORK_THRESHOLD_QB=0

# SDRP value 'du jour'
export QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.1464466

# set max device number
max_gpu=2

# This script will now run the ising_ace_depth_series.py forked
echo "Starting the script..."

# Loop from 4 to 50+ (inclusive)
for i in $(seq 4 156)
do
  echo "Running forked with parameter: $i and depth: $i"
  
  # rnd gpu selection - set max to fit your preference on max_gpu
  gpuselect=$((RANDOM % max_gpu))
  export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
  
  python3 ising_ace_depth_series.py "$i" "$i" > "$i".log &
  # depth replace second "$i" with required depth or multiples of $i

  # 30 second between threads to end and/or stack up 
  # sleep 30
  # or (default) the anykey
  read -n 1 -s -r -p "Press any key to continue..."
done

echo "Script finished - all will run in the background and produce logs and graphs"
