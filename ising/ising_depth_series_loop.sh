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

# SDRP value 'du jour' - disabled because of build-in value
#export QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.1464466

# set max devices
max_gpu=3

# This script will now run the ising_ace_depth_series.py forked
echo "Starting the script..."

# Loop from 4 to 50+ (inclusive)
for i in $(seq 4 350)
do
  echo "Running forked with parameter: $i and depth: 20"
  
  # rnd gpu selection - set max to fit your preference on max_gpu
  gpuselect=$((RANDOM % max_gpu))
  export QRACK_OCL_DEFAULT_DEVICE=$gpuselect

  # kinda middle of the road setting that might not always work right
  echo "job is running on GPU: $gpuselect "
  python3 ising_depth_series.py "$i" 20 1024 1 > "$i".log &
  # depth replace second var with required depth or multiples of $i
  # third var is number of measurements 
  # fourth is number of trials

  read -n 1 -s -r -p "Press any key to continue..."
done

echo "Script finished - all will run in the background and produce logs and graphs"
