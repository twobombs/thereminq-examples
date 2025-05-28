#!/bin/bash

# advised and tested settings for running the ising models on the qrack ace mitiq backend
# a machine with 24 threads, 32GB ram and 32 GB vram will be able to run this up to 50 qubits at the same depth in parallel

export QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1
# set here at 28 yet we want to go to 50+ in parallel and it fits (!!)
export QRACK_MAX_PAGING_QB=28

# tensor on = 0 / off is -1 
export QRACK_QTENSORNETWORK_THRESHOLD_QB=0

# change to fit your preference
export QRACK_OCL_DEFAULT_DEVICE=2

# SDRP value 'du your'
export QRACK_QUNIT_SEPARABILITY_THRESHOLD=0.1464466

# This script will now run the ising_ace_depth_series.py forked
echo "Starting the script..."

# Loop from 4 to 50+ (inclusive)
for i in $(seq 4 50)
do
  echo "Running forked with parameter: $i and depth: $i"

  python3 ising_ace_depth_series.py "$i" "$i" > "$i".log &
  # depth replace second "i$" with required depth or multiples of $i

  # 30 second between threads to end and/or stack up 
  sleep 30

done

echo "Script finished - all will run in the background and produce logs and graphs"
