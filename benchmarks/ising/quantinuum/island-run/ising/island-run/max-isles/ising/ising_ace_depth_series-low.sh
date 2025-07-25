#!/bin/bash
# This script runs a comprehensive series of simulations for the ising_ace_depth_series.py script.
# Each line corresponds to a specific qubit configuration from 4 to 1024 qubits.
#
# The command structure is:
# python3 ising_ace_depth_series.py <width> <depth> <iterations> <num_tests> <column> <row>
#
# <width>: Total number of qubits.
# <depth>: The simulation depth.
# <iterations>: The number of shots/iterations.
# <num_tests>: The number of tests (always 1 for this script).
# <column>: The number of columns in the patch.
# <row>: The number of rows in the patch.

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

# This script runs a comprehensive series of simulations for the ising_ace_depth_series.py script.
# Each line corresponds to a specific qubit configuration from 4 to 1024 qubits,
# using the MINIMUM number of islands for the patching strategy.
# The script will pause after launching each simulation and wait for a key press.
# It randomly selects a GPU for each run.
#
# The command structure is:
# python3 ising_ace_depth_series.py <width> <depth> <iterations> <column> <row> <num_tests>
#
# <width>: Total number of qubits.
# <depth>: The simulation depth.
# <iterations>: The number of shots/iterations.
# <column>: The number of columns in the patch.
# <row>: The number of rows in the patch.
# <num_tests>: The number of tests (always 1 for this script).

echo "Starting comprehensive simulation series (4 to 1024 qubits) with MINIMUM islands..."

# Set this to the number of available GPUs
max_gpu=3

# n_qubits=4, Grid=2x2, Min Islands=2, Patch=2x1
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 4 20 1024 1 2 1 > 4x20_min_islands.log &
echo "Launched width: 4, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=6, Grid=2x3, Min Islands=3, Patch=2x1
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 6 20 1024 1 2 1 > 6x20_min_islands.log &
echo "Launched width: 6, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=8, Grid=2x4, Min Islands=2, Patch=2x2
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 8 20 1024 2 2 1 > 8x20_min_islands.log &
echo "Launched width: 8, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=9, Grid=3x3, Min Islands=3, Patch=3x1
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 9 20 1024 1 3 1 > 9x20_min_islands.log &
echo "Launched width: 9, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=10, Grid=2x5, Min Islands=5, Patch=2x1
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 10 20 1024 1 2 1 > 10x20_min_islands.log &
echo "Launched width: 10, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=12, Grid=3x4, Min Islands=2, Patch=3x2
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 12 20 1024 2 3 1 > 12x20_min_islands.log &
echo "Launched width: 12, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=14, Grid=2x7, Min Islands=7, Patch=2x1
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 14 20 1024 1 2 1 > 14x20_min_islands.log &
echo "Launched width: 14, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=15, Grid=3x5, Min Islands=5, Patch=3x1
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 15 20 1024 1 3 1 > 15x20_min_islands.log &
echo "Launched width: 15, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=16, Grid=4x4, Min Islands=2, Patch=4x2
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 16 20 1024 2 4 1 > 16x20_min_islands.log &
echo "Launched width: 16, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=18, Grid=3x6, Min Islands=2, Patch=3x3
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 18 20 1024 3 3 1 > 18x20_min_islands.log &
echo "Launched width: 18, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=20, Grid=4x5, Min Islands=5, Patch=4x1
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 20 20 1024 1 4 1 > 20x20_min_islands.log &
echo "Launched width: 20, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=21, Grid=3x7, Min Islands=7, Patch=3x1
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 21 20 1024 1 3 1 > 21x20_min_islands.log &
echo "Launched width: 21, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=24, Grid=4x6, Min Islands=2, Patch=4x3
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 24 20 1024 3 4 1 > 24x20_min_islands.log &
echo "Launched width: 24, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=25, Grid=5x5, Min Islands=5, Patch=5x1
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 25 20 1024 1 5 1 > 25x20_min_islands.log &
echo "Launched width: 25, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=28, Grid=4x7, Min Islands=7, Patch=4x1
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 28 20 1024 1 4 1 > 28x20_min_islands.log &
echo "Launched width: 28, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=30, Grid=5x6, Min Islands=2, Patch=5x3
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 30 20 1024 3 5 1 > 30x20_min_islands.log &
echo "Launched width: 30, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=32, Grid=4x8, Min Islands=2, Patch=4x4
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 32 20 1024 4 4 1 > 32x20_min_islands.log &
echo "Launched width: 32, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=35, Grid=5x7, Min Islands=7, Patch=5x1
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 35 20 1024 1 5 1 > 35x20_min_islands.log &
echo "Launched width: 35, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=36, Grid=6x6, Min Islands=2, Patch=6x3
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 36 20 1024 3 6 1 > 36x20_min_islands.log &
echo "Launched width: 36, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=40, Grid=5x8, Min Islands=2, Patch=5x4
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 40 20 1024 4 5 1 > 40x20_min_islands.log &
echo "Launched width: 40, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=42, Grid=6x7, Min Islands=7, Patch=6x1
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 42 20 1024 1 6 1 > 42x20_min_islands.log &
echo "Launched width: 42, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=45, Grid=5x9, Min Islands=3, Patch=5x3
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 45 20 1024 3 5 1 > 45x20_min_islands.log &
echo "Launched width: 45, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=48, Grid=6x8, Min Islands=2, Patch=6x4
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 48 20 1024 4 6 1 > 48x20_min_islands.log &
echo "Launched width: 48, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=50, Grid=5x10, Min Islands=2, Patch=5x5
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 50 20 1024 5 5 1 > 50x20_min_islands.log &
echo "Launched width: 50, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=54, Grid=6x9, Min Islands=3, Patch=6x3
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 54 20 1024 3 6 1 > 54x20_min_islands.log &
echo "Launched width: 54, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=56, Grid=7x8, Min Islands=2, Patch=7x4
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 56 20 1024 4 7 1 > 56x20_min_islands.log &
echo "Launched width: 56, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=60, Grid=6x10, Min Islands=2, Patch=6x5
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 60 20 1024 5 6 1 > 60x20_min_islands.log &
echo "Launched width: 60, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=63, Grid=7x9, Min Islands=3, Patch=7x3
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 63 20 1024 3 7 1 > 63x20_min_islands.log &
echo "Launched width: 63, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=64, Grid=8x8, Min Islands=2, Patch=8x4
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 64 20 1024 4 8 1 > 64x20_min_islands.log &
echo "Launched width: 64, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=70, Grid=7x10, Min Islands=2, Patch=7x5
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 70 20 1024 5 7 1 > 70x20_min_islands.log &
echo "Launched width: 70, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=72, Grid=8x9, Min Islands=3, Patch=8x3
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 72 20 1024 3 8 1 > 72x20_min_islands.log &
echo "Launched width: 72, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=75, Grid=5x15, Min Islands=3, Patch=5x5
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 75 20 1024 5 5 1 > 75x20_min_islands.log &
echo "Launched width: 75, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=80, Grid=8x10, Min Islands=2, Patch=8x5
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 80 20 1024 5 8 1 > 80x20_min_islands.log &
echo "Launched width: 80, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=81, Grid=9x9, Min Islands=3, Patch=9x3
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 81 20 1024 3 9 1 > 81x20_min_islands.log &
echo "Launched width: 81, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=84, Grid=7x12, Min Islands=2, Patch=7x6
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 84 20 1024 6 7 1 > 84x20_min_islands.log &
echo "Launched width: 84, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=90, Grid=9x10, Min Islands=2, Patch=9x5
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 90 20 1024 5 9 1 > 90x20_min_islands.log &
echo "Launched width: 90, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=96, Grid=8x12, Min Islands=2, Patch=8x6
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 96 20 1024 6 8 1 > 96x20_min_islands.log &
echo "Launched width: 96, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=98, Grid=7x14, Min Islands=2, Patch=7x7
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 98 20 1024 7 7 1 > 98x20_min_islands.log &
echo "Launched width: 98, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=100, Grid=10x10, Min Islands=2, Patch=10x5
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 100 20 1024 5 10 1 > 100x20_min_islands.log &
echo "Launched width: 100, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=108, Grid=9x12, Min Islands=2, Patch=9x6
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 108 20 1024 6 9 1 > 108x20_min_islands.log &
echo "Launched width: 108, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=112, Grid=8x14, Min Islands=2, Patch=8x7
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 112 20 1024 7 8 1 > 112x20_min_islands.log &
echo "Launched width: 112, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=120, Grid=10x12, Min Islands=2, Patch=10x6
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 120 20 1024 6 10 1 > 120x20_min_islands.log &
echo "Launched width: 120, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=121, Grid=11x11, Min Islands=11, Patch=11x1
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 121 20 1024 1 11 1 > 121x20_min_islands.log &
echo "Launched width: 121, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=125, Grid=5x25, Min Islands=5, Patch=5x5
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 125 20 1024 5 5 1 > 125x20_min_islands.log &
echo "Launched width: 125, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=128, Grid=8x16, Min Islands=2, Patch=8x8
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 128 20 1024 8 8 1 > 128x20_min_islands.log &
echo "Launched width: 128, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=132, Grid=11x12, Min Islands=2, Patch=11x6
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 132 20 1024 6 11 1 > 132x20_min_islands.log &
echo "Launched width: 132, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=140, Grid=10x14, Min Islands=2, Patch=10x7
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 140 20 1024 7 10 1 > 140x20_min_islands.log &
echo "Launched width: 140, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=144, Grid=12x12, Min Islands=2, Patch=12x6
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 144 20 1024 6 12 1 > 144x20_min_islands.log &
echo "Launched width: 144, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=150, Grid=10x15, Min Islands=3, Patch=10x5
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 150 20 1024 5 10 1 > 150x20_min_islands.log &
echo "Launched width: 150, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=160, Grid=10x16, Min Islands=2, Patch=10x8
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 160 20 1024 8 10 1 > 160x20_min_islands.log &
echo "Launched width: 160, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=162, Grid=9x18, Min Islands=2, Patch=9x9
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 162 20 1024 9 9 1 > 162x20_min_islands.log &
echo "Launched width: 162, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=168, Grid=12x14, Min Islands=2, Patch=12x7
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 168 20 1024 7 12 1 > 168x20_min_islands.log &
echo "Launched width: 168, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=175, Grid=7x25, Min Islands=5, Patch=7x5
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 175 20 1024 5 7 1 > 175x20_min_islands.log &
echo "Launched width: 175, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=180, Grid=12x15, Min Islands=3, Patch=12x5
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 180 20 1024 5 12 1 > 180x20_min_islands.log &
echo "Launched width: 180, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=192, Grid=12x16, Min Islands=2, Patch=12x8
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 192 20 1024 8 12 1 > 192x20_min_islands.log &
echo "Launched width: 192, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=196, Grid=14x14, Min Islands=2, Patch=14x7
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 196 20 1024 7 14 1 > 196x20_min_islands.log &
echo "Launched width: 196, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=200, Grid=10x20, Min Islands=2, Patch=10x10
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 200 20 1024 10 10 1 > 200x20_min_islands.log &
echo "Launched width: 200, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=216, Grid=12x18, Min Islands=2, Patch=12x9
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 216 20 1024 9 12 1 > 216x20_min_islands.log &
echo "Launched width: 216, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=224, Grid=14x16, Min Islands=2, Patch=14x8
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 224 20 1024 8 14 1 > 224x20_min_islands.log &
echo "Launched width: 224, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=240, Grid=12x20, Min Islands=2, Patch=12x10
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 240 20 1024 10 12 1 > 240x20_min_islands.log &
echo "Launched width: 240, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=250, Grid=10x25, Min Islands=5, Patch=10x5
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 250 20 1024 5 10 1 > 250x20_min_islands.log &
echo "Launched width: 250, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=256, Grid=16x16, Min Islands=2, Patch=16x8
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 256 20 1024 8 16 1 > 256x20_min_islands.log &
echo "Launched width: 256, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=280, Grid=14x20, Min Islands=2, Patch=14x10
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 280 20 1024 10 14 1 > 280x20_min_islands.log &
echo "Launched width: 280, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=288, Grid=16x18, Min Islands=2, Patch=16x9
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 288 20 1024 9 16 1 > 288x20_min_islands.log &
echo "Launched width: 288, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=300, Grid=15x20, Min Islands=2, Patch=15x10
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 300 20 1024 10 15 1 > 300x20_min_islands.log &
echo "Launched width: 300, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=320, Grid=16x20, Min Islands=2, Patch=16x10
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 320 20 1024 10 16 1 > 320x20_min_islands.log &
echo "Launched width: 320, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=336, Grid=14x24, Min Islands=2, Patch=14x12
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 336 20 1024 12 14 1 > 336x20_min_islands.log &
echo "Launched width: 336, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=352, Grid=16x22, Min Islands=2, Patch=16x11
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 352 20 1024 11 16 1 > 352x20_min_islands.log &
echo "Launched width: 352, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=360, Grid=18x20, Min Islands=2, Patch=18x10
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 360 20 1024 10 18 1 > 360x20_min_islands.log &
echo "Launched width: 360, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=384, Grid=16x24, Min Islands=2, Patch=16x12
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 384 20 1024 12 16 1 > 384x20_min_islands.log &
echo "Launched width: 384, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=400, Grid=20x20, Min Islands=2, Patch=20x10
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 400 20 1024 10 20 1 > 400x20_min_islands.log &
echo "Launched width: 400, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=432, Grid=18x24, Min Islands=2, Patch=18x12
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 432 20 1024 12 18 1 > 432x20_min_islands.log &
echo "Launched width: 432, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=448, Grid=16x28, Min Islands=2, Patch=16x14
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 448 20 1024 14 16 1 > 448x20_min_islands.log &
echo "Launched width: 448, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=480, Grid=20x24, Min Islands=2, Patch=20x12
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 480 20 1024 12 20 1 > 480x20_min_islands.log &
echo "Launched width: 480, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=500, Grid=20x25, Min Islands=5, Patch=20x5
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 500 20 1024 5 20 1 > 500x20_min_islands.log &
echo "Launched width: 500, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=512, Grid=16x32, Min Islands=2, Patch=16x16
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 512 20 1024 16 16 1 > 512x20_min_islands.log &
echo "Launched width: 512, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=540, Grid=18x30, Min Islands=2, Patch=18x15
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 540 20 1024 15 18 1 > 540x20_min_islands.log &
echo "Launched width: 540, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=576, Grid=24x24, Min Islands=2, Patch=24x12
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 576 20 1024 12 24 1 > 576x20_min_islands.log &
echo "Launched width: 576, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=600, Grid=20x30, Min Islands=2, Patch=20x15
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 600 20 1024 15 20 1 > 600x20_min_islands.log &
echo "Launched width: 600, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=640, Grid=20x32, Min Islands=2, Patch=20x16
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 640 20 1024 16 20 1 > 640x20_min_islands.log &
echo "Launched width: 640, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=672, Grid=24x28, Min Islands=2, Patch=24x14
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 672 20 1024 14 24 1 > 672x20_min_islands.log &
echo "Launched width: 672, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=720, Grid=24x30, Min Islands=2, Patch=24x15
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 720 20 1024 15 24 1 > 720x20_min_islands.log &
echo "Launched width: 720, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=750, Grid=25x30, Min Islands=2, Patch=25x15
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 750 20 1024 15 25 1 > 750x20_min_islands.log &
echo "Launched width: 750, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=768, Grid=24x32, Min Islands=2, Patch=24x16
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 768 20 1024 16 24 1 > 768x20_min_islands.log &
echo "Launched width: 768, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=800, Grid=20x40, Min Islands=2, Patch=20x20
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 800 20 1024 20 20 1 > 800x20_min_islands.log &
echo "Launched width: 800, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=864, Grid=24x36, Min Islands=2, Patch=24x18
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 864 20 1024 18 24 1 > 864x20_min_islands.log &
echo "Launched width: 864, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=900, Grid=30x30, Min Islands=2, Patch=30x15
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 900 20 1024 15 30 1 > 900x20_min_islands.log &
echo "Launched width: 900, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=960, Grid=24x40, Min Islands=2, Patch=24x20
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 960 20 1024 20 24 1 > 960x20_min_islands.log &
echo "Launched width: 960, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=1000, Grid=25x40, Min Islands=2, Patch=25x20
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 1000 20 1024 20 25 1 > 1000x20_min_islands.log &
echo "Launched width: 1000, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

# n_qubits=1024, Grid=32x32, Min Islands=2, Patch=32x16
gpuselect=$((RANDOM % max_gpu)) && export QRACK_OCL_DEFAULT_DEVICE=$gpuselect
python3 ising_ace_depth_series.py 1024 20 1024 16 32 1 > 1024x20_min_islands.log &
echo "Launched width: 1024, depth: 20 on GPU: $gpuselect"
read -n 1 -s -r -p "Press any key to continue..."

wait # Wait for all background jobs to finish
echo "All simulations have completed."
