# this script creates heatmap data for running the ising model on the Qrack ACE backend

# smaller circuits can run forked en-masse on a massive multicore system
# medium circuits can run in a queue - look at system resources before starting
# large circuits should be running in dedicated multi GPU mode

# disable fid guard so the run isn't interrupted and data will be produced even with low prob.
export QRACK_DISABLE_QUNIT_FIDELITY_GUARD=1

# set here at 28 yet we will be able to go to 50+ instances in parallel
export QRACK_MAX_PAGING_QB=28

# tensor on = 0 / off is -1 
export QRACK_QTENSORNETWORK_THRESHOLD_QB=0

# press the any key to continue to the next circuit; calculations are forked and measurements are stored

export QRACK_OCL_DEFAULT_DEVICE=0
python3 ising_ace_depth_series.py 4 20 1024 False 1 2 1 > 4_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=1
python3 ising_ace_depth_series.py 6 20 1024 True 1 2 1 > 6_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=2
python3 ising_ace_depth_series.py 8 20 1024 True 2 2 1 > 8_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=0
python3 ising_ace_depth_series.py 9 20 1024 False 1 3 1 > 9_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=1
python3 ising_ace_depth_series.py 10 20 1024 True 1 2 1 > 10_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=2
python3 ising_ace_depth_series.py 12 20 1024 True 2 3 1 > 12_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=0
python3 ising_ace_depth_series.py 14 20 1024 False 1 2 1 > 14_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=1
python3 ising_ace_depth_series.py 15 20 1024 True 1 3 1 > 15_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=2
python3 ising_ace_depth_series.py 16 20 1024 False 2 4 1 > 16_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=0
python3 ising_ace_depth_series.py 18 20 1024 True 3 3 1 > 18_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=1
python3 ising_ace_depth_series.py 20 20 1024 True 1 4 1 > 20_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=2
python3 ising_ace_depth_series.py 24 20 1024 True 3 4 1 > 24_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=0
python3 ising_ace_depth_series.py 25 20 1024 False 1 5 1 > 25_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=1
python3 ising_ace_depth_series.py 28 20 1024 True 1 4 1 > 28_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=2
python3 ising_ace_depth_series.py 30 20 1024 True 3 5 1 > 30_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

# qiskit transpiler drama on thse settings
# export QRACK_OCL_DEFAULT_DEVICE=0
# python3 ising_ace_depth_series.py 32 20 1024 True 4 4 1 > 32_20.txt &
# read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=1
python3 ising_ace_depth_series.py 35 20 1024 True 1 7 1 > 35_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=2
python3 ising_ace_depth_series.py 36 20 1024 False 3 6 1 > 36_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=0
python3 ising_ace_depth_series.py 40 20 1024 False 4 5 1 > 40_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

# medium circuits

export QRACK_OCL_DEFAULT_DEVICE=1
python3 ising_ace_depth_series.py 42 20 1024 True 1 6 1 > 42_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=2
python3 ising_ace_depth_series.py 45 20 1024 True 3 4 1 > 45_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=0
python3 ising_ace_depth_series.py 48 20 1024 True 4 6 1 > 48_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=1
python3 ising_ace_depth_series.py 50 20 1024 True 5 5 1 > 50_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=2
python3 ising_ace_depth_series.py 54 20 1024 True 3 6 1 > 54_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=0
python3 ising_ace_depth_series.py 56 20 1024 True 3 7 1 > 56_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=1
python3 ising_ace_depth_series.py 60 20 1024 True 5 6 1 > 60_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=2
python3 ising_ace_depth_series.py 63 20 1024 True 3 7 1 > 63_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=0
python3 ising_ace_depth_series.py 64 20 1024 False 4 8 1 > 64_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=1
python3 ising_ace_depth_series.py 70 20 1024 True 5 7 1 > 70_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=2
python3 ising_ace_depth_series.py 72 20 1024 True 3 8 1 > 72_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=0
python3 ising_ace_depth_series.py 75 20 1024 True 5 5 1 > 75_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

export QRACK_OCL_DEFAULT_DEVICE=1
python3 ising_ace_depth_series.py 80 20 1024 False 5 8 1 > 80_20.txt &

read -n 1 -s -r -p "Press any key to continue..."

# there be dragons beyond this point
