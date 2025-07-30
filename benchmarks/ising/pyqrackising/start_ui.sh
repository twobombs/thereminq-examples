#!/bin/bash
# installs dependancies visualisations, cmd progress bars and chemical libaries

apt update && apt install -y python3-tk
pip install vedo pv tqdm basis-set-exchange

#pip install cupy-cuda12x
#pip install pyscf-gpu

# runs UIs for both sims

echo "magnetisation ui"
python3 magnetisation-iterations_ui.py &

echo "sqr magnetisation ui"
python3 sqr_magnetisation-iterations_ui.py &


tail -f /dev/null 
