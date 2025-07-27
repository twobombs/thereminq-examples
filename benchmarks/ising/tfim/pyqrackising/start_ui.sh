#!/bin/bash
# installs dependancies 

apt update && apt install -y python3-tk
pip install vedo

# runs UIs for both sims

echo "magnetisation ui"
python3 magnetisation-iterations_ui.py

echo "sqr magnetisation ui"
python3 sqr_magnetisation-iterations_ui.py


