#!/bin/bash
# installs dependancies visualisations, cmd progress bars and chemical libaries

apt update && apt install -y python3-tk
pip install vedo pv tqdm basis-set-exchange

echo "sqr magnetisation ui"
python3 sqr_magnetisation-iterations_ui.py

echo "render the movie from output"
ffmpeg -framerate 24 -i TF%*.png -c:v libx264 -pix_fmt yuv420p tfim.mp4


tail -f /dev/null 
