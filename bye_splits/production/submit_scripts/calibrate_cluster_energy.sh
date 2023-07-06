#!/usr/bin/env bash

cd /home/llr/cms/ehle/NewRepos/bye_splits/bye_splits/scripts/cluster_size/

# Coefficients (radii) stored in .txt file, run cluster step on each radius
coef_file=$1
particles=$2
while read -r line; do
    radius=$(printf '%.*f\n' 3 "$line") #round to 3 sigfigs
    python calibration.py --radius "$radius" --particles "$particles"
    #python calibration.py --radius "$radius" --particles "$particles" --pileup
done <$coef_file