#!/usr/bin/env bash

cd /home/llr/cms/ehle/NewRepos/bye_splits/bye_splits/scripts/cluster_size/condor/

# Run initial steps of reco chain: Read, Fill, Smooth, Seed
python run_init_tasks.py

# Coefficients (radii) stored in .txt file, run cluster step on each radius
coef_file=$1
while read -r line; do
    python run_cluster.py --coef "$line"
done <$coef_file

# Combine the cluster files (and add some normalization)
python run_combine.py