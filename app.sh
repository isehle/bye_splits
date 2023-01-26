#! /bin/bash
conda activate ByeSplitEnv
export PYTHONPATH=/home/llr/cms/ehle/anaconda3/envs/ByeSplitEnv/bin/python
python bye_splits/scripts/cl_app.py
#--allow-websocket-origin=bye-splits-app-hgcal-cl-size-studies.app.cern.ch
