# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 4 * "/..")
sys.path.insert(0, parent_dir)

import tasks
from utils import params, common, parsing

from data_handle.data_process import get_data_reco_chain_start

import argparse
import random

random.seed(10)
import sys

import yaml

def start_chain(pars, cfg):
    nevents = cfg["clusterStudies"]["nevents"]
    reprocess = cfg["clusterStudies"]["reprocess"]
    particles = cfg["selection"]["particles"]
    pileup = "PU0" if not cfg["clusterStudies"]["pileup"] else "PU200"

    df_gen, df_cl, df_tc = get_data_reco_chain_start(
        particles=particles, nevents=nevents, reprocess=reprocess
    )

    fill_d = params.read_task_params("fill")
    for key in ("FillOut", "FillOutComp", "FillOutPlot"):
        name = fill_d[key]
        fill_d[key] = "{}_{}_{}".format(particles, pileup, name)
    tasks.fill.fill(pars, df_gen, df_cl, df_tc, **fill_d)

    smooth_d = params.read_task_params("smooth")
    for key in ("SmoothIn", "SmoothOut"):
        name = smooth_d[key]
        smooth_d[key] =  "{}_{}_{}".format(particles, pileup, name)
    tasks.smooth.smooth(pars, **smooth_d)

    seed_d = params.read_task_params("seed")
    for key in ("SeedIn", "SeedOut"):
        name = seed_d[key]
        seed_d[key] = "{}_{}_{}".format(particles, pileup, name)
    tasks.seed.seed(pars, **seed_d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parsing.add_parameters(parser)

    FLAGS = parser.parse_args()

    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)

    for particles in ("electrons", "pions"):
        cfg["selection"]["particles"] = particles
        start_chain(common.dot_dict(vars(FLAGS)), cfg)
