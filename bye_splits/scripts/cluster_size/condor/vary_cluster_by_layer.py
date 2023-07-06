# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 4 * "/..")
sys.path.insert(0, parent_dir)

import tasks
from utils import params, common, parsing, cl_helpers

from data_handle.data_process import get_data_reco_chain_start

import argparse
import random

random.seed(10)
import numpy as np
import pandas as pd

import yaml

layer_pt_frac = [8.937328360318834e-05, 0.01205718987915559, 0.04942834280744734, 0.1119014531979668, 0.16331881834425982, 0.18914793973125396, 0.15884167070142546, 0.12219957773838658, 0.08767152805444842, 0.045133183121142444, 0.021883454545564958, 0.011466352107155851, 0.01596599342661539, 0.010895123061574148]

def cluster_coef(pars, cfg):
    cluster_d = params.read_task_params("cluster")

    particles = pars["particles"]
    pileup = pars["pileup"]
    coef = pars["coef"]

    reprocess = cfg["clusterStudies"]["reprocess"]
    nevents = cfg["clusterStudies"]["nevents"]
    tag = cfg["clusterStudies"]["parquetTag"]

    cl_size_coef = "{}_coef_{}".format(
        cfg["clusterStudies"]["clusterSizeBaseName"],
        str(round(coef, 3)).replace(".", "p"),
    )
    cluster_d["ClusterOutPlot"], cluster_d["ClusterOutValidation"] = cl_size_coef, cl_size_coef+"_valid"

    weighted_radii = np.array([[pt_frac*coef]*2 for pt_frac in layer_pt_frac]).reshape(28,)
    weighted_radii = np.append(weighted_radii, np.zeros(22))

    #cluster_d["CoeffA"] = [coef] * 50
    scale = coef*len(weighted_radii)/weighted_radii.sum()
    cluster_d["CoeffA"] = [scale*radius for radius in weighted_radii]

    cluster_d["weights"] = cfg["weights"]

    for key in ("ClusterInTC", "ClusterInSeeds", "ClusterOutPlot", "ClusterOutValidation"):
        name = cluster_d[key]
        cluster_d[key] =  "{}_{}_{}_posEta".format(particles, pileup, name)

    nevents_end = tasks.cluster.cluster_default(pars, **cluster_d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--coef", help="Coefficient to use as the max cluster radius", required=True, type=float)
    parser.add_argument("--particles", choices=("photons", "electrons", "pions"), required=True)
    parser.add_argument("--pileup", help="tag for PU200 vs PU0", choices=("PU0", "PU200"), required=True)
    parsing.add_parameters(parser)
    
    FLAGS = parser.parse_args()
    pars = common.dot_dict(vars(FLAGS))

    radius = round(pars.coef, 3)

    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)

    weight_dir = "{}/PU0/".format(params.LocalStorage)
    weights_by_particle = cl_helpers.read_weights(weight_dir, cfg)
    weights = weights_by_particle[pars.particles][radius]
    cfg["weights"] = weights
    
    cluster_coef(pars, cfg)
