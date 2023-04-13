# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 4 * "/..")
sys.path.insert(0, parent_dir)

import tasks
from utils import params, common, parsing

import argparse
import random

random.seed(10)
import numpy as np
import sys

import yaml

def cluster_coef(pars, cfg):
    cluster_d = params.read_task_params("cluster")

    coef = float(pars["coef"])

    cl_size_coef = "{}_coef_{}".format(
        cfg["clusterStudies"]["clusterSizeBaseName"],
        str(round(coef, 3)).replace(".", "p"),
    )
    cluster_d["ClusterOutPlot"], cluster_d["ClusterOutValidation"] = cl_size_coef, cl_size_coef+"_valid"
    cluster_d["CoeffA"] = [coef, 0] * 50

    nevents_end = tasks.cluster.cluster_default(pars, **cluster_d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--coef", help="Coefficient to use as the max cluster radius", required=True)
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()

    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)

    cluster_coef(common.dot_dict(vars(FLAGS)), cfg)
