# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

from utils import common, params, cl_helpers
from bye_splits.data_handle import data_process

import argparse
import random

random.seed(10)
import numpy as np
import pandas as pd
import sys

import yaml

from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression

with open(params.CfgPath, "r") as afile:
    cfg = yaml.safe_load(afile)

def selection(pars, cfg):
    pile_up = "PU0" if not pars.pileup else "PU200"
    particles = pars.particles
    radius = pars.radius
    radius_str = "/coef_{}".format(str(radius).replace(".", "p"))

    cfg_sel = cfg["selection"]
    cfg_cl = cfg["clusterStudies"]

    file = cfg_cl["dashApp"][pile_up][pars.particles][0]
    reprocess = cfg_cl["reprocess"]
    nevents = cfg_cl["nevents"]
    
    # Cuts
    ptcut=cfg_cl["optimization"]["pt_cut"]
    clptcut=cfg_cl["optimization"]["cl_pt_cut"]
    etamin=cfg_cl["optimization"]["eta_min"]
    etamax=cfg_cl["optimization"]["eta_max"]

    with pd.HDFStore(file, mode="r") as clFile:
        df = clFile[radius_str]
        df.drop(["pt_norm","en_norm"],axis=1, inplace=True)
    
    if "deltaR" not in df.keys():
        dR_thresh = cfg_sel["deltarThreshold"]
        df["dR"] = np.sqrt((df["eta"]-df["gen_eta"])**2 + (df["phi"]-df["gen_phi"])**2)
        df = df[ df.dR < dR_thresh ]
        df.drop("dR", axis=1, inplace=True)


    df = df[ df.gen_pt > ptcut ]
    df = df[ df.gen_eta > etamin ]
    df = df[ df.gen_eta < etamax ]

    df = df[ df.pt > clptcut ]
    df = df[ df.eta > etamin ]
    df = df[ df.eta < etamax ]

    ds_gen, _ , _ = data_process.get_data_reco_chain_start(nevents=nevents, reprocess=reprocess, particles=particles)
    ds_gen.set_index("event", inplace=True)
    ds_gen = ds_gen["gen_layer"].to_frame()

    joined = df.join(ds_gen, on="event", how="inner")
    joined =  joined.groupby("event").apply(max)

    return joined

def optimize(df, as_df = True):

    layers = df.groupby(["event", "gen_layer"]).apply(lambda x: x.pt.sum()).to_frame()
    layers.rename({0: "pt"}, axis=1, inplace=True)
    layer_array = layers.unstack(level="gen_layer")
    layer_array = layer_array.fillna(0.0)

    regression = lsq_linear(layer_array, df['gen_pt'],
                            bounds = (0.,2.0),
                            method='bvls',
                            lsmr_tol='auto',
                            verbose=1)

    weights = regression.x

    if as_df:
        index = [2*i+1 for i in range(len(weights))]
        columns = ["weights"]
        weights = pd.DataFrame(data = weights,
                               index = index,
                               columns = columns)

    return weights

def apply_weights(df, weights):

    df = df.groupby("event", group_keys=True).apply(lambda x: x.merge(weights,left_on="gen_layer",right_index=True))
    for col in ("en", "pt"):
        new_col = "new_{}".format(col)
        df[new_col] = df[col] * df["weights"]

    return df

def main(pars, cfg):
    # Initial selection and TC matching
    init_df = selection(pars, cfg)
    # Grouping, summing, linear regression
    weights = optimize(init_df)

    final_df = apply_weights(init_df, weights)

    return weights, final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--particles", choices=("photons", "electrons", "pions"), required=True)
    parser.add_argument("--radius", help="clustering radius to use: (0.0, 0.05]", required=True, type=float)
    parser.add_argument("--pileup", help="tag for PU200 vs PU0", action="store_true")

    FLAGS = parser.parse_args()
    pars = common.dot_dict(vars(FLAGS))

    out_dir = "{}/PU0/{}/optimization/official/".format(cfg["clusterStudies"]["localDir"], pars.particles)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    radius_str = str(round(pars.radius, 4)).replace(".", "p")
    file_name = "{}/{}_r{}.hdf5".format(out_dir, cfg["clusterStudies"]["optimization"]["baseName"], radius_str)
    
    if not os.path.exists(file_name):
        weights, df = main(pars, cfg)
        with pd.HDFStore(file_name, mode="w") as outOpt:
            outOpt["weights"] = weights
            outOpt["df"] = df
    else:
        print("\n{} already exists, skipping\n.".format(file_name))
    
