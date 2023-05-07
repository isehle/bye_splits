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
from tqdm import tqdm

with open(params.CfgPath, "r") as afile:
    cfg = yaml.safe_load(afile)

layers = cfg["selection"]["disconnectedTriggerLayers"]
layers = [l-1 for l in layers]

def selection(df_cl, df_tc, dRThresh, radius):
    if "matches" not in df_cl.keys():
        try:
            df_cl["dR"] = np.sqrt((df_cl["gen_eta"] - df_cl["eta"])**2 + (df_cl["gen_phi"] - df_cl["phi"])**2)
            df_cl["matches"] = df_cl["dR"] <= dRThresh
        except KeyError:
            assert(df_cl.empty)
            return None
    
    df_cl = df_cl[ df_cl["matches"] == True ]
    df_cl = df_cl.drop("matches", axis=1)
    cl_en, gen_en = df_cl.en.to_frame(), df_cl.gen_en.to_frame()

    df_tc = df_tc[ df_tc.tc_multicluster_id != 0 ]
    
    tc_dR = lambda df: np.sqrt(((df.x/df.z) - (df.tc_x/df.tc_z))**2 + ((df.y/df.z) - (df.tc_y/df.tc_z))**2)

    joined = df_tc.join(df_cl, on="event").dropna()
    joined["tc_dR"] = tc_dR(joined)
    joined = joined[ joined["tc_dR"] <= radius ]

    en_per_layer = joined.groupby(["event","tc_layer"]).apply(lambda x: x.tc_energy.sum()).to_frame()
    en_per_layer = en_per_layer.rename({0: "summed_energy"},axis=1).reset_index(level=["tc_layer"])
    merged_energies = pd.merge(en_per_layer, cl_en, on="event")
    merged_energies = pd.merge(merged_energies, gen_en, on="event")

    return merged_energies

def optimize(pars, cfg):
    pile_up = "PU0" if not pars.pileup else "PU200"
    particles = pars.particles
    radius = pars.radius

    cfg_sel = cfg["selection"]
    cfg_cl = cfg["clusterStudies"]

    assert(pars.particles==cfg_sel["particles"] and pars.pileup==cfg_sel["pileup"])

    file = cfg_cl["dashApp"][pile_up][pars.particles][0]
    reprocess = cfg_cl["reprocess"]
    nevents = cfg_cl["nevents"]

    dR = cfg_sel["deltarThreshold"]

    _, _, tcData = data_process.get_data_reco_chain_start(nevents=nevents, reprocess=reprocess, particles=particles)

    with pd.HDFStore(file, mode="r") as clusterSizeOut:
        radius_str = "/coef_{}".format(str(radius).replace(".","p"))
        df = clusterSizeOut[radius_str]
        vals = selection(df, tcData, dR, radius) if isinstance(df, pd.DataFrame) else None
        
    return vals

def fill_event_layers(df, layers):

    current_event = int(df.event.unique())

    needed_layers = [layer for layer in layers if layer not in list(df.tc_layer)]

    for layer in needed_layers:
        new_row = pd.Series({"event": current_event, "tc_layer": layer, "summed_energy": 0.0})
        df = pd.concat([df, new_row.to_frame().T], ignore_index=True)

    df.sort_values("tc_layer", inplace=True)
    df = df.groupby("tc_layer").apply(max)

    return df

def fill_layers(df, cfg):
    layers = cfg["selection"]["disconnectedTriggerLayers"]
    layers = [l-1 for l in layers]

    df = df.reset_index()
    
    sub_df = df[["event","en", "gen_en"]].groupby("event").apply(max).drop("event", axis=1)
    tc_df = df.drop(["en", "gen_en"], axis=1)

    df = tc_df.groupby(["event"]).apply(lambda x: fill_event_layers(x, layers)).droplevel(1).drop("event", axis=1)
    
    merged = pd.merge(df, sub_df, left_on="event", right_on="event")
    
    return merged

def get_en_norms(df_tc, df_cl_gen):

    tc_sums = df_tc.groupby("event").apply(lambda x: sum(x.summed_energy))

    merged = pd.merge(df_cl_gen, tc_sums.to_frame(), on="event").rename({0: "tc_sums"}, axis=1)
    final = pd.merge(df_tc, merged, on="event")

    final["en_norm"] = final.tc_sums/final.gen_en

    return final

def get_weights(df):
    df["new_en_sums"] = df.groupby("event",as_index=False).apply(lambda x: x.summed_energy/x.en_norm).reset_index().drop(["level_0"],axis=1).set_index("event")
    df["normed_en_sums"]=df.groupby("event").apply(lambda x: x.new_en_sums/x.new_en_sums.sum()).to_frame().reset_index(level=1,drop=True)
    weights = 1+df.groupby("tc_layer").apply(lambda x: x.normed_en_sums.mean()).to_frame().rename({0:"weights"},axis=1)

    return weights

def apply_weights(df, weights):
    df = df.groupby("event").apply(lambda x: x.merge(weights,left_on="tc_layer",right_index=True))
    df["new_en_sums"] = df.summed_energy*df.weights
    new_en_norms = df.groupby("event").apply(lambda x: x.new_en_sums.sum()/x.gen_en.mean()).to_frame().rename({0: "new_en_norm"},axis=1)
    df = df.groupby("event").apply(lambda x: x.merge(new_en_norms,left_index=True,right_index=True))

    return df

def main(pars, cfg):
    init_vals = optimize(pars, cfg)
    filled_vals = fill_layers(init_vals, cfg)

    df_cl_gen = filled_vals[["en", "gen_en"]].groupby("event").apply(max)
    df_tc = filled_vals.drop(["en", "gen_en"], axis=1)
    df = get_en_norms(df_tc, df_cl_gen)

    weights = get_weights(df)
    final_df = apply_weights(df, weights)

    return weights, final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--particles", choices=("photons", "electrons", "pions"), required=True)
    parser.add_argument("--radius", help="clustering radius to use: (0.0, 0.05]", required=True, type=float)
    parser.add_argument("--pileup", help="tag for PU200 vs PU0", action="store_true")

    FLAGS = parser.parse_args()
    pars = common.dot_dict(vars(FLAGS))

    out_dir = "{}/PU0/{}/optimization/".format(cfg["clusterStudies"]["localDir"], pars.particles)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    radius_str = str(pars.radius).replace(".", "p")
    file_name = "{}/{}_r{}.hdf5".format(out_dir, cfg["clusterStudies"]["optimization"]["baseName"], radius_str)

    if not os.path.exists(file_name):
        weights, df = main(pars, cfg)
        with pd.HDFStore(file_name, mode="w") as outOpt:
            outOpt["weights"] = weights
            outOpt["df"] = df
    else:
        print("\n{} already exists, skipping\n.".format(file_name))
    
