# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

from utils import common, params, cl_helpers, parsing
from bye_splits.data_handle import data_process

import argparse
import random

random.seed(10)
import numpy as np
import pandas as pd
import h5py

import yaml
import re

from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression

with open(params.CfgPath, "r") as afile:
    cfg = yaml.safe_load(afile)

def assign_tcs(pars, **kw):

    seed_path = common.fill_path(kw["ClusterInSeeds"], **pars)
    tc_path = common.fill_path(kw["ClusterInTC"], **pars)

    sseeds = h5py.File(seed_path, mode='r')
    stc = h5py.File(tc_path, mode='r')
        
    seed_keys = [x for x in sseeds.keys() if '_group' in x and 'central' not in x]
    tc_keys = [x for x in stc.keys() if '_tc' in x and 'central' not in x]
    assert len(seed_keys) == len(tc_keys)

    radiusCoeffB = kw["CoeffB"]
    empty_seeds = 0

    tc_info = {}
    if pars.particles == "pions":
        ecal_df = pd.DataFrame(columns=["ecal_cl_pt"])
    for tck, seedk in zip(tc_keys, seed_keys):
        tc = stc[tck]
        tc_cols = list(tc.attrs["columns"])

        radiusCoeffA = np.array([kw["CoeffA"][int(xi) - 1]
                                 for xi in tc[:, tc_cols.index("tc_layer")]])
        minDist = radiusCoeffA + radiusCoeffB * (
            kw["MidRadius"] - np.abs(tc[:, tc_cols.index("tc_eta")])
        )
        seedEn, seedXdivZ, seedYdivZ = sseeds[seedk]

        dRs = np.array([])
        z_tmp = tc[:, tc_cols.index("tc_z")]
        projx = tc[:, tc_cols.index("tc_x")] / z_tmp
        projy = tc[:, tc_cols.index("tc_y")] / z_tmp

        for _, (en, sx, sy) in enumerate(zip(seedEn, seedXdivZ, seedYdivZ)):
            dR = np.sqrt((projx - sx) * (projx - sx) + (projy - sy) * (projy - sy))
            if dRs.shape == (0,):
                dRs = np.expand_dims(dR, axis=-1)
            else:
                dRs = np.concatenate((dRs, np.expand_dims(dR, axis=-1)), axis=1)

        # checks if each event has at least one seed laying below the threshold
        thresh = dRs < np.expand_dims(minDist, axis=-1)

        if not (True in thresh):
            continue

        try:
            # assign TCs to the closest seed (within a maximum distance)
            if pars["cluster_algo"] == "min_distance":
                thresh = np.logical_or.reduce(thresh, axis=1)
                seeds_indexes = np.argmin(dRs, axis=1)

            # energy prioritization (within a maximum distance)
            elif pars["cluster_algo"] == "max_energy":
                filter_en = seedEn * thresh
                thresh = np.max(filter_en, axis=1) > 0
                seeds_indexes = np.argmax(filter_en, axis=1)
                
        except np.AxisError:
            empty_seeds += 1
            continue
        except ValueError:
            if len(seedEn)==0:
                empty_seeds += 1
                continue
            else:
                raise

        seeds_energies = np.array([seedEn[xi] for xi in seeds_indexes])
        # axis 0 stands for trigger cells
        assert tc[:].shape[0] == seeds_energies.shape[0]

        seeds_indexes = np.expand_dims(seeds_indexes[thresh], axis=-1)
        seeds_energies = np.expand_dims(seeds_energies[thresh], axis=-1)
        tc = tc[:][thresh]

        res = np.concatenate((tc, seeds_indexes, seeds_energies), axis=1)

        cols = tc_cols + ["seed_idx", "seed_energy"]
        assert len(cols) == res.shape[1]

        df = pd.DataFrame(res, columns=cols)

        search_str = "{}_([0-9]{{1,7}})_tc".format(kw["FesAlgo"])
        event_number = re.search(search_str, tck)

        event = int(event_number.group(1))

        # Apply photon weights in ECAL to pions
        if pars.particles == "pions":
            hcal = df[ df.tc_layer >= 28 ]
            ecal = df[ df.tc_layer < 28 ]
            ecal = ecal.merge(cfg["phot_weights"], left_on="tc_layer", right_index=True)
            ecal["tc_pt"] = ecal.tc_pt*ecal.weights

            try:
                ecal_cl_pt={"ecal_cl_pt": ecal.tc_pt.sum()}
                ecal_df.loc[event] = ecal_cl_pt
            except AttributeError:
                assert(ecal.empty)
                continue
            
            try:
                df = hcal.groupby(["seed_idx", "tc_layer"], group_keys=True).apply(lambda x: x.tc_pt.sum()).to_frame()
            except AttributeError:
                assert(hcal.empty)
                continue
        
        else:
            df = df.groupby(["seed_idx", "tc_layer"], group_keys=True).apply(lambda x: x.tc_pt.sum()).to_frame()
        
        df.rename({0:"tc_pt_sum"},axis=1,inplace=1)
        # Drop the first layer since this will be dominated by pile up
        #df = df[ df.index.get_level_values("tc_layer") != 1.0 ]

        if pars.particles == "pions":
            # Only calculate weights for HCAL
            df = df[ df.index.get_level_values("tc_layer") > 27 ]

        tc_info[event] = df

    tc_df = pd.concat(tc_info, keys=tc_info.keys(), names=["event"])

    layers = tc_df.unstack(level="tc_layer")
    layers=layers.fillna(0.0)
    
    if pars.particles != "pions":
        return layers
    else:
        return layers, ecal_df

'''def selection(pars, cfg):
    particles = pars.particles
    radius = pars.radius
    radius_str = "/coef_{}".format(str(radius).replace(".", "p"))

    cfg_cl = cfg["clusterStudies"]

    file = cfg_cl["optimization"]["files"][pars.particles][0]
    reprocess = cfg_cl["reprocess"]
    nevents = cfg_cl["nevents"]
    
    # Cuts
    ptcut=cfg_cl["optimization"]["pt_cut"]
    clptcut=cfg_cl["optimization"]["cl_pt_cut"]
    etamin=cfg_cl["optimization"]["eta_min"]
    etamax=cfg_cl["optimization"]["eta_max"]

    with pd.HDFStore(file, mode="r") as clFile:
        #df = clFile["data"]
        #df = clFile[radius_str]
        df = clFile[radius_str]["original"]
        # Cut events that fall out of one sig from the mean
        norm_min = df.pt_norm.mean() - df.pt_norm.std()
        norm_max = df.pt_norm.mean() + df.pt_norm.std()
        df = df[ df.pt_norm > norm_min ]
        df = df[ df.pt_norm < norm_max ]
        df = df[ df.matches == True ]
        df.drop(["pt_norm","en_norm", "dR", "matches"],axis=1, inplace=True)

    df = df[ df.gen_pt > ptcut ]
    df = df[ df.gen_eta > etamin ]
    df = df[ df.gen_eta < etamax ]

    df = df[ df.pt > clptcut ]
    df = df[ df.eta > etamin ]
    df = df[ df.eta < etamax ]

    _, _, ds_tc = data_process.get_data_reco_chain_start(nevents=nevents, reprocess=reprocess, particles=particles)

    ds_tc.set_index("event", inplace=True)
    sub_ds_tc = ds_tc[["tc_layer", "tc_pt"]]
    joined = df.join(sub_ds_tc, on="event", how="inner")


    return joined'''

def select_gen_events(layers, pars, cfg):
    particles = pars.particles

    cfg_cl = cfg["clusterStudies"]

    reprocess = cfg_cl["reprocess"]
    nevents = cfg_cl["nevents"]
    
    ds_gen, _, _ = data_process.get_data_reco_chain_start(nevents=nevents, reprocess=reprocess, particles=particles)

    ds_gen.set_index("event", inplace=True)
    ds_gen["gen_pt"] = ds_gen.gen_en/np.cosh(ds_gen.gen_eta)

    sub_gen = ds_gen.loc[layers.index.get_level_values("event")]

    if particles == "pions":
        ecal_cl_pt = cfg["ecal_cl_pt"]
        sub_gen = sub_gen.merge(ecal_cl_pt, left_index=True, right_index=True)
        sub_gen["hcal_cl_pt"] = sub_gen.gen_pt - sub_gen.ecal_cl_pt
        return sub_gen["hcal_cl_pt"]
    
    else:
        return sub_gen["gen_pt"]

'''def optimize(df, as_df = True):
    layers = df.groupby(["event", "seed_idx", "tc_layer"]).apply(lambda x: x.tc_pt.sum()).to_frame()
    layers.rename({0: "pt"}, axis=1, inplace=True)
    layer_array = layers.unstack(level="tc_layer")
    layer_array = layer_array.fillna(0.0)
    
    # We take the mean because each event contains identical copies
    # of gen info for each tc_layer
    gen_pts = df.groupby(["event", "seed_idx"]).apply(lambda x: x.gen_pt.mean())

    regression = lsq_linear(layer_array, gen_pts,
                            bounds = (0.,2.0),
                            method='bvls',
                            lsmr_tol='auto',
                            verbose=1)

    weights = regression.x

    if as_df:
        index = layer_array.keys().get_level_values("tc_layer").to_list()
        columns = ["weights"]
        weights = pd.DataFrame(data = weights,
                               index = index,
                               columns = columns)

    return weights'''

def optimize(layers, gen_pt, as_df = True):

    regression = lsq_linear(layers, gen_pt,
                            bounds = (0.,5.0),
                            method='bvls',
                            lsmr_tol='auto',
                            verbose=1)

    weights = regression.x

    if as_df:
        index = layers.keys().get_level_values("tc_layer").to_list()
        columns = ["weights"]
        weights = pd.DataFrame(data = weights,
                               index = index,
                               columns = columns)

    return weights

'''def apply_weights(df, weights):
    
    new_df = df.groupby("event", group_keys=True).apply(lambda x: x.merge(weights,left_on="tc_layer",right_index=True))
    new_df["weighted_tc_pt"] = new_df.tc_pt * new_df.weights

    weighted_cl_pt = new_df.groupby("event").apply(lambda x: x.weighted_tc_pt.sum()).to_frame()

    new_df = new_df.groupby("event").apply(lambda x: x.merge(weighted_cl_pt, left_index=True, right_index=True))
    new_df.rename({0: "weighted_pt"}, axis=1, inplace=True)
    new_df.drop(["tc_layer","tc_pt","weights","weighted_tc_pt"],axis=1, inplace=True)
    new_df = new_df.groupby("event").apply(max)
    
    return new_df'''

def apply_weights(pars, cfg, weights):
    radius = round(pars.radius, 3)
    radius_str = "/coef_{}".format(str(radius).replace(".", "p"))

    cfg_cl = cfg["clusterStudies"]

    file = cfg_cl["optimization"]["files"][pars.particles][0]
    
    # Cuts
    ptcut=cfg_cl["optimization"]["pt_cut"]
    clptcut=cfg_cl["optimization"]["cl_pt_cut"]
    etamin=cfg_cl["optimization"]["eta_min"]
    etamax=cfg_cl["optimization"]["eta_max"]

    with pd.HDFStore(file, mode="r") as clFile:
        #df = clFile["data"]
        #df = clFile[radius_str]
        #df = clFile[radius_str]["original"]
        # Cut events that fall out of one sig from the mean
        norm_min = df.pt_norm.mean() - df.pt_norm.std()
        norm_max = df.pt_norm.mean() + df.pt_norm.std()
        df = df[ df.pt_norm > norm_min ]
        df = df[ df.pt_norm < norm_max ]
        df = df[ df.matches == True ]
        df.drop(["pt_norm","en_norm", "dR", "matches"],axis=1, inplace=True)

    df = df[ df.gen_pt > ptcut ]
    df = df[ df.gen_eta > etamin ]
    df = df[ df.gen_eta < etamax ]

    df = df[ df.pt > clptcut ]
    df = df[ df.eta > etamin ]
    df = df[ df.eta < etamax ]

    new_df = df.groupby("event", group_keys=True).apply(lambda x: x.merge(weights,left_on="tc_layer",right_index=True))
    new_df["weighted_tc_pt"] = new_df.tc_pt * new_df.weights

    weighted_cl_pt = new_df.groupby("event").apply(lambda x: x.weighted_tc_pt.sum()).to_frame()

    new_df = new_df.groupby("event").apply(lambda x: x.merge(weighted_cl_pt, left_index=True, right_index=True))
    new_df.rename({0: "weighted_pt"}, axis=1, inplace=True)
    new_df.drop(["tc_layer","tc_pt","weights","weighted_tc_pt"],axis=1, inplace=True)
    new_df = new_df.groupby("event").apply(max)
    
    return new_df

def main(pars, cfg):
    cluster_d = params.read_task_params("cluster")

    cluster_d["CoeffA"] = [pars.radius] * 50

    for key in ("ClusterInTC", "ClusterInSeeds"):
        name = cluster_d[key]
        cluster_d[key] =  "{}_PU0_{}".format(pars.particles, name)
    
    #cluster_d["phot_weights"] = cfg["phot_weights"]
    
    # Assign TCs to Seeds and return pt sums / layer
    if pars.particles == "pions":
        layers, ecal_cl_pt = assign_tcs(pars, **cluster_d)
        cfg["ecal_cl_pt"] = ecal_cl_pt
    else:
        layers = assign_tcs(pars, **cluster_d)      

    # Get gen_pt for events found in layers
    #init_df = selection(pars, cfg)
    gen_pt = select_gen_events(layers, pars, cfg)

    # Linear regression
    #weights = optimize(init_df)
    weights = optimize(layers, gen_pt)

    # Weights are calculate for layers >= 3, add a "weight" of 1 to the first layer
    if pars.particles != "pions":
        weights.loc[1.0] = 1.0
        weights.sort_index(inplace=True)

    #final_df = apply_weights(init_df, weights)
    #final_df = apply_weights(pars, cfg, weights)

    #return weights, final_df
    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--particles", choices=("photons", "electrons", "pions"), required=True)
    parser.add_argument("--radius", help="clustering radius to use: (0.0, 0.05]", required=True, type=float)
    parser.add_argument("--pileup", help="tag for PU200 vs PU0", action="store_true")
    parsing.add_parameters(parser)

    FLAGS = parser.parse_args()
    pars = common.dot_dict(vars(FLAGS))

    phot_dir = "{}/PU0/photons/optimization/official/final/".format(cfg["clusterStudies"]["localDir"])
    out_dir = "{}/PU0/{}/optimization/official/final/".format(cfg["clusterStudies"]["localDir"], pars.particles)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    radius_str = str(round(pars.radius, 4)).replace(".", "p")
    
    phot_name = "{}/{}_r{}.hdf5".format(phot_dir, "optimization_final", radius_str)
    file_name = "{}/{}_r{}.hdf5".format(out_dir, cfg["clusterStudies"]["optimization"]["baseName"], radius_str)

    with pd.HDFStore(phot_name,"r") as photWeights:
        phot_weights = photWeights["weights"]

    cfg["phot_weights"] = phot_weights

    if not os.path.exists(file_name):
        #weights, df = main(pars, cfg)
        weights = main(pars, cfg)
        with pd.HDFStore(file_name, mode="w") as outOpt:
            outOpt["weights"] = pd.concat([phot_weights, weights])
            #outOpt["weights"] = weights
            #outOpt["df"] = df
    else:
        print("\n{} already exists, skipping\n.".format(file_name))
    
