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

        if len(df.seed_idx.unique()) > 1:
            df = df.loc[ df.seed_energy == df.seed_energy.max() ]

        try:
            #df = df.groupby(["seed_idx", "tc_layer"], group_keys=True).apply(lambda x: x.tc_pt.sum()).to_frame()
            #df = df[ df.tc_layer > 28 ]
            df = df.groupby(["tc_layer"], group_keys=True).apply(lambda x: x.tc_pt.sum()).to_frame()
        except AttributeError:
            assert(df.empty)
            continue
        
        df.rename({0:"tc_pt_sum"},axis=1,inplace=1)
        df = df[ df.index.get_level_values("tc_layer") != 1.0 ]


        '''if pars.particles == "pions":
            # Only calculate weights for HCAL
            df = df[ df.index.get_level_values("tc_layer") > 28 ]
        else:
            #Drop the first layer since this will be dominated by pile up
            df = df[ df.index.get_level_values("tc_layer") != 1.0 ]'''

        tc_info[event] = df

    layers = pd.concat(tc_info, keys=tc_info.keys(), names=["event"])

    cl_pt = layers.groupby("event").apply(lambda x: x.tc_pt_sum.sum()).to_frame().rename({0: "cl_pt"}, axis=1)
    #low_pt, high_pt = cl_pt.mean() - cl_pt.std(), cl_pt.mean() + cl_pt.std()
    low_pt, high_pt = cl_pt.mean() - cl_helpers.effrms(cl_pt), cl_pt.mean() + cl_helpers.effrms(cl_pt)
    cl_pt = cl_pt[ cl_pt.cl_pt >= float(low_pt) ]
    cl_pt = cl_pt[ cl_pt.cl_pt <= float(high_pt) ]
    layers = layers.loc[cl_pt.index]

    return layers

def select_gen_events(layers, pars, cfg):
    particles = pars.particles

    cfg_cl = cfg["clusterStudies"]
    reprocess = cfg_cl["reprocess"]
    nevents = cfg_cl["nevents"]

    parquet_tag = cfg_cl["parquetTag"]

    gen_pt_cut = cfg_cl["optimization"]["pt_cut"]
    cl_pt_cut = cfg_cl["optimization"]["cl_pt_cut"]

    eta_min, eta_max = cfg_cl["optimization"]["eta_min"], cfg_cl["optimization"]["eta_max"]
    
    ds_gen, _, _ = data_process.get_data_reco_chain_start(nevents=nevents, reprocess=reprocess, particles=particles, tag=parquet_tag)
    
    ds_gen = ds_gen[ ds_gen.gen_eta > 0 ]

    ds_gen.set_index("event", inplace=True)

    ds_gen["gen_pt"] = ds_gen.gen_en/np.cosh(ds_gen.gen_eta)

    ds_gen = ds_gen[ ds_gen.gen_pt > gen_pt_cut ]
    ds_gen = ds_gen[ ds_gen.gen_eta > eta_min ]
    ds_gen = ds_gen[ ds_gen.gen_eta < eta_max ]

    sub_layers = layers.groupby("event").apply(lambda x: x.tc_pt_sum.sum()).to_frame().rename({0: "cl_pt"}, axis=1)
    sub_layers = sub_layers[ sub_layers.cl_pt > cl_pt_cut ]

    layer_events = sub_layers.index.unique()
    #layer_events = layers.index.get_level_values("event").unique()
    gen_events = ds_gen.index
    events = gen_events.intersection(layer_events)

    ds_gen = ds_gen.loc[events]
    layers = layers.loc[events]

    sub_gen = ds_gen.groupby(level=0).apply(lambda x: x.gen_pt.mean())

    return sub_gen, layers

    '''if particles == "pions":
        ecal_cl_pt = cfg["ecal_cl_pt"]
        sub_gen = sub_gen.merge(ecal_cl_pt, left_index=True, right_index=True)
        sub_gen["hcal_cl_pt"] = sub_gen.gen_pt - sub_gen.ecal_cl_pt
        hcal_cl_pt = sub_gen.groupby(level=0).apply(lambda x: x.hcal_cl_pt.mean())
        return hcal_cl_pt
    
    else:
        return sub_gen["gen_pt"]'''

def optimize(layers, gen_pt, radius):
    layers = layers.sort_index(level="event")
    layers = layers.unstack(level="tc_layer").fillna(0.0)

    '''if radius < 0.005:
        max_weight = 5.0
    elif (radius >= 0.005 and radius <= 0.01):
        max_weight = 3.0
    else:
        max_weight = 2.0'''
    
    max_weight = 5.0

    if not layers.empty:

        regression = lsq_linear(layers, gen_pt,
                                bounds = (0.,max_weight),
                                method='bvls',
                                lsmr_tol='auto',
                                verbose=1)

        weights = regression.x

        index = layers.keys().get_level_values("tc_layer").to_list()
        columns = ["weights"]
        weights = pd.DataFrame(data = weights,
                    index = index,
                    columns = columns)
    else:
        layer_index = [2*i+1 for i in range(14)]
        weights = pd.DataFrame(index=layer_index, columns = ["weights"])
        for layer in layer_index:
            weights.loc[layer] = max_weight

    return weights

def layer_weights(pars, cfg):
    radius = pars.radius

    cluster_d = params.read_task_params("cluster")

    cluster_d["CoeffA"] = [pars.radius] * 50

    for key in ("ClusterInTC", "ClusterInSeeds"):
        name = cluster_d[key]
        cluster_d[key] =  "{}_PU0_{}_posEta".format(pars.particles, name)

    layers = assign_tcs(pars, **cluster_d)
    
    # Assign TCs to Seeds and return pt sums / layer
    '''if pars.particles == "pions":
        cluster_d["phot_weights"] = cfg["phot_weights"]
        layers, ecal_cl_pt = assign_tcs(pars, **cluster_d)
        cfg["ecal_cl_pt"] = ecal_cl_pt
    else:
        layers = assign_tcs(pars, **cluster_d) '''     

    # Get gen_pt for events found in layers
    gen_pt, layers = select_gen_events(layers, pars, cfg)

    # Linear regression
    weights = optimize(layers, gen_pt, radius)

    weights.loc[1.0] = 1.0
    weights.sort_index(inplace=True)

    '''if pars.particles != "pions":
    # Weights are calculate for layers >= 3, add a "weight" of 1 to the first layer
        weights.loc[1.0] = 1.0
        weights.sort_index(inplace=True)'''

    return weights

def eta_correction(df_pu):
    X = df_pu[["eta"]]
    y = df_pu.gen_pt - df_pu.pt
    corr = LinearRegression(fit_intercept=False).fit(X, y)
    return corr

def apply_eta_corr(df_pu, corr):
    if not isinstance(corr, pd.DataFrame):
        df_pu["eta_corr_to_pt"] = corr.predict(df_pu[["eta"]])
    else:
        df_pu["intercept"], df_pu["coef"] = corr.intercept_.iloc[0], corr.coef_.iloc[0]
        df_pu["eta_corr_to_pt"] = df_pu["intercept"] + df_pu["eta"]*df_pu["coef"]
        df_pu.drop(["intercept", "coef"], axis=1, inplace=True)
    df_pu["pt_corr_eta"] = df_pu.eta_corr_to_pt + df_pu.pt
    df_pu["pt_norm"] = df_pu.pt / df_pu.gen_pt
    df_pu["pt_norm_eta_corr"] = df_pu.pt_corr_eta / df_pu.gen_pt
    return df_pu

def eta_corr_per_rad(clFile, etaFile, radius, eta_corr = None):
    dfs = clFile[radius]

    df_cl = dfs["weighted"]

    upper_eff = df_cl.pt_norm.mean() + cl_helpers.effrms(df_cl.pt_norm.to_frame().rename({0: "pt_norm"}))
    lower_eff = df_cl.pt_norm.mean() - cl_helpers.effrms(df_cl.pt_norm.to_frame().rename({0: "pt_norm"}))

    df_cl = df_cl[ (df_cl.pt_norm > lower_eff) & (df_cl.pt_norm < upper_eff) ]

    corr = eta_correction(df_cl) if not isinstance(eta_corr, pd.DataFrame) else eta_corr
    new_df = apply_eta_corr(df_cl, corr)

    dfs["weighted"] = new_df

    if not isinstance(eta_corr, pd.DataFrame):
        etaFile[radius] = pd.DataFrame.from_dict(corr.__dict__)

    clFile[radius] = dfs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--particles", choices=("photons", "electrons", "pions"), required=True)
    parser.add_argument("--radius", help="clustering radius to use: (0.0, 0.05]", type=float)
    parser.add_argument("--pileup", help="tag for PU200 vs PU0", action="store_true")
    parser.add_argument("--etaCal", help="path to eta-calibration file", type=str)
    parsing.add_parameters(parser)

    FLAGS = parser.parse_args()
    pars = common.dot_dict(vars(FLAGS))

    pileup_dir = "PU0" if not pars.pileup else "PU200"

    if pars.radius != None:
        radius_str = str(round(pars.radius, 4)).replace(".", "p")
    
    if not pars.pileup:
        out_dir = "{}/{}/{}/optimization/official/final/".format(cfg["clusterStudies"]["localDir"], pileup_dir, pars.particles)
        os.makedirs(out_dir, exist_ok=True)

        file_layer_weights = "{}/{}.hdf5".format(out_dir, cfg["clusterStudies"]["optimization"]["baseName"])
        
        if not os.path.exists(file_layer_weights):
            weights = layer_weights(pars, cfg)
            with pd.HDFStore(file_layer_weights, mode="w") as outOpt:
                outOpt["weights"] = weights
        else:
            print("\n{} already exists, skipping\n.".format(file_layer_weights))

    else:
        if pars.radius != None:
            radius_key = "coef_{}".format(radius_str)
        #cl_file = cfg["clusterStudies"]["optimization"]["PU200"][pars.particles]
        #cl_file = "/home/llr/cms/ehle/NewRepos/bye_splits/data/new_algos/PU0/photons/cluster_size_etaCal_test.hdf5"
        #cl_file = "data/new_algos/PU200/electrons/cluster_size_negEta_weighted_selectOneEffRms_maxSeed_ThresholdDummyHistomaxnoareath20_filtered.hdf5"
        #cl_file = "data/new_algos/PU200/pions/cluster/weighted/selectOneStd/maxSeed/posEta/smooth/cluster_size_fullEta_weighted_selectOneEffRms_maxSeed_ThresholdDummyHistomaxnoareath20_filtered.hdf5"
        #cl_file = "data/new_algos/PU200/pions/cluster/weighted/selectOneStd/maxSeed/smooth/bc_stc/posEta/cluster_size_fullEta_weighted_selectOneEffRms_maxSeed_ThresholdDummyHistomaxnoareath20_filtered.hdf5"
        #cl_file = "data/new_algos/PU200/pions/cluster/weighted/selectOneStd/maxSeed/smooth/bc_stc/negEta/cluster_size_fullEta_weighted_selectOneEffRms_maxSeed_ThresholdDummyHistomaxnoareath20_filtered.hdf5"
        cl_file = "data/new_algos/PU200/pions/cluster/weighted/selectOneStd/maxSeed/smooth/bc_stc/posEta/cluster_size_fullEta_weighted_selectOneEffRms_maxSeed_ThresholdDummyHistomaxnoareath20_filtered_lessCols.hdf5"

        if pars.etaCal == None:
            #eta_file = os.path.dirname(cl_file) + "/etaCorr_fromCenter_new.hdf5"
            eta_file = os.path.dirname(cl_file) + "/etaCorr_noIntercept.hdf5"
            print("\nWriting: ", eta_file)
            with pd.HDFStore(cl_file, "a") as clFile:
                if pars.radius != None:
                    with pd.HDFStore(eta_file, "w") as etaFile:
                        eta_corr_per_rad(clFile, etaFile, radius_key)

                else:
                    for radius in clFile.keys():
                        if radius == clFile.keys()[0]:
                            with pd.HDFStore(eta_file, "w") as etaFile:
                                eta_corr_per_rad(clFile, etaFile, radius)
                        else:
                            with pd.HDFStore(eta_file, "a") as etaFile:
                                eta_corr_per_rad(clFile, etaFile, radius)

        else:
            eta_file = pars.etaCal
            with pd.HDFStore(cl_file, "a") as clFile, pd.HDFStore(eta_file, "r") as etaFile:
                if pars.radius != None:
                    eta_corr = etaFile[radius_key]
                    eta_corr_per_rad(clFile, etaFile, radius_key, eta_corr=eta_corr)
                    
                else:
                    for radius in etaFile.keys():
                        eta_corr = etaFile[radius]
                        eta_corr_per_rad(clFile, etaFile, radius, eta_corr=eta_corr)
    
