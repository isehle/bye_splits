# coding: utf-8

_all_ = ["cluster"]

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits import utils
from bye_splits.utils import common, params

import re
import yaml
import numpy as np
import pandas as pd
import h5py

with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)

#layers = cfg["selection"]["disconnectedTriggerLayers"]
#layers = np.asarray([l-1 for l in layers])

def write_columns(df, kw):
    if "weights" in kw.keys():
        weights = kw["weights"]
        df = df.merge(weights, left_on="tc_layer",right_index=True)
        df["weighted_tc_mipPt"] = df.tc_mipPt * df.weights
        df["weighted_tc_pt"] = df.tc_pt * df.weights


    df["cl3d_pos_x"] = df.tc_x * df.tc_mipPt
    df["cl3d_pos_y"] = df.tc_y * df.tc_mipPt
    df["cl3d_pos_z"] = df.tc_z * df.tc_mipPt

    if "weights" in kw.keys():
        df["weighted_cl3d_pos_x"] = df.tc_x * df.weighted_tc_mipPt
        df["weighted_cl3d_pos_y"] = df.tc_y * df.weighted_tc_mipPt
        df["weighted_cl3d_pos_z"] = df.tc_z * df.weighted_tc_mipPt

    cl3d_cols = ["cl3d_pos_x", "cl3d_pos_y", "cl3d_pos_z", "tc_mipPt", "tc_pt"]
    if "weights" in kw.keys():
        cl3d_cols = cl3d_cols + ["weighted_{}".format(col) for col in cl3d_cols]
        #cl3d = df.groupby(["seed_idx"], group_keys=True).sum()[cl3d_cols]
        # While the above should be equivalent, for some reason it drops the weighted columns, resulting in a KeyError
    cl3d = df.groupby(["seed_idx"], group_keys=True).apply(lambda x: x.sum())[cl3d_cols]
    
    dR_by_layer = df.groupby(["seed_idx", "tc_layer"],group_keys=True).apply(lambda x: x.seed_dRs.mean())
    for seed in dR_by_layer.index.get_level_values("seed_idx"):
        dR_layers = dR_by_layer.loc[seed].index
        layers_no_vals = np.setdiff1d(layers, dR_layers)
        for layer in layers_no_vals:
            #dR_by_layer.loc[seed, layer] = 0.0
            dR_by_layer.loc[seed, layer] = np.NaN
    
    dR_by_layer.sort_index(inplace=True)

    for seed, layer in dR_by_layer.index:
        dR = dR_by_layer.loc[seed, layer]
        '''if dR == 0:
            shift = 2.0 if layer < 28 else 1.0
            layer_min = layer-shift if layer != 1.0 else layer
            #layer_max = layer+shift if layer != 50.0 else layer
            layer_max = layer+shift if layer != 27.0 else layer
            dR = (dR_by_layer.loc[seed, layer_min] + dR_by_layer.loc[seed, layer_max])/2'''
        
        col_name = "layer_" + str(int(layer)) + "_dR"
        cl3d[col_name] = dR
    
    cl3d = cl3d.rename(
        columns={
            "cl3d_pos_x": "x",
            "cl3d_pos_y": "y",
            "cl3d_pos_z": "z",
            "tc_mipPt": "mipPt",
            "tc_pt": "pt",
        }
    )

    if "weights" in kw.keys():
        columns={
            "weighted_cl3d_pos_x": "weighted_x",
            "weighted_cl3d_pos_y": "weighted_y",
            "weighted_cl3d_pos_z": "weighted_z",
            "weighted_tc_mipPt": "weighted_mipPt",
            "weighted_tc_pt": "weighted_pt",
        }
        for layer in layers:
            col_name = "tc_pt_layer_" + str(layer) + "_frac"
            columns.update({col_name: col_name}) 
        cl3d = cl3d.rename(
            columns=columns
        )

    cl3d = cl3d[cl3d.pt > kw["PtC3dThreshold"]] if "weights" not in kw.keys() else cl3d[cl3d.weighted_pt > kw["PtC3dThreshold"]]

    if "weights" not in kw.keys():
        cl3d.loc[:, ["x", "y", "z"]] = cl3d.loc[:, ["x", "y", "z"]].div(
            cl3d.mipPt, axis=0
        )
    else:
         cl3d.loc[:, ["weighted_x", "weighted_y", "weighted_z"]] = cl3d.loc[:, ["weighted_x", "weighted_y", "weighted_z"]].div(
            cl3d.weighted_mipPt, axis=0
        )       

    cl3d_dist = np.sqrt(cl3d.x**2 + cl3d.y**2)
    cl3d["phi"] = np.arctan2(cl3d.y, cl3d.x)
    cl3d["eta"] = np.arcsinh(cl3d.z / cl3d_dist)
    cl3d["Rz"] = common.calcRzFromEta(cl3d.eta)
    cl3d["en"] = cl3d.pt * np.cosh(cl3d.eta)

    if "weights" in kw.keys():
        weighted_cl3d_dist = np.sqrt(cl3d.weighted_x**2 + cl3d.weighted_y**2)
        cl3d["weighted_phi"] = np.arctan2(cl3d.weighted_y, cl3d.weighted_x)
        cl3d["weighted_eta"] = np.arcsinh(cl3d.weighted_z / weighted_cl3d_dist)
        cl3d["weighted_Rz"] = common.calcRzFromEta(cl3d.weighted_eta)
        cl3d["weighted_en"] = cl3d.weighted_pt * np.cosh(cl3d.weighted_eta)

        tc_pt_per_layer = df.groupby("tc_layer").apply(lambda x: x.weighted_tc_pt.sum())
        tc_pt_layer_frac = tc_pt_per_layer / tc_pt_per_layer.sum()
        transv_size = df.groupby("tc_layer").apply(lambda x: np.sqrt(x.tc_x**2+x.tc_y**2).mean())
        for layer in layers:
            pt_col_name = "tc_pt_layer_" + str(layer) + "_frac"
            rad_col_name = "r_layer_" + str(layer)
            try:
                cl3d[pt_col_name] = tc_pt_layer_frac.loc[layer]
                cl3d[rad_col_name] = transv_size.loc[layer]
            except KeyError:
                cl3d[pt_col_name] = 0.0 # catch events with zero tcs in a given layer
                cl3d[rad_col_name] = 0.0
    return cl3d

def cluster(pars, in_seeds, in_tc, out_valid, out_plot, **kw):
    dfout = None
    sseeds = h5py.File(in_seeds, mode='r')
    sout = pd.HDFStore(out_valid, mode='w')
    stc = h5py.File(in_tc, mode='r')
        
    seed_keys = [x for x in sseeds.keys() if '_group' in x and 'central' not in x]
    tc_keys = [x for x in stc.keys() if '_tc' in x and 'central' not in x]
    assert len(seed_keys) == len(tc_keys)

    radiusCoeffB = kw["CoeffB"]
    empty_seeds = 0
    bad_seeds = 0
    
    global layers
    layers = cfg["selection"]["disconnectedTriggerLayers"]
    layers = np.asarray([l-1 for l in layers])

    if "pion" in in_seeds:
        hcal_layers = np.asarray([i for i in range(28,51)])
        layers = np.concatenate((layers, hcal_layers))

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
        seeds_dRs = np.array([dRs[xi] for xi in seeds_indexes])
        # axis 0 stands for trigger cells
        assert tc[:].shape[0] == seeds_energies.shape[0]
        
        seeds_indexes = np.expand_dims(seeds_indexes[thresh], axis=-1)
        seeds_energies = np.expand_dims(seeds_energies[thresh], axis=-1)
        #seeds_dRs = np.expand_dims(seeds_dRs[thresh], axis=-1)
        seeds_dRs = dRs[thresh]
        seeds_dRs = seeds_dRs[np.arange(len(seeds_indexes)), seeds_indexes[:,0]].reshape(-1,1)
        tc = tc[:][thresh]

        res = np.concatenate((tc, seeds_indexes, seeds_energies, seeds_dRs), axis=1)

        key = tck.replace("_tc", "_cl")
        cols = tc_cols + ["seed_idx", "seed_energy", "seed_dRs"]
        assert len(cols) == res.shape[1]

        df = pd.DataFrame(res, columns=cols)
        assert not df.empty

        '''if "weights" in kw.keys():
            df = df.merge(weights, left_on="tc_layer",right_index=True)
            df["weighted_tc_mipPt"] = df.tc_mipPt * df.weights
            df["weighted_pt"] = df.tc_pt * df.weights

        df["cl3d_pos_x"] = df.tc_x * df.tc_mipPt
        df["cl3d_pos_y"] = df.tc_y * df.tc_mipPt
        df["cl3d_pos_z"] = df.tc_z * df.tc_mipPt

        cl3d_cols = ["cl3d_pos_x", "cl3d_pos_y", "cl3d_pos_z", "tc_mipPt", "tc_pt"]
        cl3d = df.groupby(["seed_idx"]).sum()[cl3d_cols]
        cl3d = cl3d.rename(
            columns={
                "cl3d_pos_x": "x",
                "cl3d_pos_y": "y",
                "cl3d_pos_z": "z",
                "tc_mipPt": "mipPt",
                "tc_pt": "pt",
            }
        )

        cl3d = cl3d[cl3d.pt > kw["PtC3dThreshold"]]
        cl3d.loc[:, ["x", "y", "z"]] = cl3d.loc[:, ["x", "y", "z"]].div(
            cl3d.mipPt, axis=0
        )

        cl3d_dist = np.sqrt(cl3d.x**2 + cl3d.y**2)
        cl3d["phi"] = np.arctan2(cl3d.y, cl3d.x)
        cl3d["eta"] = np.arcsinh(cl3d.z / cl3d_dist)
        cl3d["Rz"] = common.calcRzFromEta(cl3d.eta)
        cl3d["en"] = cl3d.pt * np.cosh(cl3d.eta)'''
        
        cl3d = write_columns(df, kw)

        search_str = "{}_([0-9]{{1,7}})_tc".format(kw["FesAlgo"])
        event_number = re.search(search_str, tck)
        if not event_number:
            m = "The event number was not extracted!"
            raise ValueError(m)

        cl3d["event"] = event_number.group(1)
        cl3d_cols = ["en", "x", "y", "z", "Rz", "eta", "phi"]
        layer_dRs = ["layer_{}_dR".format(l) for l in layers]
        cl3d_cols = cl3d_cols + layer_dRs
        if "weights" in kw.keys():
            cl3d_cols = cl3d_cols + ["weighted_{}".format(col) for col in cl3d_cols]
            cl3d_cols = cl3d_cols + [col for col in cl3d.keys() if "layer" in col]
            '''for layer in layers:
                col_name = "tc_pt_layer_" + str(layer) + "_frac"
                cl3d_cols = cl3d_cols + [col_name]'''

        sout[key] = cl3d[cl3d_cols]
        if tck == tc_keys[0] and seedk == seed_keys[0]:
            dfout = cl3d[cl3d_cols + ["event"]]
        else:
            dfout = pd.concat((dfout, cl3d[cl3d_cols + ["event"]]), axis=0)

    print("[clustering step] There were {} events without seeds.".format(empty_seeds))
    print("[clustering step] Bad seeds: {}%.".format(bad_seeds/len(seedk)))

    splot = pd.HDFStore(out_plot, mode='w')
    print("\nWriting: ", splot)
    if dfout is not None:
        dfout.event = dfout.event.astype(int)
        splot["data"] = dfout        
    else:
        mes = "No output in the cluster."
        raise RuntimeError(mes)

    nevents = dfout.event.unique().shape[0]
    sseeds.close()
    sout.close()
    stc.close()
    splot.close()
    if "returnDF" in kw.keys():
        return nevents, dfout
    else:
        return nevents

def cluster_default(pars, **kw):
    in_seeds  = common.fill_path(kw["ClusterInSeeds"], **pars)
    in_tc     = common.fill_path(kw["ClusterInTC"], **pars)
    out_valid = common.fill_path(kw["ClusterOutValidation"], **pars)
    out_plot  = common.fill_path(kw["ClusterOutPlot"], **pars)
    return cluster(pars, in_seeds, in_tc, out_valid, out_plot, **kw)
    
def cluster_roi(pars, **kw):
    '''with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)'''
    extra_name = '_hexdist' if cfg['seed_roi']['hexDist'] else ''
    
    in_seeds  = common.fill_path(kw["ClusterInSeedsROI"] + extra_name, **pars)
    if cfg['cluster']['ROICylinder']:
        in_tc = common.fill_path(kw["ClusterInTCROICylinder"], **pars)
        out_valid = common.fill_path(kw["ClusterOutValidationROI"] + '_cyl', **pars)
        out_plot  = common.fill_path(kw["ClusterOutPlotROI"] + '_cyl', **pars)
    else:
        in_tc = common.fill_path(kw["ClusterInTCROI"], **pars)
        out_valid = common.fill_path(kw["ClusterOutValidationROI"], **pars)
        out_plot  = common.fill_path(kw["ClusterOutPlotROI"], **pars)

    return cluster(pars, in_seeds, in_tc, out_valid, out_plot, **kw)

if __name__ == "__main__":
    import argparse
    from bye_splits.utils import params, parsing

    parser = argparse.ArgumentParser(description="Clustering standalone step.")
    parsing.add_parameters(parser)
    parser.add_argument('--roi', action='store_true',
                        help='Cluster on ROI chain output.')
    FLAGS = parser.parse_args()

    cluster_d = params.read_task_params("cluster")
    if FLAGS.roi:
        cluster_roi(vars(FLAGS), **cluster_d)
    else:
        cluster_default(vars(FLAGS), **cluster_d)
