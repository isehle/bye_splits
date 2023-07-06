import os
import sys
import re
import argparse

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import math
import scipy.stats as st

import yaml

from bye_splits.utils import common, params

import matplotlib.pyplot as plt
import mplhep as hep

annot_str = lambda x: str(x).replace(".", "p")

def closest(list, k=0.0):
    """Find the element of a list containing strings ['coef_{float_1}', 'coef_{float_2}', ...] which is closest to some float_i"""
    try:
        list = np.reshape(np.asarray(list), 1)
    except ValueError:
        list = np.asarray(list)
    if isinstance(k, str):
        k_num = float(re.split("coef_", k)[1].replace("p", "."))
    else:
        k_num = k
    id = (np.abs(list - k_num)).argmin()
    return list[id]


def get_str(coef, df_dict):
    """Accepts a coefficient, either as a float or string starting with coef_, along with a dictionary of coefficient:DataFrame pairs.
    Returns the coefficient string in the dictionary that is the closest to the passed coef.
    """
    if not isinstance(coef, str):
        coef_str = "coef_{}".format(str(coef).replace(".", "p"))
    else:
        coef_str = coef
    if coef_str not in df_dict.keys():
        coef_list = [
            float(re.split("coef_", key)[1].replace("p", ".")) for key in df_dict.keys()
        ]
        new_coef = closest(coef_list, coef)
        coef_str = "/coef_{}".format(str(new_coef).replace(".", "p"))
    return coef_str


'''def get_dfs(init_files, coef, weighted=False):
    """Takes a dictionary of input files (keys corresponding to particles, values corresponding to file paths containing DataFrames by coefficient), with a desired coefficient.
    Returns a new dictionary with the same keys, whose values correspond to the DataFrame of that particular coefficient.
    """
    df_dict = dict.fromkeys(init_files.keys(), [0.0])
    for key in init_files.keys():
        file = pd.HDFStore(init_files[key], "r")
        if not isinstance(coef, str):
            coef = get_str(coef, file)
        if not coef in file.keys():
            coef = file.keys()[0]
        df = file[coef]["original"] if not weighted else file[coef]["weighted"]
        file.close()
        df_dict[key] = df
    return common.dot_dict(df_dict)'''

def get_dfs(init_files, coef, particles="photons"):
    """Takes a dictionary of input files (keys corresponding to particles, values corresponding to file paths containing DataFrames by coefficient), with a desired coefficient.
    Returns a new dictionary with the same keys, whose values correspond to the DataFrame of that particular coefficient.
    """
    file = pd.HDFStore(init_files[particles], "r")
    if not isinstance(coef, str):
        coef = get_str(coef, file)
    if not coef in file.keys():
        coef = file.keys()[0]
    df_o, df_w= file[coef]["original"], file[coef]["weighted"]
    file.close()
    return df_o, df_w


def get_keys(init_files):
    """Returns the list of exact coefficient keys in the initial files; they're the same for all files, so we only need to check one."""
    file_path = init_files["photons"]

    with pd.HDFStore(file_path, "r") as file:
        keys = file.keys()

    return keys


def read_cl_size_params():
    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)
    return cfg["clusterStudies"]


'''def filter_dfs(dfs_by_particle, eta_range, pt_cut):
    filtered_dfs = {}
    for particle, df in dfs_by_particle.items():
        if not isinstance(df, list) and not df.empty:
            with common.SupressSettingWithCopyWarning():
                filtered_dfs[particle] = df[ (df.gen_eta > eta_range[0]) & (df.gen_eta < eta_range[1]) & (df.pt > pt_cut) ]
                #df = df[ df.matches == True ]
                #filtered_dfs[particle] = df.groupby("event").apply(lambda x: x.loc[x.pt.idxmax()])
    return filtered_dfs'''

def filter_dfs(df, eta_range, pt_cut):
    with common.SupressSettingWithCopyWarning():
        filtered_df = df[ (abs(df.gen_eta) > eta_range[0]) & (df.gen_eta < eta_range[1]) & (df.gen_pt > pt_cut) ]
        #filtered_df = df[ (abs(df.gen_eta) > eta_range[0]) & (df.gen_eta < eta_range[1]) & (df.gen_pt > pt_cut[0]) & (df.gen_pt < pt_cut[1]) ]        
    return filtered_df

def effrms(data, c=0.68):
    """Compute half-width of the shortest interval
    containing a fraction 'c' of items in a 1D array.
    """
    if isinstance(data, pd.DataFrame):
        assert(len(data.keys())==1)
        key = list(data.keys())[0]
        sorted = data.sort_values(by=key)
        m = int(c* len(sorted)) + 1
        low, high = sorted.iloc[:-m], sorted.iloc[m:]
        try:
            new = high.reset_index()-low.reset_index()
            out = np.min(new[key]) / 2.0
        except TypeError:
            new = high.reset_index()[key]-low.reset_index()[key]
            out = np.min(new) / 2.0
    else:
        assert data.shape == (data.shape[0],)
        new_series = data.dropna()
        x = np.sort(new_series, kind="mergesort")
        m = int(c * len(x)) + 1
        out = np.min(x[m:] - x[:-m]) / 2.0

    return out


def get_y_max(dist, nbins=100):
    hist, _ = np.histogram(dist, bins=nbins)
    return np.max(hist)

'''def read_weights(dir, cfg, version="final", mode="weights"):
    weights_by_particle = {}
    for particle in ("photons", "electrons", "pions"):
        if particle == "photons":
            basename = "optimization_selectOnestd_adjustMaxWeight"
        elif particle == "electrons":
            basename = "optimization_selectOnestd"
        else:
            basename = "optimization_bound5_selectOnestd"
        
        version_dir = "{}/".format(version)
        particle_dir = "{}{}/optimization/official/{}".format(dir, particle, version_dir)

        files = [f for f in os.listdir(particle_dir) if basename in f]
        weights_by_radius = {}
        for file in files:
            radius = float(file.replace(".hdf5","").replace(f"{basename}_","").replace("r0","0").replace("p","."))
            infile = particle_dir+file
            with pd.HDFStore(infile, "r") as optWeights:
                weights_by_radius[radius] = optWeights[mode]
    
        weights_by_particle[particle] = weights_by_radius
    
    return weights_by_particle'''

def read_weights(dir, cfg, version="final", mode="weights"):
    weights_by_particle = {}
    #for particle in ("photons", "electrons", "pions"):
    for particle in ("photons", "pions"):
        basename = "optimization_selectOneEffRms_maxSeed_bc_stc" if particle == "pions" else "optimization_selectOneStd_adjustMaxWeight_maxSeed"
        
        version_dir = "{}/".format(version)
        particle_dir = "{}{}/optimization/official/{}".format(dir, particle, version_dir)

        files = [f for f in os.listdir(particle_dir) if basename in f]
        weights_by_radius = {}
        for file in files:
            radius = float(file.replace(".hdf5","").replace(f"{basename}_","").replace("r0","0").replace("p","."))
            infile = particle_dir+file
            with pd.HDFStore(infile, "r") as optWeights:
                weights_by_radius[radius] = optWeights[mode]
    
        weights_by_particle[particle] = weights_by_radius
    
    weights_by_particle["electrons"] = weights_by_particle["photons"]
    
    return weights_by_particle

def fill_filename(filename, weight, pileup, eta_corr=0):
    """Takes a base <filename> and three ints which correspond
    to the number of clicks of a given button. Updates the output
    filename accordingly."""

    if eta_corr%2!=0:
        assert(pileup%2!=0)
        filename += "_etaCorrected"
    
    filename += "_PU200" if pileup%2!=0 else "_PU0"

    filename += "_weighted.hdf5" if weight%2!=0 else ".hdf5"

    return filename

def get_global_res(file, eta_range, pt_cut, pileup, col="pt_norm"):
    pt_cut = float(pt_cut) if pt_cut != "PT Cut" else 0.0
    cols = ["pt_norm"] if pileup=="PU0" else ["pt_norm", "pt_norm_eta_corr"]
    original, weighted, eta = [], [], []
    original_eff, weighted_eff, eta_eff = [], [], []
    with pd.HDFStore(file, mode="r") as dfs:
        for radius in dfs.keys():
            df_o, df_w = dfs[radius]["original"], dfs[radius]["weighted"]
            df_o = df_o[ (df_o.eta > eta_range[0]) & (df_o.eta < eta_range[1]) & (df_o.gen_pt > pt_cut) ]
            df_w = df_w[ (df_w.eta > eta_range[0]) & (df_w.eta < eta_range[1]) & (df_w.gen_pt > pt_cut) ]
            #df_o = df_o[ (df_o.eta > eta_range[0]) & (df_o.eta < eta_range[1]) & (df_o.gen_pt > pt_cut[0]) & (df_o.gen_pt < pt_cut[1]) ]
            #df_w = df_w[ (df_w.eta > eta_range[0]) & (df_w.eta < eta_range[1]) & (df_w.gen_pt > pt_cut[0]) & (df_w.gen_pt < pt_cut[1]) ]
            res_original = df_o["pt_norm"].std()/df_o["pt_norm"].mean()
            res_original_eff = effrms(df_o["pt_norm"])/df_o["pt_norm"].mean()
            original.append(res_original)
            original_eff.append(res_original_eff)
            for col in cols:
                res_weighted = df_w[col].std()/df_w[col].mean()
                res_weighted_eff = effrms(df_w[col])/df_w[col].mean()
                weighted.append(res_weighted) if col == "pt_norm" else eta.append(res_weighted)
                weighted_eff.append(res_weighted_eff) if col == "pt_norm" else eta_eff.append(res_weighted_eff)
        res = {"original": {"rms": original, "eff": original_eff},
                "layer": {"rms": weighted, "eff": weighted_eff}}
        if len(eta) > 1:
            res.update({"eta": {"rms": eta, "eff": eta_eff}})
    
    return res

def get_global_pt(file, eta_range, pt_cut, pileup, mode="mean"):
    pt_cut = float(pt_cut) if pt_cut != "PT Cut" else 0.0
    cols = ["pt_norm"] if pileup=="PU0" else ["pt_norm", "pt_norm_eta_corr"]
    original, weighted, eta = [0.0], [0.0], [0.0]
    with pd.HDFStore(file, mode="r") as dfs:
        for radius in dfs.keys():
            df_o, df_w = dfs[radius]["original"], dfs[radius]["weighted"]
            df_o, df_w = filter_dfs(df_o, eta_range, pt_cut), filter_dfs(df_w, eta_range, pt_cut)
            '''df_o = df_o[ (df_o.eta > eta_range[0]) & (df_o.eta < eta_range[1]) & (df_o.gen_pt > pt_cut) ]
            df_w = df_w[ (df_w.eta > eta_range[0]) & (df_w.eta < eta_range[1]) & (df_w.gen_pt > pt_cut) ]'''
            if mode=="mean":
                mean_original = df_o["pt_norm"].mean()
            else:
                counts, values = np.histogram(df_o["pt_norm"],bins=1000)
                mode_index = np.argmax(counts)
                mean_original = values[mode_index]
            original.append(mean_original)
            for col in cols:
                if mode=="mean":
                    mean_weighted = df_w[col].mean()
                else:
                    counts, values = np.histogram(df_w[col],bins=1000)
                    mode_index = np.argmax(counts)
                    mean_weighted = values[mode_index]
                weighted.append(mean_weighted) if col == "pt_norm" else eta.append(mean_weighted)
        pt_mean = {"original": original,
                   "layer": weighted}
        if len(eta) > 1:
            pt_mean.update({"eta": eta})
    
    return pt_mean

def get_dataframes(init_files, particles, coef, eta_range, pt_cut):
    df_o, df_w = get_dfs(init_files, coef, particles)
    df_o["gen_eta"], df_w["gen_eta"] = abs(df_o.gen_eta), abs(df_w.gen_eta)
    pt_cut = float(pt_cut) if pt_cut!="PT Cut" else 0
    df_o, df_w = filter_dfs(df_o, eta_range, pt_cut), filter_dfs(df_w, eta_range, pt_cut)

    return df_o, df_w

def get_hline_vals(lower_dist, higher_dist, step=0.02):
    min, max = float(lower_dist.min()), float(higher_dist.max())
    min = round( min / step ) * step
    max = math.ceil( max / step ) * step + step
    return np.arange(min, max, step=step)

def rel_err(df, col=None, bin_col=None):
    """Takes a dataframe and a column key to calculate the error+relative error of.
    Optionally accepts a supplementary 'bin column' if you wish to calculate errors in
    specific bins instead of across the entire distribution.
    Standard error is calculated as sigma/sqrt(N), either per bin or for the whole distribution.
    Relative error is the standard_error/value, where the 'value' in the binned case corresponds to the
    mean in a given bin.
    In the binned case, returns error and relative error, while the total case adds these values as columns to the dataframe."""
    if bin_col != None:
        assert(col!=None)
        mean = df.groupby(bin_col).apply(lambda x: x[col].mean())
        sig = df.groupby(bin_col).apply(lambda x: x[col].std())
        sqrtN = df.groupby(bin_col).apply(lambda x: np.sqrt(len(x[col])))

        err = sig/sqrtN
        rel_err = err/mean    
    
    else:
        sig = df.std()
        sqrtN = np.sqrt(len(df))
        
        err = sig/sqrtN
        rel_err = [err / val for val in df]

    return err, rel_err

def get_binned_modes(df, bin_col, var_col, bins=50):
    test = df.groupby(bin_col).apply(lambda x: pd.cut(x[var_col], bins=bins, labels=False))
    test = test.to_frame()
    test.index = test.index.rename([bin_col,'idx'])
    test = test.droplevel('idx')
    var_bin_col = var_col + "_bin"
    test = test.rename({var_col: var_bin_col}, axis=1)
    modes = test.groupby(bin_col).apply(lambda x: st.mode(x[var_bin_col])[0]).to_frame()
    modes = modes.rename({0: var_bin_col}, axis=1)
    df = df.merge(modes, left_on=bin_col, right_index=True, how='left')
    binned_vals = df.groupby(bin_col).apply(lambda x: np.histogram(x[var_col], bins=bins)[1])
    for bin in df[bin_col].unique():
        norm_bin = df[ df[bin_col] == bin ][var_bin_col].unique()[0]
        binned_vals.loc[bin] = binned_vals.loc[bin][norm_bin]
    new_var = var_col + "_binned"
    binned_vals = binned_vals.to_frame().rename({0: new_var},axis=1)
    #print(df)
    df = df.merge(binned_vals, left_on=bin_col, right_index=True, how="left")
    #print(df)
    return df.groupby(bin_col).apply(lambda x: x[new_var].mean())
    
def write_cms_plots(**kwargs):
    hep.style.use("CMS")
    #hep.cms.text("Simulation",loc=1)
    plt.rcParams['text.usetex'] = True
    plt.grid(visible=True, which="both")

    if kwargs["hist"] == False:
        max_x, max_y = [], []
        for label, trace in kwargs["traces"].items():
            x_data, y_data = trace["x_data"], trace["y_data"]
            max_x.append(max(x_data)), max_y.append(max(y_data))
            plt.scatter(x_data, y_data, label = label)
        plt.legend()
    
    if "hline" in kwargs.keys():
        val, col, style = kwargs["hline"].values()
        plt.axhline(y=val, color=col, linestyle=style)

    
    x_pos = 0.7*max(max_x)
    #y_pos = 0.9*min(max_y)
    plt.text(x_pos,0.8, kwargs["eta_text"])
    #plt.text(0.95, 0.7, kwargs["pt_text"])

    x_label, y_label = kwargs["x_label"], kwargs["y_label"]
    plt.xlabel(x_label), plt.ylabel(y_label)
    plt.title(kwargs['plot_title'])
    plt.savefig(kwargs["plot_path"])


def update_particle_callback():
    from dash import callback, Input, Output, State, callback_context
    @callback(
            Output("particle", "value"),
            Input("particle", "value")
    )
    def update_particle(particle):
        return particle

def update_pileup_callback():
    from dash import callback, Input, Output, State, callback_context
    @callback(
        Output("pileup", "color"),
        Input("pileup", "n_clicks")
    )
    def update_pileup(n_clicks):
        if n_clicks%2==0:
            return "primary"
        else:
            return "success"
    