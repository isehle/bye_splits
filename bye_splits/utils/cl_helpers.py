import os
import sys
import re
import argparse

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import yaml

from bye_splits.utils import common, params

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


def get_dfs(init_files, coef):
    """Takes a dictionary of input files (keys corresponding to particles, values corresponding to file paths containing DataFrames by coefficient), with a desired coefficient.
    Returns a new dictionary with the same keys, whose values correspond to the DataFrame of that particular coefficient.
    """
    df_dict = dict.fromkeys(init_files.keys(), [0.0])

    for key in init_files.keys():
        if len(init_files[key]) == 0:
            continue
        elif len(init_files[key]) == 1:
            file = pd.HDFStore(init_files[key][0], "r")

            if not isinstance(coef, str):
                coef = get_str(coef, file)

            df = file[coef]
            file.close()
        else:
            file_list = [pd.HDFStore(val, "r") for val in init_files[key]]
            if not isinstance(coef, str):
                coef = get_str(coef, file_list[0])
            df_list = [file_list[i][coef] for i in range(len(file_list))]
            df = pd.concat(df_list)
            for file in file_list: file.close()
        df_dict[key] = df

    return common.dot_dict(df_dict)


def get_keys(init_files):
    """Returns the list of exact coefficient keys in the initial files; they're the same for all files, so we only need to check one."""
    file_path = init_files["photons"][0]

    with pd.HDFStore(file_path, "r") as file:
        keys = file.keys()

    return keys


def read_cl_size_params():
    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)
    return cfg["clusterStudies"]


def filter_dfs(dfs_by_particle, eta_range, pt_cut):
    filtered_dfs = {}
    for particle, df in dfs_by_particle.items():
        if not isinstance(df, list) and not df.empty:
            with common.SupressSettingWithCopyWarning():
                filtered_dfs[particle] = df[ (df.gen_eta > eta_range[0]) & (df.gen_eta < eta_range[1]) & (df.pt > pt_cut) ]
                if "deltaRsq" not in df.keys():
                    filtered_dfs[particle]["deltaRsq"] = (filtered_dfs[particle].loc[:,"gen_phi"]-filtered_dfs[particle].loc[:,"phi"])**2 + (filtered_dfs[particle].loc[:,"gen_eta"]-filtered_dfs[particle].loc[:,"eta"])**2
                    filtered_dfs[particle]["matches"] = filtered_dfs[particle].loc[:,"deltaRsq"] <= 0.05**2
    return filtered_dfs
    

class ClusterSizeData:
    def __init__(self):
        self._dir = None
        self._particles = None
        self._radius = None
        self._pileup = None

    def __str__(self):
        return self.__repr__()
    
    def __repr__(self):
        s = ('ClusterSizeData instance:\n' +
             'dir = {}\n'.format(self._dir) +
             'particles = {}\n'.format(self._particles) +
             'radius = {}\n'.format(self._radius) +
             'pileup = {}\n'.format(self._pileup))
        return s
    
    @property
    def dir(self):
        return self._dir
    
    @dir.setter
    def dir(self, dir):
        self._dir = dir
             
    @property
    def particles(self):
        return self._particles
    
    @particles.setter
    def particles(self, particles):
        self._particles = particles

    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, radius):
        self._radius = radius

    @property
    def pileup(self):
        return self._pileup

    @pileup.setter
    def pileup(self, pileup):
        self._pileup = pileup

    @property
    def radius_str(self):
        return "coef_{}".format(str(self._radius).replace(".","p"))
    