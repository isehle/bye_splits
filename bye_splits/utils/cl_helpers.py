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


def get_dfs(init_files, coef, weighted=False):
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
    return common.dot_dict(df_dict)


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

def read_weights(dir, cfg, version="final", mode="weights"):
    weights_by_particle = {}
    for particle in ("photons", "electrons", "pions"):
        basename = "optimization_final" if particle!= "pions" else "optimization_photAndBound5"
        #basename += "_bound5" if particle=="pions" else ""
        
        version_dir = "{}/".format(version)
        if particle!="electrons":
            particle_dir = "{}{}/optimization/official/{}".format(dir, particle, version_dir)
        else:
            particle_dir = "{}photons/optimization/official/{}".format(dir, version_dir)
        #particle_dir += "bound5/" if particle=="pions" else ""

        plot_dir = particle_dir+"/plots/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        files = [f for f in os.listdir(particle_dir) if basename in f]
        weights_by_radius = {}
        for file in files:
            radius = float(file.replace(".hdf5","").replace(f"{basename}_","").replace("r","").replace("p","."))
            infile = particle_dir+file
            with pd.HDFStore(infile, "r") as optWeights:
                weights_by_radius[radius] = optWeights[mode]
    
        weights_by_particle[particle] = weights_by_radius
    
    return weights_by_particle

def update_button(n_clicks):
    if n_clicks%2==0:
        return "primary"
    else:
        return "success"
    

'''class ClusterSizeData:
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
        return "coef_{}".format(str(self._radius).replace(".","p"))'''
    