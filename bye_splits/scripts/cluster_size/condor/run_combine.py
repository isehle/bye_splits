# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 4 * "/..")
sys.path.insert(0, parent_dir)

from utils import params, common

from data_handle.data_process import get_data_reco_chain_start

import random
import re

random.seed(10)
import numpy as np
import pandas as pd
import sys

import yaml
from tqdm import tqdm

def split_dfs(cl_df):
    weighted_cols = [col for col in cl_df.keys() if "weighted" in col]
    original_cols = [col.replace("weighted_","") for col in weighted_cols]
    weighted_cols += ["event"]
    original_cols += ["event"]

    original_df = cl_df[original_cols]
    weighted_df = cl_df[weighted_cols].rename(dict(zip(weighted_cols, original_cols)), axis=1)

    return original_df, weighted_df

def normalize_df(cl_df, gen_df, dRThresh):

    combined_df = cl_df.set_index("event").join(
        gen_df.set_index("event"), on="event", how="inner"
    )

    if "dR" not in combined_df.keys():
        combined_df["dR"] = np.sqrt((combined_df["eta"]-combined_df["gen_eta"])**2+(combined_df["phi"]-combined_df["gen_phi"])**2)
    if "matches" not in combined_df.keys():
        combined_df["matches"] = combined_df["dR"] <= dRThresh

    combined_df["pt"] = combined_df["en"] / np.cosh(combined_df["eta"])
    combined_df["gen_pt"] = combined_df["gen_en"] / np.cosh(combined_df["gen_eta"])

    combined_df["pt_norm"] = combined_df["pt"] / combined_df["gen_pt"]
    combined_df["en_norm"] = combined_df["en"] / combined_df["gen_en"]

    return combined_df


def combine_files_by_coef(in_dir, file_pattern):
    files = [
        file for file in os.listdir(in_dir) if re.search(file_pattern, file) != None and "valid" not in file
    ]
    coef_pattern = r"coef_0p(\d+)"
    out_path = common.fill_path(file_pattern, data_dir=in_dir)
    with pd.HDFStore(out_path, "w") as clusterSizeOut:
        print("\nCombining Files:\n")
        for file in tqdm(files):
            key = re.search(coef_pattern, file).group()
            with pd.HDFStore(in_dir + "/" + file, "r") as clSizeCoef:
                clusterSizeOut[key] = clSizeCoef["/data"]


def combine_cluster(cfg):
    nevents = cfg["clusterStudies"]["nevents"]
    particles = cfg["particles"]
    pileup = "PU0" if not cfg["clusterStudies"]["pileup"] else "PU200"

    dir = "{}/{}/{}/cluster/weighted/".format(params.LocalStorage, pileup, particles)
    cl_size_out = common.fill_path(cfg["clusterStudies"]["clusterSizeBaseName"], data_dir=dir)

    combine_files_by_coef(dir, cfg["clusterStudies"]["clusterSizeBaseName"])

    with pd.HDFStore(cl_size_out, mode="a") as clSizeOut:
        df_gen, _, _ = get_data_reco_chain_start(
            particles=particles, nevents=nevents, reprocess=False
        )
        dRthresh = cfg["selection"]["deltarThreshold"]
        coef_keys = clSizeOut.keys()
        print("\nNormalizing Files:\n")
        for coef in tqdm(coef_keys):
            original_df, weighted_df = split_dfs(clSizeOut[coef])
            normed_df, normed_weighted_df = normalize_df(original_df, df_gen, dRthresh), normalize_df(weighted_df, df_gen, dRthresh)
            df_dict = {"original": normed_df,
                       "weighted": normed_weighted_df}
            clSizeOut[coef] = pd.Series(df_dict)

if __name__ == "__main__":
    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)

    for particles in ("photons", "electrons", "pions"):
        cfg.update({"particles": particles})
        combine_cluster(cfg)
