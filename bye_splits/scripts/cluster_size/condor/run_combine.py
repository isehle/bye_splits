# coding: utf-8

_all_ = []

import os
import sys
import argparse

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
    weighted_cols += [col for col in cl_df.keys() if "layer" in col]
    original_cols = [col.replace("weighted_","") for col in weighted_cols]
    weighted_cols += ["event"]
    original_cols += ["event"]

    original_df = cl_df[original_cols]
    weighted_df = cl_df[weighted_cols].rename(dict(zip(weighted_cols, original_cols)), axis=1)

    return original_df, weighted_df

def normalize_df(cl_df, gen_df, dRThresh):
    cl_df=cl_df.reset_index().set_index(["event","seed_idx"])
    combined_df = cl_df.join(
        gen_df.set_index("event"), on="event", how="inner"
    )

    if "dR" not in combined_df.keys():
        combined_df["dR"] = np.sqrt((abs(combined_df["eta"])-abs(combined_df["gen_eta"]))**2+(combined_df["phi"]-combined_df["gen_phi"])**2)
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
    breakpoint()
    with pd.HDFStore(out_path, "w") as clusterSizeOut:
        print("\nCombining Files:\n")
        for file in tqdm(files):
            key = re.search(coef_pattern, file).group()
            with pd.HDFStore(in_dir + "/" + file, "r") as clSizeCoef:
                clusterSizeOut[key] = clSizeCoef["/data"]

def split_and_norm(df_cl, df_gen, dRthresh):
    original_df, weighted_df = split_dfs(df_cl)
    normed_df, normed_weighted_df = normalize_df(original_df, df_gen, dRthresh), normalize_df(weighted_df, df_gen, dRthresh)
    df_dict = {"original": normed_df,
                "weighted": normed_weighted_df}
    df_cl = pd.Series(df_dict)
    return df_cl

def combine_cluster(cfg, **pars):
    """Originally designed to combine the files returned by cluster for each radii,
    and to normalize each by the gen_particle information. Now accepts an optional --file
    parameter to normalize this, skipping the combinination step."""

    input_file_path = pars["file"] if "file" in pars.keys() else None
    unweighted = pars["unweighted"] if "unweighted" in pars.keys() else False

    particles = cfg["particles"]
    nevents = cfg["clusterStudies"]["nevents"]

    if input_file_path == None:
        pileup = "PU0" if not cfg["clusterStudies"]["pileup"] else "PU200"

        basename = cfg["clusterStudies"]["combination"][pileup][particles]["basename"]
        sub_dir = cfg["clusterStudies"]["combination"][pileup][particles]["sub_dir"]

        dir = "{}/{}/{}".format(params.LocalStorage, pileup, sub_dir)

        combine_files_by_coef(dir, basename)

        cl_size_out = common.fill_path(basename, data_dir=dir)

    else:
        cl_size_out = input_file_path

    with pd.HDFStore(cl_size_out, mode="a") as clSizeOut:
        df_gen, _, _ = get_data_reco_chain_start(
            particles=particles, nevents=nevents, reprocess=False, tag = cfg["clusterStudies"]["parquetTag"]
        )
        #if "negEta" in sub_dir:
        if "negEta" in cl_size_out:
            df_gen = df_gen[ df_gen.gen_eta < 0 ]
            df_gen["gen_eta"] = abs(df_gen.gen_eta)
        else:
            df_gen = df_gen[ df_gen.gen_eta > 0 ]
        dRthresh = cfg["selection"]["deltarThreshold"]
        if input_file_path != None:
            clSizeOut["data"] = split_and_norm(clSizeOut["data"], df_gen, dRthresh) if not unweighted else normalize_df(clSizeOut["data"], df_gen, dRthresh)
        else:
            coef_keys = clSizeOut.keys()
            print("\nNormalizing Files:\n")
            for coef in tqdm(coef_keys):
                clSizeOut[coef] = split_and_norm(clSizeOut[coef], df_gen, dRthresh) if not unweighted else normalize_df(clSizeOut[coef], df_gen, dRthresh)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--file", type=str)
    parser.add_argument("--unweighted", action="store_true")

    FLAGS = parser.parse_args()
    pars = common.dot_dict(vars(FLAGS))

    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)

    cfg.update({"particles": "pions"})
    combine_cluster(cfg, **pars)

    '''for particles in ("photons", "electrons", "pions"):
    #for particles in ("photons", "electrons"):
        cfg.update({"particles": particles})
        combine_cluster(cfg)'''