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

def normalize_df(cl_df, gen_df):
    cl_df["pt"] = cl_df["en"] / np.cosh(cl_df["eta"])
    gen_df["gen_pt"] = gen_df["gen_en"] / np.cosh(gen_df["gen_eta"])

    cl_df = cl_df.set_index("event").join(
        gen_df.set_index("event"), on="event", how="inner"
    )

    cl_df["pt_norm"] = cl_df["pt"] / cl_df["gen_pt"]
    cl_df["en_norm"] = cl_df["en"] / cl_df["gen_en"]

    return cl_df


def combine_files_by_coef(in_dir, file_pattern):
    files = [
        file for file in os.listdir(in_dir) if re.search(file_pattern, file) != None
    ]
    coef_pattern = r"coef_0p(\d+)"
    out_path = common.fill_path(file_pattern)
    with pd.HDFStore(out_path, "w") as clusterSizeOut:
        print("\nCombining Files:\n")
        for file in tqdm(files):
            key = re.search(coef_pattern, file).group()
            with pd.HDFStore(in_dir + "/" + file, "r") as clSizeCoef:
                clusterSizeOut[key] = clSizeCoef["/data"]


def combine_cluster(cfg):
    nevents = cfg["clusterStudies"]["nevents"]

    cl_basename = cfg["clusterStudies"]["clusterSizeBaseName"]
    
    combine_files_by_coef(params.LocalStorage, cl_basename)

    cl_size_out = common.fill_path(cl_basename)
    with pd.HDFStore(cl_size_out, mode="a") as clSizeOut:
        df_gen, _, _ = get_data_reco_chain_start(
            nevents=nevents, reprocess=False
        )
        coef_keys = clSizeOut.keys()
        print("\nNormalizing Files:\n")
        for coef in tqdm(coef_keys[1:]):
            clSizeOut[coef] = normalize_df(clSizeOut[coef], df_gen)

if __name__ == "__main__":
    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)

    combine_cluster(cfg)
