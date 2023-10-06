# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

from utils import common, cl_helpers

import argparse
import random

import numpy as np
import pandas as pd

import yaml

from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression

class Correction:
    def __init__(self, particle, eta_range, pt_cut):
        self.cluster_data = cl_helpers.clusterData(dir="optimization")
        self.particle = particle
        self.eta_range = eta_range
        self.pt_cut = pt_cut
        self.radii_keys = self.cluster_data.get_keys("PU0", particle)

    def _lin_reg_add(self, df, corr_col, pt_col="pt", mode="diff"):
        X = df[[corr_col]]
        if pt_col == "pt":
            y = df.pt - df.gen_pt if mode=="diff" else df.pt/df.gen_pt
        else:
            y = df[pt_col] - df["gen_pt"] if mode=="diff" else df[pt_col]/df["gen_pt"]
        corr = LinearRegression().fit(X, y)
        return corr.intercept_, corr.coef_[0]

    def _apply_add_corr(self, df, corr, corr_col, pt_col, mode="diff"):
        if mode == "diff":
            cl_pt = np.asarray(df[pt_col])
        else:
            cl_pt = np.asarray(df.pt) if pt_col == "pt_norm" else np.asarray(df.pt_en_corrected)

        cl_variable = np.asarray(df[corr_col])

        intercept = np.full(cl_variable.shape, corr["intercept"])
        slope = np.full(cl_variable.shape, corr["slope"])

        corrected_pt = cl_pt*(intercept + cl_variable*slope)**(-1)

        return corrected_pt

    def _add_corr_to_df(self, df, corr_col, pt_col, corr=None, mode="diff"):
        df.sort_values(corr_col, inplace=True)

        df["pt_diff"] = df.pt - df.gen_pt

        if corr==None:
            o_corr = None

            intercept, slope = self._lin_reg_add(df=df,
                                                 corr_col=corr_col,
                                                 pt_col=pt_col.replace("_norm", ""),
                                                 mode=mode)

            corr = {"slope": slope,
                    "intercept": intercept}

        else:
            o_corr = corr

        if pt_col == "pt" or pt_col == "pt_norm":
            df["pt_en_corrected"] = self._apply_add_corr(df=df,
                                                         corr=corr,
                                                         corr_col="en",
                                                         pt_col="pt" if mode == "diff" else "pt_norm",
                                                         #pt_col = pt_col,
                                                         mode=mode)

            df["pt_diff_en_corrected"] = df.pt_en_corrected - df.gen_pt
            df["pt_norm_en_corrected"] = df.pt_en_corrected/df.gen_pt

        elif pt_col == "pt_en_corrected" or pt_col == "pt_norm_en_corrected":
            df["pt_eta_corrected"] = self._apply_add_corr(df=df,
                                                          corr=corr,
                                                          corr_col="eta",
                                                          #pt_col="pt_en_corrected" if mode == "diff" else "pt_norm_en_corrected",
                                                          #pt_col = pt_col,
                                                          pt_col = "pt_en_corrected",
                                                          mode=mode)

            df["pt_diff_eta_corrected"] = df.pt_eta_corrected - df.gen_pt
            df["pt_norm_eta_corrected"] = df.pt_eta_corrected/df.gen_pt
        
        if o_corr == None:
            return df, corr
        else:
            return df

    def apply_corrections(self, zero_pu_path, pu_path, mode="diff"):
        df_dict, df_dict_pu = {}, {}
        en_dict, eta_dict = {}, {}
        with pd.HDFStore(zero_pu_path, "w") as zeroPuFile, pd.HDFStore(pu_path, "w") as puFile:
            for radius in self.radii_keys:
                # Calculate energy correction from the layer weighted PU0 distribution
                # and apply it to the pt column
                df_o, df_w = self.cluster_data.get_dataframes(pileup="PU0",
                                                            particles=self.particle,
                                                            coef=radius,
                                                            eta_range=self.eta_range,
                                                            pt_cut=self.pt_cut)

                df_w, en_fit_info = self._add_corr_to_df(df=df_w,
                                                        corr_col="en",
                                                        pt_col="pt",
                                                        mode=mode)
                
                df_dict[radius] = pd.Series({"original": df_o,
                                             "weighted": df_w})

                en_dict[radius] = pd.Series(en_fit_info)

                # Apply energy correction to layer weighted PU200 pt column
                df_pu_o, df_pu_w = self.cluster_data.get_dataframes(pileup="PU200",
                                                                    particles=self.particle,
                                                                    coef=radius,
                                                                    eta_range=self.eta_range,
                                                                    pt_cut=self.pt_cut)
                df_pu_w = self._add_corr_to_df(df=df_pu_w,
                                            corr_col="en",
                                            pt_col="pt" if mode=="diff" else "pt_norm",
                                            corr=en_fit_info,
                                            mode=mode)

                # Calculate and apply eta correction from the pt_en_corrected column
                df_pu_w, eta_fit_info = self._add_corr_to_df(df=df_pu_w,
                                                            corr_col="eta",
                                                            pt_col="pt_en_corrected" if mode=="diff" else "pt_norm_en_corrected",
                                                            mode=mode)
                
                df_dict_pu[radius] = pd.Series({"original": df_pu_o,
                                                "weighted": df_pu_w})

                eta_dict[radius] = pd.Series(eta_fit_info)

                zeroPuFile[radius] = df_dict[radius]
                puFile[radius] = df_dict_pu[radius]
        
        return pd.DataFrame(df_dict), pd.DataFrame(df_dict_pu), pd.DataFrame(en_dict), pd.DataFrame(eta_dict)

corr = Correction(particle="pions", eta_range=[1.7, 2.7], pt_cut=10)

zero_pu_path = "data/new_algos/PU0/pions/cluster/weighted/final/bound5/selectOneStd/maxSeed/fullEta/posEta/bc_stc/cluster_size_fullEta_weighted_selectOneEffRms_maxSeed_smooth_bc_stc_ThresholdDummyHistomaxnoareath20_filtered_fullEta_enCalibrated_byNorm.hdf5"
pu_path = "data/new_algos/PU200/pions/cluster/weighted/selectOneStd/maxSeed/smooth/bc_stc/posEta/cluster_size_fullEta_weighted_selectOneEffRms_maxSeed_ThresholdDummyHistomaxnoareath20_filtered_en_eta_calibrated_byNorm.hdf5"

dfs, dfs_pu, en_fits, eta_fits = corr.apply_corrections(zero_pu_path=zero_pu_path,
                                                        pu_path=pu_path,
                                                        mode="norm")