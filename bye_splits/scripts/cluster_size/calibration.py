# coding: utf-8

_all_ = []

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

from utils import common, cl_helpers, params, parsing
from tasks import cluster
from data_handle import data_process

import argparse
import random
import re

import numpy as np
import pandas as pd

import yaml
from tqdm import tqdm

from scipy.optimize import lsq_linear
from sklearn.linear_model import LinearRegression

class Correction:
    def __init__(self, particles, eta_range, pt_cut):
        self.cluster_data = cl_helpers.clusterData(dir="optimization")
        self.particles = particles
        self.radius = (0.01)*(self.particles=="photons") + (0.015)*(self.particles=="electrons") + (0.02)*(self.particles=="pions")
        self.eta_range = eta_range
        self.pt_cut = pt_cut
        self.radii_keys = self.cluster_data.get_keys("PU0", particles)

    def _get_params_and_cfg(self):
        parser = argparse.ArgumentParser(description="parameters")
        parsing.add_parameters(parser)
        FLAGS = parser.parse_args()

        with open(params.CfgPath, "r") as afile:
            cfg = yaml.safe_load(afile)

        return common.dot_dict(vars(FLAGS)), cfg

    def _get_cl_input_files(self, radius):
        pars, cfg = self._get_params_and_cfg()

        cluster_d = params.read_task_params("cluster")

        if isinstance(radius, str):
            radius = float(re.split("coef_",radius)[1].replace("p","."))

        cl_size_coef = "{}_radius_{}".format(
            cfg["clusterStudies"]["clusterSizeBaseName"],
            str(round(radius, 3)).replace(".", "p"),
        )
        cluster_d["ClusterOutPlot"], cluster_d["ClusterOutValidation"] = cl_size_coef, cl_size_coef+"_valid"
        cluster_d["CoeffA"] = [radius] * 50

        for key in ("ClusterInTC", "ClusterInSeeds", "ClusterOutPlot", "ClusterOutValidation"):
            name = cluster_d[key]

            cluster_d[key] =  "{}_PU0_{}_posEta_10oct".format(self.particles, name)

        cluster_d["layerWeights"] = True

        return pars, cluster_d

    def _get_tc_pt_by_layer(self, radius):

        pars, cluster_d = self._get_cl_input_files(radius=radius)
        
        tc_info = cluster.cluster_default(pars, **cluster_d)
        
        return tc_info
    
    def _get_layers(self, radius):
        tc_info = self._get_tc_pt_by_layer(radius)

        layers = pd.concat(tc_info, keys=tc_info.keys(), names=["event"])

        # Select events whose clusters are within the core of the distribution (μ +- σ)
        cl_pt = layers.groupby("event").apply(lambda x: x.tc_pt_sum.sum()).to_frame().rename({0: "cl_pt"}, axis=1)
        low_pt, high_pt = cl_pt.mean() - cl_helpers.effrms(cl_pt), cl_pt.mean() + cl_helpers.effrms(cl_pt)

        cl_pt = cl_pt[ (cl_pt.cl_pt >= low_pt.values[0]) & (cl_pt.cl_pt <= high_pt.values[0]) ]

        layers = layers.loc[cl_pt.index]

        return layers

    def _get_gen_df(self):
        pars, cfg = self._get_params_and_cfg()
        ds_gen, _, _ = data_process.get_data_reco_chain_start(nevents=pars["nevents"],
                                                              reprocess="False",
                                                              tag=cfg["clusterStudies"]["parquetTag"],
                                                              particles=self.particles)

        if "gen_pt" not in ds_gen.keys():
            ds_gen["gen_pt"] = ds_gen.gen_en/np.cosh(ds_gen.gen_eta)

        ds_gen.set_index("event", inplace=True)

        gen_pt_cut = cfg["clusterStudies"]["optimization"]["pt_cut"]

        eta_min = cfg["clusterStudies"]["optimization"]["eta_min"]
        eta_max = cfg["clusterStudies"]["optimization"]["eta_max"]
        
        ds_gen = ds_gen[ (ds_gen.gen_pt >= gen_pt_cut) & (ds_gen.gen_eta >= eta_min) & (ds_gen.gen_eta <= eta_max) ]

        return ds_gen

    def _select_events(self, radius, ds_gen):
        _, cfg = self._get_params_and_cfg()
        cl_pt_cut = cfg["clusterStudies"]["optimization"]["cl_pt_cut"]
        
        layers = self._get_layers(radius)

        sub_layers = layers.groupby("event").apply(lambda x: x.tc_pt_sum.sum()).to_frame().rename({0: "cl_pt"}, axis=1)
        sub_layers = sub_layers[ sub_layers.cl_pt >= cl_pt_cut ]

        layer_events = sub_layers.index.unique()
        gen_events = ds_gen.index

        events = gen_events.intersection(layer_events)

        ds_gen, layers = ds_gen.loc[events], layers.loc[events]

        return ds_gen, layers

    def get_layer_weights(self, radius, ds_gen):
        ds_gen, layers = self._select_events(radius, ds_gen)
        
        ds_gen.sort_index(level="event", inplace=True)
        layers.sort_index(level="event", inplace=True)
        
        layers = layers.unstack(level="tc_layer").fillna(0.0)
        gen_pt = ds_gen.gen_pt

        max_weight = 2.0 if self.particles != "pions" else 5.0

        regression = lsq_linear(layers,
                                #ds_gen,
                                gen_pt,
                                bounds=(0.0, max_weight),
                                method="bvls",
                                lsmr_tol="auto",
                                verbose=1
                                )

        weights = regression.x

        index = layers.keys().get_level_values("tc_layer").to_list()
        weights = pd.DataFrame(data = weights,
                               index=index,
                               columns=["weights"])

        # First layer not used for calibration, manually set a "weight" of 1.
        weights.loc[1.0] = 1.0
        weights.sort_index(inplace=True)

        return weights

    def _weight_file_path(self, all_radii=True):
        pars, cfg = self._get_params_and_cfg()
        
        local = cfg["clusterStudies"]["local"]

        base_path = params.LocalStorage if local else params.EOSStorage("iehle", "data")
        basename = cfg["clusterStudies"]["optimization"]["baseName"]

        if not all_radii: basename += "_r_" + str(self.radius).replace(".", "p")

        out_path = "{}PU0/{}/weights/{}.hdf5".format(base_path, self.particles, basename)

        return out_path

    def save_layer_weights(self, all_radii=True):
        ds_gen = self._get_gen_df()

        out_path = self._weight_file_path(all_radii=all_radii)

        with pd.HDFStore(out_path, "w") as weightFile:
            print("\nSaving weights to {}\n".format(out_path))
            if all_radii:
                for r in tqdm(self.radii_keys[1:], total=len(self.radii_keys[1:])):
                    weightFile[r] = self.get_layer_weights(r, ds_gen)
            else:
                weightFile["weights"] = self.get_layer_weights(self.radius, ds_gen)

    def read_weights(self, radius=None, all_radii_file=True, type="layer"):
        path = self._weight_file_path(all_radii=all_radii_file)

        with pd.HDFStore(path, "r") as weightFile:
            if all_radii_file:
                return weightFile[self.radius] if radius==None else weightFile[radius]
            else:
                return weightFile["weights"]

    def apply_layer_weights(self, radius=None, all_radii_file=True):
        r = self.radius if radius == None else radius

        pars, cluster_d = self._get_cl_input_files(radius=r)
        weights = self.read_weights(radius=radius, all_radii_file=all_radii_file)

        for key in ("ClusterOutPlot", "ClusterOutValidation"):
            name = cluster_d[key]

            cluster_d[key] =  "{}_NOTweighted".format(name)

        cluster_d["layerWeights"] = False
        cluster_d["applyWeights"] = weights

        _ = cluster.cluster_default(pars, **cluster_d)


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
                                                            particles=self.particles,
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
                                                                    particles=self.particles,
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

corr = Correction(particles="photons",
                  eta_range=[1.7, 2.7],
                  pt_cut=10)

#corr.save_layer_weights(all_radii=False)

#weight_df = corr.read_weights(all_radii_file=False)

corr.apply_layer_weights(all_radii_file=False)