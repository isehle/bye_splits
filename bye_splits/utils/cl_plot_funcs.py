# pyright: reportUnboundVariable=false
# pyright: reportUnusedExpression=false

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd

from scipy.stats import crystalball
from scipy.special import erf
import scipy.special as sp

import matplotlib.pyplot as plt
import mplhep as hep

from dash import callback, Input, Output, State, callback_context
import plotly.graph_objects as go

from bye_splits.utils import cl_helpers

tup = lambda x: tuple([int(i) for i in x])

eta_text = lambda eta_range: r"${} < \eta < {}$".format(eta_range[0], eta_range[1])
pt_text = lambda pt_cut: r"${p_T}^{Gen} > $" + r"${}$".format(pt_cut)
radius_text = lambda r: r"$r = {}$".format(r)

title = lambda tit, pu: tit + r"$({})$".format(pu)
path = lambda base, pu, eta, pt, particle: base.format(pu, cl_helpers.annot_str(eta[0]), cl_helpers.annot_str(eta[1]), pt, particle)


class dashPlot:
    def __init__(self, particle, pileup, **kwargs):
        self.particle     = particle
        self.pileup       = pileup
        self.kwargs       = kwargs
        self.cluster_data = cl_helpers.clusterData()
        self.plot_data    = dict.fromkeys(("plot_type", "hline", "vline", "traces"), None)

    def _set_props(self):

        labels = ["Original", "Layer Weighted"]
        colors = ["blue", "red"]
        cols   = ["pt_norm", "pt_norm"]

        if self.particle == "pions":
            labels.append("Energy Corrected")
            colors.append("purple")
            cols.append("pt_norm_en_corrected")
        
        if self.pileup == "PU200":
            labels.append("Eta Corrected")
            colors.append("green")
            cols.append("pt_norm_eta_corr") if self.particle != "pions" else cols.append("pt_norm_eta_corrected")

        return labels, colors, cols

    def plot_hist(self, radius, df_o, df_w, fig, rowcol=(1,1), version="particle", calib = None):
        labels, colors, cols = self._set_props()

        self.plot_data["plot_type"] = "hist"

        if version == "particle":
            for label, color, column in zip(labels, colors, cols):
                df = df_o if label == "Original" else df_w

                mean_value = df[column].mean() if self.particle != "electrons" else self.cluster_data._get_gaus_mean(df, column)

                fig.add_trace(
                    go.Histogram(
                        x = df[column],
                        nbinsx=1000 if self.particle != "pions" else 100,
                        autobinx=False,
                        name=label,
                        histnorm="probability density",
                        marker_color=color
                    ),
                    row=rowcol[0],
                    col=rowcol[1]
                )
                
                fig.add_vline(x=mean_value, row=rowcol[0], col=rowcol[1], line_dash="dash", line_color=color)
        else:
            column_matching = {"original": "pt_norm",
                               "layer"   : "pt_norm",
                               "eta"     : "pt_norm_eta_corr"}
            
            
            df_type = df_o if calib == "original" else df_w

            for particle,color in zip(df_o.keys(), ("blue", "red", "green")):
                if particle == "pions":
                    column_matching["eta"] = "pt_norm_eta_corrected"
                
                column = column_matching[calib]
                df = df_type[particle]

                mean_value = df[column].mean() if particle != "electrons" else self.cluster_data._get_gaus_mean(df, column)

                fig.add_trace(
                    go.Histogram(
                        x = df[column],
                        nbinsx=1000 if self.particle != "pions" else 100,
                        autobinx=False,
                        name=particle.capitalize(),
                        histnorm="probability density",
                        marker_color=color
                    ),
                    row=rowcol[0],
                    col=rowcol[1]
                )

                fig.add_vline(x=mean_value, row=rowcol[0], col=rowcol[1], line_dash="dash", line_color=color)

        return fig

    def plot_res(self, radius, df_o, df_w, x_axis, fig, rowcol, version):
        labels, colors, cols = self._set_props()

        self.plot_data["plot_type"] = "scatter"

        for label, color, column in zip(labels, colors, cols):
            df = df_o if label == "Original" else df_w

            df[x_axis + "_bin"] = pd.cut(df[x_axis], bins=20, labels=False)

            x_data = df.groupby(x_axis + "_bin").apply(lambda x: x[x_axis].mean())
            res    = df.groupby(x_axis + "_bin").apply(lambda x: x[column].std()/x[column].mean())
            eff    = df.groupby(x_axis + "_bin").apply(lambda x: cl_helpers.effrms(x[column])/x[column].mean())

            for i, y_data in enumerate([res, eff]):
                fig.add_trace(
                    go.Scatter(
                        x = x_data,
                        y = y_data,
                        name = label + " RMS/Mean" if i==0 else label + " Eff_RMS/Mean",
                        line = dict(color=color) if i==0 else dict(dash="dash", color=color),
                        mode="lines"
                    ),
                    row=rowcol[0],
                    col=rowcol[1]
                )
        
        return fig

    def plot_global_res(self, glob_res, fig, rowcol):
        labels, colors, cols = self._set_props()

        self.plot_data["plot_type"] = "plot"

        label_matching = {"Original"        : "original",
                          "Layer Weighted"  : "layer",
                          "Energy Corrected": "energy",
                          "Eta Corrected"   : "eta"}

        for label, color, column in zip(labels, colors, cols):
            res_key = label_matching[label]

            for i, res in enumerate(glob_res[res_key].values()):
                fig.add_trace(
                    go.Scatter(
                        x = np.arange(0.001, 0.05, 0.001),
                        y = res,
                        name = label + " RMS" if i==0 else label + " Eff_RMS",
                        line = dict(color=color) if i==0 else dict(dash="dash", color=color)
                    ),
                    row=rowcol[0],
                    col=rowcol[1]
                )

        return fig

    def plot_global_pt(self, glob_pt, fig, version="particle", calib=None):
        fig.add_hline(y = 1.0, line_dash="dash", line_color="black")

        #self.plot_data["plot_type"] = "plot"
        self.plot_data["hline"]     = {"val": 1.0, "color": "black", "linestyle": "--"}
        self.plot_data["x_title"]   = r"$r^{Cl}$"
        self.plot_data["y_title"]   = r"$\langle p_T^{Cl}/{p_T^{Gen}} \rangle$"
        self.plot_data["traces"]    = {}

        if version=="particle":
            labels, colors, cols = self._set_props()

            label_matching = {"Original"        : "original",
                            "Layer Weighted"  : "weighted",
                            "Energy Corrected": "energy",
                            "Eta Corrected"   : "eta"}

            for label, color, column in zip(labels, colors, cols):
                pt_key = label_matching[label]
            
                trace_data = {"plot_type": "plot",
                            "color"    : color,
                            "x_data"   : np.arange(0.001, 0.05, 0.001),
                            "y_data"   : glob_pt[pt_key],
                            "linestyle": "-",
                            }

                self.plot_data["traces"].update({label: trace_data})

                fig.add_trace(
                    go.Scatter(
                        x = np.arange(0.001, 0.05, 0.001),
                        y = glob_pt[pt_key],
                        name = label,
                    )
                )
        else:
            label_matching = {"original": "original",
                              "layer"   : "weighted",
                              "eta"     : "eta"}

            for particle, data in glob_pt.items():
                key = label_matching[calib]
                
                if particle == "photons":
                    color = "blue"
                else:
                    color = "red" if particle == "electrons" else "green"

                trace_data = {"plot_type": "plot",
                              "color"    : color,
                              "x_data"   : np.arange(0.001, 0.05, 0.001),
                              "y_data"   : data[key],
                              "linestyle": "-"} 

                self.plot_data["traces"].update({particle: trace_data})               

                fig.add_trace(
                    go.Scatter(
                        x = np.arange(0.001, 0.05, 0.001),
                        y = data[key],
                        name = particle.capitalize(),
                    )
                )

        return fig
            
    def download_plot(self, plot_title, plot_path):
        cmsPlot(plot_title, plot_path, **self.plot_data).write_fig()

class cmsPlot:
    def __init__(self, plot_title, plot_path, **kwargs):
        self.kwargs = kwargs
        self.plot_title = plot_title
        self.plot_path = plot_path
        hep.style.use("CMS")
        plt.rcParams['text.usetex'] = True

    def _plot_hist(self, info, label):
        counts, bins = np.histogram(info["data"], bins=info["bins"])
        plt.stairs(counts, bins, label=label, color = info["color"])

        mean_val, color, linestyle = info["vline"].values()
        ax.axvline(x=mean_val, color=color, linestyle=linestyle)

    def write_fig(self):
        fig, ax = plt.subplots()
        traces = self.kwargs["traces"]

        if "yrange" in self.kwargs.keys():
            y_min, y_max = self.kwargs["yrange"]
            plt.ylim(y_min, y_max)
        if "xrange" in self.kwargs.keys():
            x_min, x_max = self.kwargs["xrange"]
            plt.xlim(x_min, x_max)

        if self.kwargs["hline"] is not None:
            val, col, style = self.kwargs["hline"].values()
            ax.axline((0, val), slope=0, color=col, linestyle=style)
        if self.kwargs["vline"] is not None:
            val, col, style = self.kwargs["vline"].values()

        particle_legend = {"photons"  : r"$\gamma$",
                            "electrons": r"$e$",
                            "pions"    : r"$\pi$"}
        
        for label, info in traces.items():

            if label in particle_legend.keys(): label = particle_legend[label]

            if info["plot_type"] == "scatter":
                ax.scatter(info["x_data"], info["y_data"], label = label, c = info["color"])

            elif info["plot_type"] == "plot":
                ax.plot(info["x_data"], info["y_data"], label = label, color = info["color"], linestyle=info["linestyle"])

            elif info["plot_type"] == "hist":
                self._plot_hist(info, label)
            
            if "yscale" in info.keys():
                plt.yscale(info["yscale"])
        
        ax.set_xlabel(self.kwargs["x_title"]), ax.set_ylabel(self.kwargs["y_title"])

        legend = ax.legend()

        ax.grid(visible=True, which="both")

        ax.set_title(self.plot_title, loc="left")
        
        print("\nWriting: ", self.plot_path)
        fig.savefig(self.plot_path)

def gaus(x, mean, std):
    return pow(std*np.sqrt(2*np.pi), -1)*np.exp(-pow((x-mean)/std,2)/2)

def pLaw(x, mean, std, n, A, B):
    return A*pow((B-(x-mean)/std),-n)

def crysCoefs(alpha, n):
    absalpha = np.abs(alpha)
    pi = np.pi
    sq_pi_2 = np.sqrt(pi/2)

    A = pow(n/absalpha,n)*np.exp(-pow(alpha,2)/2)
    B = n/absalpha - absalpha
    C = n/absalpha*pow(n-1,-1)*np.exp(-pow(alpha,2)/2)
    D = sq_pi_2*(1+erf(absalpha/np.sqrt(2)))

    return (A, B, C, D)

def crystal_ball(data, alpha, n, mean, std):
    A, B, C, D = crysCoefs(alpha, n)

    N = pow(std*(C+D),-1)

    split = (data-mean)/std
    cond = split > -alpha

    peak = N*gaus(data, mean, std)
    tail = N*pLaw(data, mean, std, n, A, B)

    return np.where(cond, peak, tail)

def update_particle_callback():
    @callback(
            Output("particle", "value"),
            Input("particle", "value")
    )
    def update_particle(particle):
        return particle

def update_pileup_callback():
    @callback(
        Output("pileup", "color"),
        Input("pileup", "n_clicks")
    )
    def update_pileup(n_clicks):
        if n_clicks%2==0:
            return "primary"
        else:
            return "success"