# pyright: reportUnboundVariable=false
# pyright: reportUnusedExpression=false

import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

import numpy as np
from scipy.stats import crystalball
from scipy.special import erf
import scipy.special as sp

import matplotlib.pyplot as plt
import mplhep as hep

from dash import callback, Input, Output, State, callback_context

from bye_splits.utils import cl_helpers

tup = lambda x: tuple([int(i) for i in x])

eta_text = lambda eta_range: r"${} < \eta < {}$".format(eta_range[0], eta_range[1])
pt_text = lambda pt_cut: r"${p_T}^{Gen} > $" + r"${}$".format(pt_cut)
radius_text = lambda r: r"$r = {}$".format(r)

title = lambda tit, pu: tit + r"$({})$".format(pu)
path = lambda base, pu, eta, pt, particle: base.format(pu, cl_helpers.annot_str(eta[0]), cl_helpers.annot_str(eta[1]), pt, particle)


class cmsPlot:
    def __init__(self, plot_title, plot_path, **kwargs):
        self.kwargs = kwargs
        self.plot_title = plot_title
        self.plot_path = plot_path
        hep.style.use("CMS")
        plt.rcParams['text.usetex'] = True
        self.fig = plt.figure()

    def write_fig(self):
        traces = self.kwargs["traces"]

        if "yrange" in self.kwargs.keys():
            y_min, y_max = self.kwargs["yrange"]
            plt.ylim(y_min, y_max)
        if "xrange" in self.kwargs.keys():
            x_min, x_max = self.kwargs["xrange"]
            plt.xlim(x_min, x_max)

        if "hline" in self.kwargs.keys():
            val, col, style = self.kwargs["hline"].values()
        if "vline" in self.kwargs.keys():
            val, col, style = self.kwargs["vline"].values()
        
        for key in traces.keys():

            n_rows, n_cols, fig_num = tup(key)
            ax = self.fig.add_subplot(n_rows, n_cols, fig_num)

            if "hline" in self.kwargs.keys():
                ax.axhline(y=val, color=col, linestyle=style)
            if "vline" in self.kwargs.keys():
                ax.axvline(x=val, color=col, linestyle=style)
                
            for label, info in traces[key].items():
                if info["plot_type"] == "scatter":
                    ax.scatter(info["x_data"], info["y_data"], label = label, c = info["color"])

                    if (("xerr" in info.keys()) and ("yerr" in info.keys())):
                        plt.errorbar(info["x_data"], info["y_data"], yerr=info["yerr"], xerr=info["xerr"], ecolor = info["color"])
                    elif "xerr" in info.keys():
                        plt.errorbar(info["x_data"], info["y_data"], xerr=info["xerr"], ecolor = info["color"])
                    elif "yerr" in info.keys():
                        plt.errorbar(info["x_data"], info["y_data"], yerr=info["yerr"], ecolor = info["color"])

                elif info["plot_type"] == "plot":
                    ax.plot(info["x_data"], info["y_data"], label = label, color = info["color"], linestyle=info["linestyle"])
                    
                    if (("xerr" in info.keys()) and ("yerr" in info.keys())):
                        plt.errorbar(info["x_data"], info["y_data"], yerr=info["yerr"], xerr=info["xerr"], ecolor = info["color"])
                    elif "xerr" in info.keys():
                        plt.errorbar(info["x_data"], info["y_data"], xerr=info["xerr"], ecolor = info["color"])
                    elif "yerr" in info.keys():
                        plt.errorbar(info["x_data"], info["y_data"], yerr=info["yerr"], ecolor = info["color"])
                    
                    if "yscale" in info.keys():
                        plt.yscale(info["yscale"])

                elif info["plot_type"] == "hist":
                    counts, bins = np.histogram(info["data"], bins=info["bins"])
                    plt.stairs(counts, bins, label=label, color = info["color"])
                    plt.yscale("log")

                    mean_val, color, linestyle = info["vline"].values()
                    ax.axvline(x=mean_val, color=color, linestyle=linestyle)
                
                elif info["plot_type"] == "violin":
                    ax.violin(info["data"], positions=info["layers"], showmeans=True, showmedians=True)
                
                ax.set_xlabel(info["x_title"]), ax.set_ylabel(info["y_title"])

            legend = ax.legend()

            if "pt_text" in self.kwargs.keys():
                plt.figtext(0.2, 0.65, self.kwargs["radius_text"])
                plt.figtext(0.2, 0.6, self.kwargs["pt_text"])
                plt.figtext(0.2, 0.55, self.kwargs["eta_text"])

            ax.grid(visible=True, which="both")

            ax.set_title(self.plot_title, loc="left")
        
        print("\nSaving ", self.plot_path)
        self.fig.savefig(self.plot_path)

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