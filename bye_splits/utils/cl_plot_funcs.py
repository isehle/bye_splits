import os
import sys

parent_dir = os.path.abspath(__file__ + 3 * "/..")
sys.path.insert(0, parent_dir)

import numpy as np

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
                
            max_x, max_y = [], []
            for label, info in traces[key].items():

                if info["plot_type"] == "scatter":
                    max_x.append(max(info["x_data"])), max_y.append(max(info["y_data"]))
                    ax.scatter(info["x_data"], info["y_data"], label = label, c = info["color"])

                    if (("xerr" in info.keys()) and ("yerr" in info.keys())):
                        plt.errorbar(info["x_data"], info["y_data"], yerr=info["yerr"], xerr=info["xerr"], ecolor = info["color"])
                    elif "xerr" in info.keys():
                        plt.errorbar(info["x_data"], info["y_data"], xerr=info["xerr"], ecolor = info["color"])
                    elif "yerr" in info.keys():
                        plt.errorbar(info["x_data"], info["y_data"], yerr=info["yerr"], ecolor = info["color"])

                elif info["plot_type"] == "plot":
                    max_x.append(max(info["x_data"])), max_y.append(max(info["y_data"]))
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
                    max_x.append(max(bins)), max_y.append(max(counts))
                    plt.stairs(counts, bins, label=label, color = info["color"])
                    plt.yscale("log")

                    mean_val, color, linestyle = info["vline"].values()
                    ax.axvline(x=mean_val, color=color, linestyle=linestyle)
                
                ax.set_xlabel(info["x_title"]), ax.set_ylabel(info["y_title"])

            legend = ax.legend()

            if "pt_text" in self.kwargs.keys():
                plt.figtext(0.2, 0.65, self.kwargs["radius_text"])
                plt.figtext(0.2, 0.6, self.kwargs["pt_text"])
                plt.figtext(0.2, 0.55, self.kwargs["eta_text"])
                print(self.kwargs["radius_text"])

            ax.grid(visible=True, which="both")

            ax.set_title(self.plot_title, loc="left")
        
        print(self.plot_path)
        self.fig.savefig(self.plot_path)

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