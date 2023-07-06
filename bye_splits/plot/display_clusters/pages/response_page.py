import os
import sys

parent_dir = os.path.abspath(__file__ + 5 * "/..")
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import scipy.stats as st

from dash import dcc, html, Input, Output, callback
import dash

import dash_bootstrap_components as dbc

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import argparse
from bye_splits.utils import params, parsing, cl_helpers, cl_plot_funcs

parser = argparse.ArgumentParser(description="")
parsing.add_parameters(parser)
FLAGS = parser.parse_args()

cfg = cl_helpers.read_cl_size_params()

if cfg["local"]:
    data_dir = cfg["localDir"]
else:
    data_dir = params.EOSStorage(FLAGS.user, cfg["dataFolder"])

input_files = cfg["dashApp"]

marks = {
    coef: {"label": format(coef, ".3f"), "style": {"transform": "rotate(-90deg)"}}
    for coef in np.arange(0.0, 0.05, 0.001)
}

dash.register_page(__name__, title="Response", name="Response")

layout = html.Div(
    [
        html.H4("Average Normalized Cluster PT / Gen Pt"),
        html.Br(),
        dcc.Tabs(
            id = "particle",
            value = "photons",
            children = [
                dcc.Tab(label = "Photons", value = "photons"),
                dcc.Tab(label = "Electrons", value = "electrons"),
                dcc.Tab(label = "Pions", value = "pions")
                ]
        ),
        html.Br(),
        html.Div([dbc.Button("Pile Up", id="pileup", color="primary", n_clicks=0),
                  dcc.Input(id="pt_cut", value="10", type='text'),
                  dbc.Button("Download Figure", id="download", color="primary", n_clicks=0)]),
        html.Br(),
        html.Label("Coef:"),
        dcc.Slider(id="coef", min=0.0, max=0.05, value=0.01, marks=marks),
        html.Br(),
        html.P("EtaRange:"),
        dcc.RangeSlider(id="eta_range", min=1.7, max=2.9, step=0.1, value=[1.7, 2.7]),
        html.Br(),
        dcc.Graph(id="graph", mathjax=True),
    ]
)

@callback(
    Output("graph", "figure"),
    Input("eta_range", "value"),
    Input("pt_cut", "value"),
    Input("pileup", "n_clicks"),
    Input("particle", "value"),
    Input("coef", "value"),
    Input("download", "n_clicks")
)
def plot_norm(
    eta_range, pt_cut, pileup, particle, coef, download
):
    # even number of clicks --> PU0, odd --> PU200 (will reset with other callbacks otherwise)
    pileup_key = "PU0" if pileup%2==0 else "PU200"
    init_files = input_files[pileup_key]
    
    df_original, df_weighted = cl_helpers.get_dataframes(init_files, particle, coef, eta_range, pt_cut)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("PT Norm Vs Gen Pt", "PT Norm Vs Gen Eta"))
    vals = {}
    pt_plot_args = {"traces": {"111": {}}}
    eta_plot_args = {"traces": {"111": {}}}
    y_axis_title = r"$<\frac{{p_T}^{Cl}}{{p_T}^{Gen}}>$" if particle != "electrons" else r"$mode(\frac{{p_T}^{Cl}}{{p_T}^{Gen}})$"
    for key, df in zip(("original", "weighted"), (df_original, df_weighted)):
        bins = 20 if particle != "pions" else 5

        df["gen_pt_bin"] = pd.cut(df.gen_pt, bins=bins, labels=False)
        df["gen_eta_bin"] = pd.cut(df.gen_eta, bins=bins, labels=False)

        if ((pileup_key == "PU200") & (key == "weighted")):
            norm_cols = ["pt_norm", "pt_norm_eta_corr"]
            print(df.keys())
            sub_df  = df[["eta", "gen_eta", "gen_eta_bin", "pt", "gen_pt", "pt_norm", "pt_norm_eta_corr", "gen_pt_bin"]].reset_index()
        else:
            norm_cols = ["pt_norm"]
            sub_df  = df[["eta", "gen_eta", "gen_eta_bin", "pt", "gen_pt", "pt_norm", "gen_pt_bin"]].reset_index()
        
        gen_pt = sub_df.groupby("gen_pt_bin").apply(lambda x: x.gen_pt.mean())
        gen_eta = sub_df.groupby("gen_eta_bin").apply(lambda x: x.gen_eta.mean())

        # x-axis errors (gen-level)
        gen_pt_err, rel_gen_pt_err = cl_helpers.rel_err(sub_df, "gen_pt", bin_col="gen_pt_bin")
        gen_eta_err, _ = cl_helpers.rel_err(sub_df, "gen_eta", bin_col="gen_eta_bin")

        for col in norm_cols:
            legend_name = "layer_corr" if key == "weighted" else "unweighted"
            legend_name += "_eta_corr" if col == "pt_norm_eta_corr" else ""

            if key == "original":
                color = "blue"
            else:
                color = "red" if col == "pt_norm" else "green"

            if particle == "electrons":
                avg_pt_vs_gen_pt = cl_helpers.get_binned_modes(sub_df, "gen_pt_bin", col)
                avg_pt_vs_gen_eta = cl_helpers.get_binned_modes(sub_df, "gen_eta_bin", col)
            else:
                avg_pt_vs_gen_pt = sub_df.groupby("gen_pt_bin").apply(lambda x: x[col].mean())
                avg_pt_vs_gen_eta = sub_df.groupby("gen_eta_bin").apply(lambda x: x[col].mean())

            _, rel_pt_err_vs_gen_pt = cl_helpers.rel_err(sub_df, col, bin_col="gen_pt_bin")
            _, rel_pt_err_vs_gen_eta = cl_helpers.rel_err(sub_df, col, bin_col="gen_eta_bin")

            pt_norm_err_vs_gen_pt = avg_pt_vs_gen_pt*np.sqrt(rel_pt_err_vs_gen_pt**2+rel_gen_pt_err**2)
            pt_norm_err_vs_gen_eta = avg_pt_vs_gen_eta*np.sqrt(rel_pt_err_vs_gen_eta**2+rel_gen_pt_err**2)

            pt_df = pd.DataFrame.from_dict({"gen_pt": gen_pt, "pt_norm": avg_pt_vs_gen_pt})
            eta_df = pd.DataFrame.from_dict({"gen_eta": gen_eta, "pt_norm": avg_pt_vs_gen_eta})

            means = {"pt": pt_df, "eta": eta_df}

            vals[key] = means

            pt_plot_args["traces"]["111"][legend_name] = {"plot_type": "scatter",
                                                          "x_data": vals[key]["pt"]["gen_pt"],
                                                          "y_data": vals[key]["pt"]["pt_norm"],
                                                          "x_title": r"${p_T}^{Gen}$",
                                                          "y_title": y_axis_title,
                                                          "color": color,
                                                          #"xerr": gen_pt_err,
                                                          #"yerr": pt_norm_err_vs_gen_pt
                                                          }
            fig.add_trace(
                go.Scatter(
                    x = vals[key]["pt"]["gen_pt"],
                    y = vals[key]["pt"]["pt_norm"],
                    name = legend_name + " ( pT )",
                    line = dict(color=color),
                    mode = "markers",
                    error_x = dict(
                        type="data",
                        array=gen_pt_err,
                        visible=True
                    ),
                    error_y = dict(
                        type="data",
                        array=pt_norm_err_vs_gen_pt,
                        visible=True
                    ),
                ),
                row=1,
                col=1
            )

            eta_plot_args["traces"]["111"][legend_name] = {"plot_type": "scatter",
                                                           "x_data": vals[key]["eta"]["gen_eta"],
                                                           "y_data": vals[key]["eta"]["pt_norm"],
                                                           "x_title": r"$|{\eta}^{Gen}|$",
                                                           "y_title": y_axis_title,
                                                           "color": color,
                                                           #"xerr": gen_eta_err,
                                                           #"yerr": pt_norm_err_vs_gen_eta
                                                           }

            fig.add_trace(
                go.Scatter(
                    x = vals[key]["eta"]["gen_eta"],
                    y = vals[key]["eta"]["pt_norm"],
                    name = legend_name + " ( |eta| )",
                    line = dict(color=color),
                    mode = "markers",
                    error_x = dict(
                        type="data",
                        array=gen_eta_err,
                        visible=True
                    ),
                    error_y = dict(
                        type="data",
                        array=pt_norm_err_vs_gen_eta,
                        visible=True
                    ),
                ),
                row=1,
                col=2
            )

        fig.update_xaxes(title_text=r"$\huge{{p_T}^{Gen}}$",
                         row=1,
                         col=1,
                         minor=dict(showgrid=True, dtick=20))
        fig.update_xaxes(title_text=r"$\huge{|\eta^{Gen}|}$",
                         row=1,
                         col=2,
                         minor=dict(showgrid=True, dtick=0.1))
        fig.update_yaxes(row=1,
                         col=1,
                         minor=dict(showgrid=True, dtick=0.05))
        fig.update_yaxes(row=1,
                         col=2,
                         minor=dict(showgrid=True, dtick=0.05))

    y_axis_title = r"$\huge{<\frac{{p_T}^{Cl}}{{p_T}^{Gen}}>}$" if particle != "electrons" else r"$\huge{mode(\frac{{p_T}^{Cl}}{{p_T}^{Gen}})}$"
    fig.update_layout(
        yaxis_title_text=y_axis_title,
    )

    #pt_text = r"${p_T}^{Gen} > $" + r"${}$".format(pt_cut)
    #eta_text = r"${} < \eta < {}$".format(eta_range[0], eta_range[1])

    pt_text = cl_plot_funcs.pt_text(pt_cut)
    eta_text = cl_plot_funcs.eta_text(eta_range)
    radius_text = cl_plot_funcs.radius_text(coef)

    pt_plot_args["eta_text"] = eta_text
    pt_plot_args["pt_text"] = pt_text
    pt_plot_args["radius_text"] = radius_text

    eta_plot_args["eta_text"] = eta_text
    eta_plot_args["pt_text"] = pt_text
    eta_plot_args["radius_text"] = radius_text

    pt_plot_title = r"$p_T \: Response \: vs. \: {p_T}^{Gen}$"
    pt_plot_title = cl_plot_funcs.title(pt_plot_title, pileup_key)

    pt_plot_path = "plots/png/pT_response_vs_PT_{}_eta_{}_{}_ptGtr_{}_{}_err.png"
    pt_plot_path = cl_plot_funcs.path(pt_plot_path, pileup_key, eta_range, pt_cut, particle)

    eta_plot_title = r"$p_T \: Response \: vs. \: |{\eta}^{Gen}|$"
    eta_plot_title = cl_plot_funcs.title(eta_plot_title, pileup_key)

    eta_plot_path = "plots/png/pT_response_vs_Eta_{}_eta_{}_{}_ptGtr_{}_{}_err.png"
    eta_plot_path = cl_plot_funcs.path(eta_plot_path, pileup_key, eta_range, pt_cut, particle)

    pt_cms_plot = cl_plot_funcs.cmsPlot(pt_plot_title, pt_plot_path, **pt_plot_args)
    eta_cms_plot = cl_plot_funcs.cmsPlot(eta_plot_title, eta_plot_path, **eta_plot_args)

    if download > 0:
        pt_cms_plot.write_fig()
        eta_cms_plot.write_fig()

    return fig

if __name__=="__main__":
    #test = plot_norm([1.6, 2.7], 0, 0, "electrons", 0.01)
    cl_plot_funcs.update_particle_callback()
    cl_plot_funcs.update_pileup_callback()
