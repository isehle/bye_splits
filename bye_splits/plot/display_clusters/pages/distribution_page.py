# pyright: reportUnboundVariable=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportInvalidStringEscapeSequence=false

import os
import sys

from dash import dcc, html, Input, Output, callback, ctx
import dash
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
from scipy.stats import crystalball
from scipy.optimize import curve_fit, Bounds

from flask import send_file

import io

parent_dir = os.path.abspath(__file__ + 5 * "/..")
sys.path.insert(0, parent_dir)

import argparse
from bye_splits.utils import params, parsing, cl_helpers, cl_plot_funcs

parser = argparse.ArgumentParser(description="")
parsing.add_parameters(parser)
FLAGS = parser.parse_args()

cluster_data = cl_helpers.clusterData()

# Dash page setup
##############################################################################################################################

marks = {
    coef: {"label": format(coef, ".3f"), "style": {"transform": "rotate(-90deg)"}}
    for coef in np.arange(0.0, 0.05, 0.001)
}

dash.register_page(__name__, title="Distributions", name="Distributions")

layout = dbc.Container(
    [
        dbc.Row(
            [
                html.Div(
                    "Distributions and Resolutions",
                    style={"fontSize": 40, "textAlign": "center"},
                )
            ]
        ),
        html.Br(),
        dcc.Tabs(
            id = "tab",
            value = "photons",
            children = [
                dcc.Tab(label = "Photons", value = "photons"),
                dcc.Tab(label = "Electrons", value = "electrons"),
                dcc.Tab(label = "Pions", value = "pions")
                ]),
        html.Br(),
        dbc.Row(
            children = [
                dbc.Col(
                    dbc.Button("Pile Up", id="pileup", color="primary", n_clicks=0),
                    width = 2,
                ),
                dbc.Col(
                    children = [
                        html.Label("PT Cut: "),
                        dcc.Input(id="pt_cut", value=10.0),
                    ],
                    width = 2,
                ),
                dbc.Col(
                    dbc.Button("Download Figure", id="download", color="primary", n_clicks=0),
                    width = 2
                ),
                dbc.Col(
                    dbc.Button("Calibration View", id="switch_tab", color="primary", n_clicks=0)
                )
            ],
            justify="start",
            align="center",
        ),
        html.Br(),
        html.Label("Coef:"),
        dcc.Slider(id="coef", min=0.0, max=0.05, value=0.01, marks=marks),
        html.Label("EtaRange:"),
        dcc.RangeSlider(id="eta_range", min=1.7, max=2.9, step=0.1, value=[1.7, 2.7]),
        html.Label("Distribution Range (Eff_RMS):"),
        dcc.RangeSlider(id="sig_range", min=-5,max=5,step=1, value=[-2, 2]),
        dbc.Row(
            [
                dcc.Graph(id="pt_dists", mathjax=True),
            ]
        )
    ]
)

@callback(
    Output("tab", "children"),
    Output("tab", "value"),
    Input("tab", "value"),
    Input("switch_tab", "n_clicks"),
    Input("pileup", "n_clicks")
)
def switch_tabs(tab, switch_tab, pileup):
    if switch_tab % 2 == 0:
        return (
            [dcc.Tab(label = "Photons", value = "photons"),
             dcc.Tab(label = "Electrons", value = "electrons"),
             dcc.Tab(label = "Pions", value = "pions")],
             tab
        )
    else:
        return (
            [dcc.Tab(label = "Original", value = "original"),
             dcc.Tab(label = "Layer", value = "layer"),
             dcc.Tab(label = "Eta", value = "eta")],
             "layer"
        )

@callback(
    Output("pt_dists", "figure"),
    Input("tab", "value"),
    Input("coef", "value"),
    Input("eta_range", "value"),
    Input("sig_range", "value"),
    Input("pt_cut","value"),
    Input("pileup", "n_clicks"),
    Input("switch_tab", "n_clicks"),
    Input("download", "n_clicks")
)
##############################################################################################################################

def plot_dists(tab, coef, eta_range, sig_range, pt_cut, pileup, switch_tab, download):
    global glob_res
    pileup_key = "PU0" if pileup%2==0 else "PU200"

    dash_plot = cl_plot_funcs.dashPlot(tab, pileup_key)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Distributions", "Resolutions"))
    if switch_tab % 2 == 0:
        df_original, df_weighted = cluster_data.get_dataframes(pileup_key, tab, coef, eta_range, pt_cut)
        
        fig = dash_plot.plot_hist(coef, df_original, df_weighted, fig)
    else:
        dfs_o, dfs_w = dict.fromkeys(("photons", "electrons", "pions")), dict.fromkeys(("photons", "electrons", "pions"))
        for particle in dfs_o.keys():
            df_o, df_w = cluster_data.get_dataframes(pileup_key, particle, coef, eta_range, pt_cut)

            dfs_o[particle] = df_o
            dfs_w[particle] = df_w

        fig = dash_plot.plot_hist(coef, dfs_o, dfs_w, fig, version="calibration", calib=tab)


    x_title = r"$\Huge{\frac{{p_T}^{Cl}}{{p_T}^{Gen}}}$"
    fig.update_layout(barmode="overlay")

    fig.update_xaxes(title_text=x_title,
                     row=1,
                     col=1)
    fig.update_yaxes(title_text=r"$\Large{{Events_{bin}}/{Events_{tot}}}$",
                     row=1,
                     col=1,
                     minor=dict(showgrid=True, dtick=10, gridwidth=1, gridcolor="darkgrey"),
                     type="log"
                     )
    
    fig.update_xaxes(title_text=r"$\Huge{r^{Cl}}$",
                     row=1,
                     col=2,
                     minor=dict(showgrid=True, dtick=0.005, gridwidth=1))
    fig.update_yaxes(title_text=r"$\Large{Res}$",
                     row=1,
                     col=2,
                     minor=dict(showgrid=True, dtick=0.01, gridwidth=1))

    fig.update_traces(opacity=0.5, row=1, col=1)

    if ctx.triggered_id != "coef" and ctx.triggered_id != "switch_tab":
        glob_res = cluster_data.get_global_res(tab, eta_range, pt_cut, pileup_key)

    fig = dash_plot.plot_global_res(glob_res, fig, (1, 2))

    fig.update_yaxes(type="log",
                     row=1,
                     col=2)
    
    '''if download > 0:

        if particle == "photons":
            part_sym = "\gamma"
        elif particle == "electrons":
            part_sym = "e"
        elif particle == "pions":
            part_sym = "\pi"

        hist_title = r"${} \: {} \: Distribution \: ({}, r={})$".format(part_sym, "{p_T}^{Norm}", pileup_key, coef)
        hist_path = "plots/png/pTNormDist_{}_eta_{}_{}_ptGtr_{}_log_{}_bc_stc_radius{}_TEST.png".format(pileup_key, cl_helpers.annot_str(eta_range[0]), cl_helpers.annot_str(eta_range[1]), pt_cut, particle, str(coef).replace(".","p"))

        hist_cms = cl_plot_funcs.cmsPlot(hist_title, hist_path, **dist_args)

        hist_cms.write_fig()

        res_title = r"${}$ Res vs. ".format(part_sym) + r"$r^{Cl}$, (" + r"{}, ".format(pileup_key) + r"${p_T}^{Gen} > 10 GeV)$"
        res_path = "plots/png/res_vs_radius_{}_eta_{}_{}_ptGtr_{}_{}_bc_stc_TEST.png".format(pileup_key, cl_helpers.annot_str(eta_range[0]), cl_helpers.annot_str(eta_range[1]), pt_cut, particle)

        res_cms = cl_plot_funcs.cmsPlot(res_title, res_path, **res_args)

        res_cms.write_fig()'''

    return fig

if __name__ == "__main__":
    cl_plot_funcs.update_particle_callback()
    cl_plot_funcs.update_pileup_callback()