import os
import sys

parent_dir = os.path.abspath(__file__ + 5 * "/..")
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import math

from dash import dcc, html, Input, Output, callback
import dash

import dash_bootstrap_components as dbc

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import argparse
from bye_splits.utils import params, parsing, cl_helpers, cl_plot_funcs

import scipy.stats as st

parser = argparse.ArgumentParser(description="")
parsing.add_parameters(parser)
FLAGS = parser.parse_args()

cfg = cl_helpers.read_cl_size_params()

if cfg["local"]:
    data_dir = cfg["localDir"]
else:
    data_dir = params.EOSStorage(FLAGS.user, cfg["dataFolder"])

#input_files = cfg["dashApp"]
input_files = cfg["dashApp"]["local"] if cfg["local"] else cfg["dashApp"]["EOS"]

marks = {
    coef: {"label": format(coef, ".3f"), "style": {"transform": "rotate(-90deg)"}}
    for coef in np.arange(0.0, 0.05, 0.001)
}

dash.register_page(__name__, title="Res (vs. PT, Eta)", name="Res (vs. PT, Eta)")

layout = html.Div(
    [
        html.H4("Resolution vs. PT and Eta"),
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
        dcc.Graph(id="res_graph", mathjax=True),
    ]
)

@callback(
    Output("res_graph", "figure"),
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

    dash_plot = cl_plot_funcs.dashPlot(particle, pileup_key)
    dash_plot.plot_res(coef, df_original, df_weighted, "gen_pt", fig, (1,1))
    dash_plot.plot_res(coef, df_original, df_weighted, "gen_eta", fig, (1,2))

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
                        minor=dict(showgrid=True, dtick=0.01))
    fig.update_yaxes(row=1,
                        col=2,
                        minor=dict(showgrid=True, dtick=0.01))

    y_axis_title = r"$\huge{Res}$"
    fig.update_layout(
        yaxis_title_text=y_axis_title,
    )

    """pt_text = r"${p_T}^{Gen} > $" + r"${}$".format(pt_cut)
    eta_text = r"${} < \eta < {}$".format(eta_range[0], eta_range[1])

    '''pt_plot_args["eta_text"] = eta_text
    pt_plot_args["pt_text"] = pt_text
    pt_plot_args["radius_text"] = cl_plot_funcs.radius_text(coef)

    eta_plot_args["eta_text"] = eta_text
    eta_plot_args["pt_text"] = pt_text
    eta_plot_args["radius_text"] = cl_plot_funcs.radius_text(coef)'''

    pt_plot_title = r"$Resolution \: vs. \: {p_T}^{Gen}$"
    pt_plot_title = cl_plot_funcs.title(pt_plot_title, pileup_key)

    #pt_plot_path = "plots/png/res_vs_pT_{}_eta_{}_{}_ptGtr_{}_{}.png".format(pileup_key, cl_helpers.annot_str(eta_range[0]), cl_helpers.annot_str(eta_range[1]), cl_helpers.annot_str(pt_cut), particle)
    pt_plot_path = "plots/png/res_vs_pT_{}_eta_{}_{}_ptGtr_{}_{}_final.png"
    pt_plot_path = cl_plot_funcs.path(pt_plot_path, pileup_key, eta_range, pt_cut, particle)

    eta_plot_title = r"$Resolution \: vs. \: {\eta}^{Gen}$"
    eta_plot_title = cl_plot_funcs.title(eta_plot_title, pileup_key)

    #eta_plot_path = "plots/png/res_vs_eta_{}_eta_{}_{}_ptGtr_{}_{}.png".format(pileup_key, cl_helpers.annot_str(eta_range[0]), cl_helpers.annot_str(eta_range[1]), cl_helpers.annot_str(pt_cut), particle)
    eta_plot_path = "plots/png/res_vs_eta_{}_eta_{}_{}_ptGtr_{}_{}_final.png"
    eta_plot_path = cl_plot_funcs.path(eta_plot_path, pileup_key, eta_range, pt_cut, particle)

    pt_cms_plot = cl_plot_funcs.cmsPlot(pt_plot_title, pt_plot_path, **pt_plot_args)
    eta_cms_plot = cl_plot_funcs.cmsPlot(eta_plot_title, eta_plot_path, **eta_plot_args)

    if download > 0:
        pt_cms_plot.write_fig()
        eta_cms_plot.write_fig()"""

    return fig

if __name__=="__main__":
    #test = plot_norm([1.6, 2.7], 0, 0, "electrons", 0.01)
    cl_plot_funcs.update_particle_callback()
    cl_plot_funcs.update_pileup_callback()
