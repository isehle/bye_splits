#!/usr/bin/env python

import os
import sys

parent_dir = os.path.abspath(__file__ + 4 * "/..")
sys.path.insert(0, parent_dir)

from bye_splits.utils import params, cl_helpers

import pandas as pd
import numpy as np
import math

from dash import dcc, html, Input, Output, callback, ctx, Dash
import dash
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import argparse

import yaml

radii = np.linspace(0.0, 0.05, 50)
radii_rounded = [round(radius, 3) for radius in radii]

dash.register_page(__name__, title="Energy Weights / Layer", name="Weights")

layout = html.Div(
    [
        dbc.Row(
            [
                html.Div(
                    "Cluster Energy Weights by Layer",
                    style={"fontSize": 40, "textAlign": "center"},
                )
            ]
        ),
        html.P("Radius:"),
        html.Div([dcc.Dropdown(radii_rounded,0.01,id="radius"),
                  #dcc.Dropdown(["originalWeights", "ptNormGtr0p8", "withinOneSig"], "originalWeights", id="version"),
                  dcc.Dropdown(["Weights", "Distributions"], "Weights", id="mode"),
                  html.P("PT_Range"),
                  dcc.RangeSlider(id="pt_range", min=10., max=200., step=10., value=[10., 200.])]),
        html.Hr(),
        dcc.Graph(id="cl-en-weights", mathjax=True),
        html.Hr(),
        dcc.Graph(id="eta_corr", mathjax=True)
    ]
)

with open(params.CfgPath, mode="r") as afile:
    cfg = yaml.safe_load(afile)

opt_dir = "{}/PU0/".format(params.LocalStorage)
pu_dir = "{}/PU200/".format(params.LocalStorage)

@callback(
    Output("cl-en-weights", "figure"),
    Output("eta_corr", "figure"),
    Input("radius", "value"),
    #Input("version", "value"),
    Input("mode", "value"),
    Input("pt_range", "value")
)
def plot_weights(radius, mode, pt_range):
    plot = "weights" if mode=="Weights" else "df"
    
    weights_by_particle = cl_helpers.read_weights(opt_dir, cfg, mode=plot)
    eta_weights_by_particle = cl_helpers.read_pu_weights(cfg)

    particle_files = cfg["clusterStudies"]["dashApp"]["PU200"]

    eta = np.linspace(1.7, 2.7, 50)

    layer_fig = make_subplots(rows=2, cols=1, subplot_titles=("EM", "Hadronic")) if mode=="Weights" else make_subplots(rows=2, cols=2, specs=[[{}, {}],  [{"colspan": 2}, None]], subplot_titles=("Photons", "Electrons", "Pions", "N/A"))
    eta_fig = make_subplots(rows=1, cols=2, subplot_titles=("EM", "Hadronic"))

    #for particle in particles:
    for particle in ("photons", "pions"):
        _, weighted_particle_dfs = cl_helpers.get_dataframes(particle_files, particle, radius, [1.7, 2.7], 10)

        weighted_particle_dfs = weighted_particle_dfs[ (weighted_particle_dfs.gen_pt > pt_range[0]) & (weighted_particle_dfs.gen_pt < pt_range[1]) ]

        weighted_particle_dfs.sort_values("gen_eta", inplace=True)
        
        weighted_particle_dfs["gen_eta_bin"] = pd.cut(weighted_particle_dfs.gen_eta, bins=50, labels=False)
        weighted_particle_dfs["pt_diff"] = weighted_particle_dfs.gen_pt - weighted_particle_dfs.pt

        pt_diff = np.asarray(weighted_particle_dfs.groupby("gen_eta_bin").apply(lambda x: x.pt_diff.mean()))

        weights = weights_by_particle[particle][radius]

        pu_weights = eta_weights_by_particle[particle][radius]
        slope, intercept = np.full(eta.shape, pu_weights["slope"]), np.full(eta.shape, pu_weights["intercept"])
        pu_weight_vals = eta*slope + intercept

        '''if particle == "pions":
            weight_offset = np.full(pu_weight_vals.shape, pu_weight_vals[0])
            pu_weight_vals -= weight_offset

            diff_offset = np.full(pt_diff.shape, pt_diff[0])
            pt_diff -= diff_offset'''

        mse = np.square(np.subtract(pt_diff, pu_weight_vals)).mean()
        rmse = np.sqrt(mse)
        rmse_norm = rmse/pt_diff.mean()

        #weights = weights[ weights.index <= 28 ]
        if mode == "Weights":
            layer_fig.add_trace(
                go.Scatter(
                name=particle,
                x=weights.index,
                y=weights.weights,
                #text=weights.weights,
                #textposition="auto"
                ),
                row=1 if particle != "pions" else 2,
                col=1
            )
            eta_fig.add_trace(
                go.Scatter(
                    name = "PU Weights",
                    x = eta,
                    y = pu_weight_vals,
                ),
                row = 1,
                col = 1 if particle != "pions"  else 2
            )
            eta_fig.add_trace(
                go.Scatter(
                    name = particle,
                    x = eta,
                    y = pt_diff,
                ),
                row = 1,
                col = 1 if particle != "pions" else 2
            )
            eta_fig.add_annotation(
                text = "RMSE = "  + str(round(rmse_norm, 3)),
                xref = "paper",
                yref = "paper",
                x = 0.25 if particle != "pions" else 0.9,
                y = 0.9,
                showarrow = False,
            )
        else:
            old_norm = weights.pt/weights.gen_pt
            new_norm = weights.new_pt/weights.gen_pt

            layer_fig.add_trace(
                go.Histogram(
                    x = old_norm,
                    nbinsx=1000,
                    name="old_en_norm",
                    opacity=0.75,
                ),
                row=1 if particle != "pions" else 2,
                col=1 if particle != "electrons" else 2
            )
            layer_fig.add_trace(
                go.Histogram(
                    x = new_norm,
                    nbinsx=1000,
                    name="new_en_norm",
                    opacity=0.75,
                ),
                row=1 if particle != "pions" else 2,
                col=1 if particle != "electrons" else 2
            )

    if mode == "Weights":
        layer_fig.update_layout(barmode="group",
                        yaxis=dict(title="Weights"),
                        xaxis=dict(title="layer"))
        layer_fig.update_yaxes(type="log")

        eta_fig.update_layout(yaxis=dict(title=r"$a|\eta|+b$"),
                              xaxis=dict(title=r"$\eta$"))

    else:
        layer_fig.update_layout(
            barmode="overlay",
            title={
                "text": r"$\Huge{\frac{E^{Cl}}{E^{Gen}}}$",
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            },
            yaxis=dict(title="Counts"),
            #xaxis=dict(title=r"$\Huge{\frac{E^{Cl}}{E^{Gen}}}$")
        )

    layer_fig.update_yaxes(automargin=True)
    layer_fig.update_xaxes(automargin=True)

    eta_fig.update_yaxes(automargin=True)
    eta_fig.update_xaxes(automargin=True)

    return layer_fig, eta_fig

#layer_fig = plot_weights(0.01, "official", "weights")

