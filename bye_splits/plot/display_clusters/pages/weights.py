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
                  dcc.Dropdown(["Weights", "Distributions"], "Weights", id="mode")]),
        html.Hr(),
        dcc.Graph(id="cl-en-weights", mathjax=True),
    ]
)

with open(params.CfgPath, mode="r") as afile:
    cfg = yaml.safe_load(afile)

opt_dir = "{}/PU0/".format(params.LocalStorage)

@callback(
    Output("cl-en-weights", "figure"),
    Input("radius", "value"),
    #Input("version", "value"),
    Input("mode", "value")
)
def plot_weights(radius, mode):
    plot = "weights" if mode=="Weights" else "df"
    
    weights_by_particle = cl_helpers.read_weights(opt_dir, cfg, mode=plot)

    particles = weights_by_particle.keys()

    fig = make_subplots(rows=2, cols=1, subplot_titles=("EM", "Hadronic")) if mode=="Weights" else make_subplots(rows=2, cols=2, specs=[[{}, {}],  [{"colspan": 2}, None]], subplot_titles=("Photons", "Electrons", "Pions", "N/A"))

    for particle in particles:
        weights = weights_by_particle[particle][radius]
        if mode == "Weights":
            fig.add_trace(
                go.Bar(
                name=particle,
                x=weights.index,
                y=weights.weights,
                text=weights.weights,
                textposition="auto"
                ),
                row=1 if particle != "pions" else 2,
                col=1
            )
        else:
            old_norm = weights.pt/weights.gen_pt
            new_norm = weights.new_pt/weights.gen_pt

            fig.add_trace(
                go.Histogram(
                    x = old_norm,
                    nbinsx=1000,
                    name="old_en_norm",
                    opacity=0.75,
                ),
                row=1 if particle != "pions" else 2,
                col=1 if particle != "electrons" else 2
            )
            fig.add_trace(
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
        fig.update_layout(barmode="group",
                        yaxis=dict(title="Weights"),
                        xaxis=dict(title="layer"))
        fig.update_yaxes(type="log")
    else:
        fig.update_layout(
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

    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
    return fig

#fig = plot_weights(0.01, "official", "weights")

