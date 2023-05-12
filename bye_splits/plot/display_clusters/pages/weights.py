#!/usr/bin/env python

import os
import sys

parent_dir = os.path.abspath(__file__ + 4 * "/..")
sys.path.insert(0, parent_dir)

from bye_splits.utils import params, common

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
                  dcc.Dropdown(["Version 1", "Official"], "Official", id="version"),
                  dcc.Dropdown(["Weights", "Distributions"], "Weights", id="mode")]),
        html.Hr(),
        dcc.Graph(id="cl-en-weights", mathjax=True),
    ]
)

def get_weights(dir, cfg, version, mode):
    weights_by_particle = {}
    for particle in ("photons", "electrons", "pions"):
        particle_dir = dir+particle+"/optimization/"
        particle_dir += "v1/" if version=="Version 1" else "official/"

        plot_dir = particle_dir+"/plots/"
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        basename = cfg["clusterStudies"]["optimization"]["baseName"]
        files = [f for f in os.listdir(particle_dir) if basename in f]

        weights_by_radius = {}
        for file in files:
            radius = float(file.replace(".hdf5","").replace("optimization_","").replace("r","").replace("p","."))
            infile = particle_dir+file
            with pd.HDFStore(infile, "r") as optWeights:
                weights_by_radius[radius] = optWeights[mode]
    
        weights_by_particle[particle] = weights_by_radius
    
    return weights_by_particle

with open(params.CfgPath, mode="r") as afile:
    cfg = yaml.safe_load(afile)

opt_dir = "{}/PU0/".format(cfg["clusterStudies"]["localDir"])

@callback(
    Output("cl-en-weights", "figure"),
    Input("radius", "value"),
    Input("version", "value"),
    Input("mode", "value")
)
def plot_weights(radius, version, mode):
    plot = "weights" if mode=="Weights" else "df"
    weights_by_particle = get_weights(opt_dir, cfg, version, plot)

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
            old_norm = weights.en/weights.gen_en
            new_norm = weights.new_en/weights.gen_en

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

def save_fig(dir, pars, cfg):
    weights_by_particle = get_weights(dir, cfg)

    fig = plot_weights(weights_by_particle, pars.radius)

    out_plot_dir = "{}/plots/".format(dir)
    if not os.path.exists(out_plot_dir):
        os.makedirs(out_plot_dir)
    
    radius_str = str(pars.radius).replace(".","p")
    out_file = "{}/{}_r{}.html".format(out_plot_dir, cfg["clusterStudies"]["optimization"]["baseName"], radius_str)

    fig.write_html(out_file)

