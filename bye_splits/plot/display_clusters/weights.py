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

app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])
from dash_bootstrap_templates import load_figure_template

app.layout = html.Div(
    [
        html.H4("Cluster Energy Weights by Layer"),
        html.P("Radius:"),
        html.Div([dcc.Dropdown(radii_rounded,0.01,id="radius"),
                  dcc.Dropdown(["Version 1", "Version 2"], "Version 1", id="version")]),
        html.Hr(),
        dcc.Graph(id="cl-en-weights", mathjax=True),
    ]
)

def get_weights(dir, cfg, version):
    weights_by_particle = {}
    for particle in ("photons", "electrons", "pions"):
        particle_dir = dir+particle+"/optimization/"
        particle_dir += "" if version=="Version 1" else "v2/"

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
                weights_by_radius[radius] = optWeights["weights"]
    
        weights_by_particle[particle] = weights_by_radius
    
    return weights_by_particle

with open(params.CfgPath, mode="r") as afile:
    cfg = yaml.safe_load(afile)

opt_dir = "{}/PU0/".format(cfg["clusterStudies"]["localDir"])

@app.callback(
    Output("cl-en-weights", "figure"),
    Input("radius", "value"),
    Input("version", "value")
)
def plot_weights(radius, version):
    weights_by_particle = get_weights(opt_dir, cfg, version)

    particles = ("photons", "electrons", "pions") if version == "Version 1" else ("photons","electrons")

    fig = go.Figure()

    for particle in particles:
        weights = weights_by_particle[particle][radius]

        fig.add_trace(
            go.Bar(
            name=particle,
            x=weights.index,
            y=weights.weights,
            text=weights.weights,
            textposition="auto"
            )
        )

    fig.update_layout(barmode="group",
                      yaxis=dict(title="Weights"),
                      xaxis=dict(title="layer"))
    fig.update_yaxes(type="log")

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
    

if __name__ == "__main__":
    host, port = "0.0.0.0", 8080
    
    app.run_server(port=port, host=host, debug=True)

