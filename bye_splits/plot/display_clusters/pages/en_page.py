import os
import sys

parent_dir = os.path.abspath(__file__ + 5 * "/..")
sys.path.insert(0, parent_dir)

from bye_splits.utils import common

import re
import numpy as np
import pandas as pd
import yaml

from dash import Dash, dcc, html, Input, Output, callback, ctx
import dash

import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import argparse
import bye_splits
from bye_splits.utils import params, parsing, cl_helpers

parser = argparse.ArgumentParser(description="")
parsing.add_parameters(parser)
FLAGS = parser.parse_args()

cfg = cl_helpers.read_cl_size_params()

if cfg["local"]:
    data_dir = cfg["localDir"]
else:
    data_dir = params.EOSStorage(FLAGS.user, cfg["dataFolder"])

input_files = cfg["dashApp"]

dash.register_page(__name__, title="Energy", name="Energy")

layout = html.Div(
    [
        html.H4("Normalized Cluster Energy"),
        html.Hr(),
        html.Div([dbc.Button("Pile Up", id="pileup_en", color="primary", n_clicks=0),
                  dcc.Input(id="pt_cut", value="PT Cut", type='text'),
                  dbc.Button("Weighted", id="weight_en", color="primary", n_clicks=0)]),
        #html.P("Weight Version (Electrons)"),
        #dcc.Dropdown(["originalWeights", "ptNormGtr0p8", "withinOneSig"], "originalWeights", id="weight_version"),
        dcc.Graph(id="cl-en-graph", mathjax=True),
        html.P("EtaRange:"),
        dcc.RangeSlider(id="eta_range", min=1.4, max=2.7, step=0.1, value=[1.6, 2.7]),
        html.P("Normalization:"),
        dcc.Dropdown(["Energy", "PT"], "PT", id="normby"),
    ]
)


def fill_dict_w_mean_norm(key, eta, pt_cut, df, norm, out_dict):
    pt_cut = float(pt_cut) if pt_cut!="PT Cut" else 0
    df = df[(df.gen_eta > eta[0]) & (df.gen_eta < eta[1]) & (df.pt > pt_cut) ]
    if norm == "PT":
        mean_energy = df["pt_norm"].mean()
    else:
        mean_energy = df["en_norm"].mean()

    out_dict[key] = np.append(out_dict[key], mean_energy)


def write_plot_file(input_files, norm, eta, pt, weight, coefs, outfile):
    normed_energies = {}
    for key in input_files.keys():
        # Initialize at 0 since we only consider coefs[1:] (coefs[0] is an empty dataframe)
        if len(input_files[key]) > 0:
            normed_energies[key] = [0.0]

    normed_energies = {}
    for coef in coefs:
        dfs_by_particle = cl_helpers.get_dfs(input_files, coef, weight)
        dfs_by_particle = cl_helpers.filter_dfs(dfs_by_particle, eta, pt)
        for particle in dfs_by_particle.keys():
            if particle not in normed_energies.keys():
                normed_energies[particle] = [0.0]
            else:
                df = dfs_by_particle[particle]
                df = df[ df["matches"] == True ]
                fill_dict_w_mean_norm(particle, eta, pt, df, norm, normed_energies)

    with pd.HDFStore(outfile, "w") as PlotFile:
        normed_df = pd.DataFrame.from_dict(normed_energies)
        PlotFile.put("Normed_Dist", normed_df)

    return normed_energies

'''coef_keys = cl_helpers.get_keys(input_files["PU0"])
test = write_plot_file(input_files["PU0"], "PT", [1.6,2.7], 10, True, coef_keys, "something")'''

@callback(
        Output("pileup_en", "color"),
        Input("pileup_en", "n_clicks")
)
def update_pu_button(n_clicks):
    return cl_helpers.update_button(n_clicks)
    
@callback(
        Output("weight_en", "color"),
        Input("weight_en", "n_clicks")
)
def update_weight_button(n_clicks):
    return cl_helpers.update_button(n_clicks)

@callback(
    Output("cl-en-graph", "figure"),
    Input("normby", "value"),
    Input("eta_range", "value"),
    Input("pt_cut", "value"),
    Input("pileup_en", "n_clicks"),
    Input("weight_en", "n_clicks"),
    #Input("weight_version", "value")
)
def plot_norm(
    normby, eta_range, pt_cut, pileup_en, weight_en, plot_file="normed_distribution"
):
    # even number of clicks --> PU0, odd --> PU200 (will reset with other callbacks otherwise)
    init_files = input_files["PU0"] if pileup_en%2==0 else input_files["PU200"]
    weight_bool = False if weight_en%2==0 else True

    coef_keys = cl_helpers.get_keys(init_files)
    global y_axis_title

    if normby == "Energy":
        y_axis_title = r"$\huge{\frac{\bar{E_{Cl}}}{\bar{E}_{Gen}}}$"
    elif normby == "PT":
        y_axis_title = r"$\huge{\frac{\bar{p_T}^{Cl}}{\bar{p_T}^{Gen}}}$"
    else:
        y_axis_title = r"$\huge{\frac{E_{Cl}}{E_{Max}}}$"

    pt_str = "0" if pt_cut=="PT Cut" else str(pt_cut)
    pt_cut = float(pt_str)

    plot_filename = "{}_{}_eta_{}_pt_gtr_{}_{}_matched".format(
        normby, plot_file, eta_range[0], eta_range[1], pt_str
    )
    plot_filename += "" if weight_en%2==0 else "_weightedFinal"
    plot_filename += "_PU0.hdf5" if pileup_en%2==0 else "_PU200_AllParticles.hdf5"
    
    pile_up_dir = "PU0" if pileup_en%2==0 else "PU200"

    plot_filename_user = "{}{}/{}".format(data_dir, pile_up_dir, plot_filename)
    plot_filename_iehle = "{}{}{}/new_{}".format(
        cfg["ehleDir"],
        cfg["dataFolder"],
        pile_up_dir,
        plot_filename,
    )

    if os.path.exists(plot_filename_user):
        with pd.HDFStore(plot_filename_user, "r") as PlotFile:
            normed_dist = PlotFile["/Normed_Dist"].to_dict(orient="list")
    elif os.path.exists(plot_filename_iehle):
        with pd.HDFStore(plot_filename_iehle, "r") as PlotFile:
            normed_dist = PlotFile["/Normed_Dist"].to_dict(orient="list")
    else:
        normed_dist = write_plot_file(init_files, normby, eta_range, pt_cut, weight_bool, coef_keys, plot_filename_user)

    start, end, tot = cfg["coeffs"]
    coefs = np.linspace(start, end, tot)

    coef_labels = [round(coef, 3) for coef in coefs]
    coef_labels = coef_labels[0::5]

    fig = make_subplots(rows=1, cols=2, subplot_titles=("EM", "Hadronic"))
    
    particles = normed_dist.keys()
    colors = ["green", "purple", "red"]
    colors = colors[:len(particles)]

    for particle, color in zip(particles, colors):
        fig.add_trace(go.Scatter(
                            x=coefs,
                            y=normed_dist[particle],
                            name=particle.capitalize(),
                            line_color=color,
                        ),
                        row=1,
                        col=2 if particle=="pions" else 1
        )

    fig.add_hline(y=1.0, line_dash="dash", line_color="green")

    fig.update_xaxes(title_text="Radius (Coeff)")

    fig.update_layout(
        title_text="Normalized {} Distribution".format(normby),
        yaxis_title_text=y_axis_title,
    )

    return fig
