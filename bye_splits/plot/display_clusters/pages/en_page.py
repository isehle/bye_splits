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
        html.Div([dbc.Button("Pile Up", id="pileup", color="primary", n_clicks=0),
                  dcc.Input(id="pt_cut", value="PT Cut", type='text'),
                  dbc.Button("dR Matching", id='match', color="primary", n_clicks=0)]),
        html.Hr(),
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


def write_plot_file(input_files, norm, eta, pt, match, outfile, pars=vars(FLAGS)):
    normed_energies = {}
    for key in input_files.keys():
        # Initialize at 0 since we only consider coefs[1:] (coefs[0] is an empty dataframe)
        if len(input_files[key]) > 0:
            normed_energies[key] = [0.0]

    for key in input_files.keys():
        if len(input_files[key]) == 0:
            continue
        elif len(input_files[key]) == 1:
            with pd.HDFStore(input_files[key][0], "r") as File:
                coef_strs = File.keys()
                for coef in coef_strs[1:]:
                    df = File[coef] if File[coef].index.name=="event" else File[coef].set_index("event")
                    df = df[ df["matches"] == True ] if match%2 !=0 else df
                    fill_dict_w_mean_norm(
                        key, eta, pt, df, norm, normed_energies
                    )
        else:
            file_list = [pd.HDFStore(val, "r") for val in input_files[key]]
            coef_strs = file_list[0].keys()
            for coef in coef_strs[1:]:
                df_list = [file_list[i][coef] for i in range(len(file_list))]
                full_df = pd.concat(df_list)
                full_df = (
                    full_df.set_index("event")
                    if not full_df.index.name == "event"
                    else full_df
                )
                full_df = full_df[ full_df["matches"]==True ] if match%2 !=0 else full_df
                fill_dict_w_mean_norm(
                    key, eta, pt, full_df, norm, normed_energies
                )

            for file in file_list:
                file.close()

    with pd.HDFStore(outfile, "w") as PlotFile:
        normed_df = pd.DataFrame.from_dict(normed_energies)
        PlotFile.put("Normed_Dist", normed_df)

    return normed_energies


@callback(
    Output("cl-en-graph", "figure"),
    Input("normby", "value"),
    Input("eta_range", "value"),
    Input("pt_cut", "value"),
    Input("pileup", "n_clicks"),
    Input("match", "n_clicks"),
)
def plot_norm(
    normby, eta_range, pt_cut, pileup, match, plot_file="normed_distribution"
):
    # even number of clicks --> PU0, odd --> PU200 (will reset with other callbacks otherwise)
    init_files = input_files["PU0"] if pileup%2==0 else input_files["PU200"]
    global y_axis_title

    if normby == "Energy":
        y_axis_title = r"$\huge{\frac{\bar{E_{Cl}}}{\bar{E}_{Gen}}}$"
    elif normby == "PT":
        y_axis_title = r"$\huge{\frac{\bar{p_T}^{Cl}}{\bar{p_T}^{Gen}}}$"
    else:
        y_axis_title = r"$\huge{\frac{E_{Cl}}{E_{Max}}}$"

    pt_str = "0" if pt_cut=="PT Cut" else pt_cut

    plot_filename = "{}_{}_eta_{}_pt_gtr_{}_{}".format(
        normby, plot_file, eta_range[0], eta_range[1], pt_str
    )
    plot_filename += "_matched" if match%2 != 0 else ""
    plot_filename += "_PU0.hdf5" if pileup%2==0 else "_PU200_withElec.hdf5"
    
    pile_up_dir = "PU0" if pileup%2==0 else "PU200"

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
        normed_dist = write_plot_file(init_files, normby, eta_range, pt_cut, match, plot_filename_user)

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
