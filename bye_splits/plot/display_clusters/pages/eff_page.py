import os
import sys

from dash import dcc, html, Input, Output, callback, ctx
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import yaml

parent_dir = os.path.abspath(__file__ + 5 * "/..")
sys.path.insert(0, parent_dir)

import argparse
from bye_splits.utils import params, parsing, cl_helpers

parser = argparse.ArgumentParser(description="Clustering standalone step.")
parsing.add_parameters(parser)
FLAGS = parser.parse_args()

cfg = cl_helpers.read_cl_size_params()

if cfg["local"]:
    data_dir = cfg["localDir"]
else:
    data_dir = params.EOSStorage(FLAGS.user, cfg["dataFolder"])

input_files = cfg["dashApp"]

def binned_effs(df, norm, perc=0.1):
    """Takes a dataframe 'df' with a column 'norm' to normalize by, and returns
    1) a binned matching efficiency list
    2) a binned list corresponding to 'norm'
    where the binning is done by percentage 'perc' of the size of the 'norm' column"""
    eff_list = [0]
    en_list = [0]
    en_bin_size = perc * (df[norm].max() - df[norm].min())
    if perc < 1.0:
        current_en = 0
        for i in range(1, 101):
            match_column = df.loc[
                df[norm].between(current_en, (i) * en_bin_size, "left"), "matches"
            ]
            if not match_column.empty:
                try:
                    eff = float(match_column.value_counts(normalize=True))
                except TypeError:
                    eff = match_column.value_counts(normalize=True)[True]
                eff_list.append(eff)
                current_en += en_bin_size
                en_list.append(current_en)
    else:
        match_column = df.loc[df[norm].between(0, en_bin_size, "left"), "matches"]
        if not match_column.empty:
            try:
                eff = float(match_column.value_counts(normalize=True))
            except TypeError:
                eff = match_column.value_counts(normalize=True)[True]
            eff_list = eff
    return eff_list, en_list


# Dash page setup
##############################################################################################################################

marks = {
    coef: {"label": format(coef, ".3f"), "style": {"transform": "rotate(-90deg)"}}
    for coef in np.arange(0.0, 0.05, 0.001)
}

dash.register_page(__name__, title="Efficiency", name="Efficiency")

layout = dbc.Container(
    [
        dbc.Row(
            [
                html.Div(
                    "Reconstruction Efficiency",
                    style={"fontSize": 30, "textAlign": "center"},
                )
            ]
        ),
        html.Hr(),
        html.Div([dbc.Button("Pile Up", id="pileup", color="primary", n_clicks=0),
                  dcc.Input(id="pt_cut", value="PT Cut", type='text')]),
        html.Hr(),
        dcc.Graph(id="eff-graph", mathjax=True),
        html.P("Coef:"),
        dcc.Slider(id="coef", min=0.0, max=0.05, value=0.001, marks=marks),
        html.P("EtaRange:"),
        dcc.RangeSlider(id="eta_range", min=1.4, max=2.7, step=0.1, value=[1.6, 2.7]),
        html.P("Normalization:"),
        dcc.Dropdown(["Energy", "PT"], "PT", id="normby"),
        html.Hr(),
        dbc.Row(
            [
                dcc.Markdown(
                    "Global Efficiencies", style={"fontSize": 30, "textAlign": "center"}
                )
            ]
        ),
        html.Div(id="glob-effs"),
        html.Hr(),
        dbc.Row(
            [
                dcc.Markdown(
                    "Efficiencies By Coefficent",
                    style={"fontSize": 30, "textAlign": "center"},
                )
            ]
        ),
        dcc.Graph(id="glob-eff-graph", mathjax=True),
    ]
)


# Callback function for display_color() which displays binned efficiency/energy graphs
@callback(
    Output("eff-graph", "figure"),
    Output("glob-effs", "children"),
    Input("coef", "value"),
    Input("eta_range", "value"),
    Input("pt_cut", "value"),
    Input("normby", "value"),
    Input("pileup", "n_clicks"),
)

##############################################################################################################################

def display_color(coef, eta_range, pt_cut, normby, pileup):
    # even number of clicks --> PU0, odd --> PU200 (will reset with other callbacks otherwise)
    init_files = input_files["PU0"] if pileup%2==0 else input_files["PU200"]

    df_by_particle = cl_helpers.get_dfs(init_files, coef)
    pt_cut = float(pt_cut) if pt_cut!="PT Cut" else 0
    df_by_particle = cl_helpers.filter_dfs(df_by_particle, eta_range, pt_cut)  # {particle: (df, rms, eff_rms)}

    fig = make_subplots(rows=1, cols=2, subplot_titles=("EM", "Hadronic"))
    col_name = "gen_pt" if normby != "Energy" else "gen_en"

    particles = df_by_particle.keys()
    colors = ["green", "purple", "red"]
    colors = colors[:len(particles)]

    glob_eff_dict = {}
    for particle in particles:
        df = df_by_particle[particle]
        effs, x = binned_effs(df, col_name)
        glob_eff_dict[particle] = np.mean(effs[1:])
        fig.add_trace(
            go.Scatter(
            x=x,
            y=effs,
            name=particle.capitalize()
            ),
            row=1,
            col=2 if particle=="pions" else 1
        )

    glob_effs = pd.DataFrame(glob_eff_dict, index=[0])

    fig.update_xaxes(title_text="{} (GeV)".format(normby))

    fig.update_yaxes(type="log")

    fig.update_layout(
        title_text="Efficiency/{}".format(normby),
        yaxis_title_text=r"$Eff (\frac{N_{Cl}}{N_{Gen}})$",
    )

    return fig, dbc.Table.from_dataframe(glob_effs)

def write_eff_file(init_files, norm, coefs, eta, pt_cut, file):
    pt_cut = float(pt_cut) if pt_cut!="PT Cut" else 0
    binned_var = "gen_en" if norm == "Energy" else "gen_pt"
    
    effs_dict = {}
    for coef in coefs:
        dfs_by_particle = cl_helpers.get_dfs(init_files, coef)
        dfs_by_particle = cl_helpers.filter_dfs(dfs_by_particle, eta, pt_cut)

        for particle in dfs_by_particle.keys():
            # Note that a cluster having radius (coef) zero also has zero efficiency, so we initialize as such
            if particle not in effs_dict.keys():
                effs_dict[particle] = [0.0]
            else:
                eff, _ = binned_effs(dfs_by_particle[particle], binned_var, 1.0)
                effs_dict[particle] = np.append(effs_dict[particle], eff)

    with pd.HDFStore(file, "w") as glob_eff_file:
        glob_eff_file.put("Eff", pd.DataFrame.from_dict(effs_dict))

    return effs_dict


# Callback function for global_effs() which displays global efficiency as a function of the coefficent/radius
@callback(
    Output("glob-eff-graph", "figure"),
    Input("eta_range", "value"),
    Input("pt_cut", "value"),
    Input("normby", "value"),
    Input("pileup", "n_clicks"),
)
def global_effs(eta_range, pt_cut, normby, pileup, file="global_eff"):
    # even number of clicks --> PU0, odd --> PU200 (will reset with other callbacks otherwise)
    init_files = input_files["PU0"] if pileup%2==0 else input_files["PU200"]

    coefs = cl_helpers.get_keys(init_files)

    pt_str = "0" if pt_cut=="PT Cut" else pt_cut

    filename = "{}_eta_{}_{}_pt_gtr_{}_{}".format(normby, eta_range[0], eta_range[1], pt_str, file)
    filename += "_PU0.hdf5" if pileup%2==0 else "_PU200.hdf5"
    
    pile_up_dir = "PU0" if pileup%2==0 else "PU200"

    filename_user = "{}{}/{}".format(data_dir, pile_up_dir, filename)
    filename_iehle = "{}{}{}/{}".format(
        cfg["ehleDir"], cfg["dataFolder"], pile_up_dir, filename
    )

    if os.path.exists(filename_user):
        with pd.HDFStore(filename_user, "r") as glob_eff_file:
            effs_by_coef = glob_eff_file["/Eff"].to_dict(orient="list")
    elif os.path.exists(filename_iehle):
        with pd.HDFStore(filename_iehle, "r") as glob_eff_file:
            effs_by_coef = glob_eff_file["/Eff"].to_dict(orient="list")
    else:
        effs_by_coef = write_eff_file(init_files, normby, coefs, eta_range, pt_cut, filename_user)

    coefs = np.linspace(0.0, 0.05, 50)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("EM", "Hadronic"))

    particles = effs_by_coef.keys()
    colors = ["green", "purple", "red"]
    colors = colors[:len(particles)]

    for particle, color in zip(particles, colors):
        fig.add_trace(
            go.Scatter(
                x=coefs,
                y=effs_by_coef[particle],
                name=particle.capitalize(),
                line_color=color,
            ),
            row=1,
            col=2 if particle == "pions" else 1
        )

    fig.update_xaxes(title_text="Radius (Coefficient)")

    fig.update_layout(
        title_text="Efficiency/Radius",
        yaxis_title_text=r"$Eff (\frac{N_{Cl}}{N_{Gen}})$",
    )

    return fig
