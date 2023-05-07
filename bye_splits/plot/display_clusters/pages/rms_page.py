import os
import sys
from dash import dcc, html, Input, Output, callback, ctx, Dash
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import random

parent_dir = os.path.abspath(__file__ + 5 * "/..")
sys.path.insert(0, parent_dir)

import argparse
from bye_splits.utils import params, parsing, cl_helpers

parser = argparse.ArgumentParser(description="")
parsing.add_parameters(parser)
FLAGS = parser.parse_args()

cfg = cl_helpers.read_cl_size_params()

if cfg["local"]:
    data_dir = cfg["localDir"]
else:
    data_dir = params.EOSStorage(FLAGS.user, "data/")

input_files = cfg["dashApp"]

def rms(data):
    return np.sqrt(np.mean(np.square(data)))

def effrms(data, c=0.68):
    """Compute half-width of the shortest interval
    containing a fraction 'c' of items in a 1D array.
    """
    assert data.shape == (data.shape[0],)
    new_series = data.dropna()
    x = np.sort(new_series, kind="mergesort")
    m = int(c * len(x)) + 1
    out = np.min(x[m:] - x[:-m]) / 2.0

    return out

def get_rms(init_files, coef, eta_range, pt_cut, normby, match, rms_dict=None, rms_eff_dict=None):
    dfs_by_particle = cl_helpers.get_dfs(init_files, coef)
    pt_cut = float(pt_cut) if pt_cut!="PT Cut" else 0

    dfs_by_particle = cl_helpers.filter_dfs(dfs_by_particle, eta_range, pt_cut)

    new_dict = {}
    for particle, df in dfs_by_particle.items():
        df = df[ df["matches"] == True] if match%2!=0 else df
        norm = df["en_norm"] if normby == "Energy" else df["pt_norm"]
        rms = norm.std() / norm.mean()
        eff_rms = effrms(norm) / norm.mean()
        new_dict[particle] = (df, rms, eff_rms)
        if rms_dict!=None:
            rms_dict[particle] = np.append(rms_dict[particle], rms)
            rms_eff_dict[particle] = np.append(rms_eff_dict[particle], eff_rms)
    
    if rms_dict==None:
        return new_dict


# Dash page setup
##############################################################################################################################

marks = {
    coef: {"label": format(coef, ".3f"), "style": {"transform": "rotate(-90deg)"}}
    for coef in np.arange(0.0, 0.05, 0.001)
}

dash.register_page(__name__, title="RMS", name="RMS")

layout = dbc.Container(
    [
        dbc.Row(
            [
                html.Div(
                    "Interactive Normal Distribution",
                    style={"fontSize": 40, "textAlign": "center"},
                )
            ]
        ),
        html.Hr(),
        html.Div([dbc.Button("Pile Up", id="pileup", color="primary", n_clicks=0),
                  dcc.Input(id="pt_cut", value="PT Cut", type='text'),
                  dbc.Button("dR Matching", id='match', color="primary", n_clicks=0)]),
        html.Hr(),
        dcc.Graph(id="histograms-x-graph", mathjax=True),
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
                    r"Gaussianity := $\frac{|RMS-RMS_{Eff}|}{RMS}$",
                    mathjax=True,
                    style={"fontSize": 30, "textAlign": "center"},
                )
            ]
        ),
        html.Hr(),
        html.Div(id="my_table"),
        html.Hr(),
        dbc.Row(
            [dcc.Markdown("Resolution", style={"fontSize": 30, "textAlign": "center"})]
        ),
        dcc.Graph(id="global-rms-graph", mathjax=True),
    ]
)


@callback(
    Output("histograms-x-graph", "figure"),
    Output("my_table", "children"),
    Input("coef", "value"),
    Input("eta_range", "value"),
    Input("pt_cut","value"),
    Input("normby", "value"),
    Input("pileup", "n_clicks"),
    Input("match", "n_clicks"),
)

##############################################################################################################################


def plot_dists(coef, eta_range, pt_cut, normby, pileup, match):
    init_files = input_files["PU0"] if pileup%2==0 else input_files["PU200"]

    data_by_particle = get_rms(init_files, coef, eta_range, pt_cut, normby, match) # {particle: (df, rms, eff_rms)}
    
    fig = go.Figure()

    col_name = "pt_norm" if normby!="Energy" else "en_norm"
    vals = {}
    for particle in data_by_particle.keys():
        df, rms, eff_rms = data_by_particle[particle]

        bins = np.linspace(min(df[col_name]),max(df[col_name]), 100)
        df[col_name + "_bin"] = pd.cut(df[col_name], bins=bins, labels=False)
        
        sub_df = df[[col_name, col_name + "_bin"]].reset_index()
        binned_rand_evs = sub_df.groupby(col_name + "_bin")["event"].apply(lambda x: "Random Event: "+str(np.random.choice(x)))

        vals_with_overflow = [np.minimum(1.2, val) for val in df[col_name]]
        #tick_marks = [str(val) if val != 1.2 else " > 1.2 " for val in vals_with_overflow] # Not currently showing up

        fig.add_trace(go.Histogram(x = vals_with_overflow, nbinsx=100, autobinx=False, name=particle.capitalize(), hovertext=binned_rand_evs))
        #fig.update_xaxes(ticktext=tick_marks)

        gauss_diff = np.abs(eff_rms - rms) / rms
        gauss_str = format(gauss_diff, ".3f")

        vals[particle] = {"RMS": rms, "Effective RMS": eff_rms, "Gaussianity": gauss_str}

        val_df = pd.DataFrame(vals).reset_index()
        val_df = val_df.rename(columns={"index": ""})

        val_table = dbc.Table.from_dataframe(val_df)

    if normby == "Energy":
        x_title = r"$\Huge{\frac{E_{Cl}}{E_{Gen}}}$"
    else:
        x_title = r"$\Huge{\frac{{p_T}^{Cl}}{{p_T}^{Gen}}}$"

    fig.update_layout(
        barmode="overlay",
        title_text="Normalized Cluster {}".format(normby),
        xaxis_title=x_title,
        yaxis_title_text=r"$\Large{Events}$",
    )

    fig.update_traces(opacity=0.5)

    return fig, val_table

#fig, tab = plot_dists(0.05, [1.6,2.7], 10, "PT", 1, 1)

def write_rms_file(init_files, coefs, eta, pt, norm, match, filename):
    rms_by_part, rms_eff_by_part = {}, {}
    for particle in init_files.keys():
        if len(init_files[particle])>0:
            rms_by_part[particle] = []
            rms_eff_by_part[particle] = []

    for coef in coefs[1:]:
        get_rms(init_files, coef, eta, pt, norm, match, rms_by_part, rms_eff_by_part)

    with pd.HDFStore(filename, "w") as glob_rms_file:
        glob_rms_file.put("RMS", pd.DataFrame.from_dict(rms_by_part))
        glob_rms_file.put("Eff_RMS", pd.DataFrame.from_dict(rms_eff_by_part))

    return rms_by_part, rms_eff_by_part


@callback(
    Output("global-rms-graph", "figure"),
    Input("eta_range", "value"),
    Input("pt_cut", "value"),
    Input("normby", "value"),
    Input("pileup", "n_clicks"),
    Input("match", "n_clicks"),
)
def glob_rms(eta_range, pt_cut, normby, pileup, match, file="rms_and_eff"):
    # even number of clicks --> PU0, odd --> PU200 (will reset with other callbacks otherwise)
    init_files = input_files["PU0"] if pileup%2==0 else input_files["PU200"]

    coefs = cl_helpers.get_keys(init_files)

    pt_cut = "0" if pt_cut=="PT Cut" else pt_cut

    filename = "{}_eta_{}_{}_pt_gtr_{}_{}".format(
        normby, str(eta_range[0]), str(eta_range[1]), pt_cut, file
    )

    filename += "_matched" if match%2!=0 else ""

    filename += "_PU0.hdf5" if pileup%2==0 else "_PU200_AllParticles.hdf5"
    pile_up_dir = "PU0" if pileup%2==0 else "PU200"

    filename_user = "{}{}/{}".format(data_dir, pile_up_dir, filename)
    filename_iehle = "{}{}{}/{}".format(
        cfg["ehleDir"], cfg["dataFolder"], pile_up_dir, filename
    )

    if os.path.exists(filename_user):
        with pd.HDFStore(filename_user, "r") as glob_rms_file:
            rms_by_part, rms_eff_by_part = glob_rms_file["/RMS"].to_dict(
                orient="list"
            ), glob_rms_file["/Eff_RMS"].to_dict(orient="list")
    elif os.path.exists(filename_iehle):
        with pd.HDFStore(filename_iehle, "r") as glob_rms_file:
            rms_by_part, rms_eff_by_part = glob_rms_file["/RMS"].to_dict(
                orient="list"
            ), glob_rms_file["/Eff_RMS"].to_dict(orient="list")
    else:
        rms_by_part, rms_eff_by_part = write_rms_file(
            init_files, coefs, eta_range, pt_cut, normby, match, filename_user
        )

    nice_coefs = np.linspace(0.0, 0.05, 50)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("EM", "Hadronic"))

    particles = rms_by_part.keys()
    colors = ["green", "purple", "red"]
    colors = colors[:len(particles)]

    for var in ("RMS", "Eff-RMS"):
        for particle, color in zip(particles, colors):
            fig.add_trace(
                go.Scatter(
                    x=nice_coefs,
                    y=rms_by_part[particle]
                    if var == "RMS"
                    else rms_eff_by_part[particle],
                    name="{} {}".format(particle.capitalize(), var),
                    line_color=color,
                    mode="lines" if var == "RMS" else "markers",
                ),
                row=1,
                col=2 if particle == "pions" else 1,
            )

    fig.update_xaxes(title_text="Radius (Coefficient)")

    fig.update_layout(
        title_text="Resolution in {}".format(normby), yaxis_title_text="(Effective) RMS"
    )

    return fig