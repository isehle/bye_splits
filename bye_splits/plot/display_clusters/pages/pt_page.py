import os
import sys

parent_dir = os.path.abspath(__file__ + 5 * "/..")
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd

from dash import dcc, html, Input, Output, callback, ctx
import dash

import dash_bootstrap_components as dbc

import plotly.graph_objects as go

import argparse
from bye_splits.utils import params, parsing, cl_helpers, cl_plot_funcs
from bye_splits.utils.cl_plot_funcs import cmsPlot

parser = argparse.ArgumentParser(description="")
parsing.add_parameters(parser)
FLAGS = parser.parse_args()

cluster_data = cl_helpers.clusterData()

dash.register_page(__name__, title="PT", name="PT")

layout = html.Div(
    [
        html.H4("Normalized Cluster PT / Radius"),
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
        html.Div([dbc.Button("Pile Up", id="pileup", color="primary", n_clicks=1),
                  dcc.Input(id="pt_cut", value=10.0),
                  dbc.Button("Download Figure", id="download", color="primary", n_clicks=0),
                  dbc.Button("Particle/Calibration View", id="tab_switch", color="primary", n_clicks=0)]),
        dcc.Graph(id="cl-pt-graph", mathjax=True),
        html.P("EtaRange:"),
        dcc.RangeSlider(id="eta_range", min=1.7, max=2.7, step=0.1, value=[1.7, 2.7]),
    ]
)

@callback(
    Output("particle", "children"),
    Output("particle", "value"),
    Input("particle", "value"),
    Input("tab_switch", "n_clicks"),
    Input("pileup", "n_clicks")
)
def switch_type(particle, tab_switch, pileup):
    if tab_switch % 2 == 0:
        return (
            [dcc.Tab(label = "Photons", value = "photons"),
             dcc.Tab(label = "Electrons", value = "electrons"),
             dcc.Tab(label = "Pions", value = "pions")],
             particle
        )
    else:
        return (
            [dcc.Tab(label = "Original", value = "original"),
             dcc.Tab(label = "Layer", value = "layer"),
             dcc.Tab(label = "Eta", value = "eta")],
             "layer" if pileup%2==0 else "eta"
        )


@callback(
    Output("cl-pt-graph", "figure"),
    Input("eta_range", "value"),
    Input("pt_cut", "value"),
    Input("pileup", "n_clicks"),
    Input("tab_switch", "n_clicks"),
    Input("particle", "value"),
    Input("download", "n_clicks")
)
def plot_norm(
    eta_range, pt_cut, pileup, tab_switch, particle, download
):
    # even number of clicks --> PU0, odd --> PU200 (will reset with other callbacks otherwise)
    pileup_key = "PU0" if pileup%2==0 else "PU200"

    fig = go.Figure()
    dash_plot = cl_plot_funcs.dashPlot(particle, pileup_key)

    if tab_switch % 2 == 0:
        if particle != "electrons":
            glob_pt_norm = cluster_data.get_global_pt(particle, eta_range, pt_cut, pileup_key)
        else:
            glob_pt_norm = cluster_data.get_global_pt(particle, eta_range, pt_cut, pileup_key, "peak")
        
        fig = dash_plot.plot_global_pt(glob_pt_norm, fig)
    else:
        glob_pt_norm = {
            "photons"  : cluster_data.get_global_pt("photons", eta_range, pt_cut, pileup_key),
            "electrons": cluster_data.get_global_pt("electrons", eta_range, pt_cut, pileup_key, "peak"),
            "pions"    : cluster_data.get_global_pt("pions", eta_range, pt_cut, pileup_key)
        }

        fig = dash_plot.plot_global_pt(glob_pt_norm, fig, version="calib", calib=particle)

    y_axis_title = r"$\huge{\langle \frac{{p_T}^{Cl}}{{p_T}^{Gen}} \rangle}$" if particle != "electrons" else r"$\huge{mode(\frac{{p_T}^{Cl}}{{p_T}^{Gen}})}$"
    x_axis_title = "Radius (Coeff)"
    
    fig.update_xaxes(title_text=x_axis_title, minor=dict(showgrid=True, dtick=0.001))

    fig.update_layout(
        title_text="Normalized PT Distribution",
    )

    fig.update_yaxes(title_text=y_axis_title, minor=dict(showgrid=True, dtick=0.05))

    if download > 0:
        symbols = {"photons": "\gamma",
                   "electrons": "e",
                   "pions"    : "\pi",
                   "original" : "Original",
                   "layer"    : "Layer \; Weighted",
                   "eta"      : "|\eta| \; Calibrated"}

        symbol = symbols[particle]

        plot_title = r"${} \;".format(symbol) + r"p_T \; Response \; vs. \; r^{Cl}$"

        plot_path  = os.path.join(parent_dir, "plots/png/pT_response_vs_radius_{}_eta_{}_{}_ptGtr_{}_{}_calibration.png".format(pileup_key, eta_range[0], eta_range[1], pt_cut, particle))
        dash_plot.download_plot(plot_title, plot_path)

    return fig

if __name__=="__main__":
    cl_plot_funcs.update_particle_callback()
    cl_plot_funcs.update_pileup_callback()
