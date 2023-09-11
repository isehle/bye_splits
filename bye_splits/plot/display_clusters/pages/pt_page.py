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

cfg = cl_helpers.read_cl_size_params()

if cfg["local"]:
    data_dir = cfg["localDir"]
else:
    data_dir = params.EOSStorage(FLAGS.user, cfg["dataFolder"])

input_files = cfg["dashApp"]

dash.register_page(__name__, title="PT", name="PT")

layout = html.Div(
    [
        html.H4("Normalized Cluster PT / Radius"),
        html.Br(),
        dcc.Tabs(
            id = "particle",
            #value = "photons",
            #value = "electrons",
            value = "pions",
            children = [
                dcc.Tab(label = "Photons", value = "photons"),
                dcc.Tab(label = "Electrons", value = "electrons"),
                dcc.Tab(label = "Pions", value = "pions")
                ]
        ),
        html.Div([dbc.Button("Pile Up", id="pileup", color="primary", n_clicks=1),
                  dcc.Input(id="pt_cut", value=10.0),
                  dbc.Button("Download Figure", id="download", color="primary", n_clicks=0)]),
        dcc.Graph(id="cl-pt-graph", mathjax=True),
        html.P("EtaRange:"),
        dcc.RangeSlider(id="eta_range", min=1.7, max=2.7, step=0.1, value=[1.7, 2.7]),
    ]
)

@callback(
    Output("cl-pt-graph", "figure"),
    Input("eta_range", "value"),
    Input("pt_cut", "value"),
    Input("pileup", "n_clicks"),
    Input("particle", "value"),
    Input("download", "n_clicks")
)
def plot_norm(
    eta_range, pt_cut, pileup, particle, download
):
    # even number of clicks --> PU0, odd --> PU200 (will reset with other callbacks otherwise)
    pileup_key = "PU0" if pileup%2==0 else "PU200"
    init_files = input_files[pileup_key]

    if particle != "electrons":
        glob_pt_norm = cl_helpers.get_global_pt(init_files[particle], eta_range, pt_cut, pileup_key)
    else:
        #glob_pt_norm = cl_helpers.get_global_pt(init_files[particle], eta_range, pt_cut, pileup_key, mode="mode")
        glob_pt_norm = cl_helpers.get_peak(init_files[particle], eta_range, pt_cut, pileup_key)

    fig = go.Figure()

    if particle == "photons":
        part_sym = "$\gamma$"
    elif particle == "electrons":
        part_sym = "$e$"
    elif particle == "pions":
        part_sym = "$\pi$"

    #plot_title = r"${} p_T$ Response".format(particle.capitalize()[:-1])
    #plot_title = r"${} \: p_T$ Response \: (PU200, {}\: Gev)".format(part_sym, "${p_T}^{Gen} > 50 $")
    plot_title = r"{} Response (PU200, {} GeV)".format(part_sym, "${p_T}^{Gen} > 10$")
    plot_path = "plots/png/pT_response_{}_eta_{}_{}_ptGtr_{}_{}_grid_zeroInterceptWeights.png".format(pileup_key, str(eta_range[0]).replace(".","p"), str(eta_range[1]).replace(".","p"), str(pt_cut).replace(".","p"), particle)
    
    y_axis_title = r"$\huge{\langle \frac{{p_T}^{Cl}}{{p_T}^{Gen}} \rangle}$" if particle != "electrons" else r"$\huge{mode(\frac{{p_T}^{Cl}}{{p_T}^{Gen}})}$"
    x_axis_title = "Radius (Coeff)"

    plot_args = {"traces": {"111": {}}}

    for key in glob_pt_norm.keys():
        #info= ["scatter", np.arange(0.0, 0.05, 0.001), glob_pt_norm[key], x_axis_title, y_axis_title, None]
        if key == "original":
            color = "blue"
        elif key == "layer":
            color = "red"
        else:
            color = "green"

        info = {"plot_type": "plot",
                            "x_data": np.arange(0.001, 0.05, 0.001),
                            "y_data": glob_pt_norm[key][1:],
                            "x_title": x_axis_title,
                            "y_title": y_axis_title,
                            "color": color,
                            "linestyle": "solid"}
        
        plot_args["traces"]["111"][key] = info

        '''if key == "original":
            info = {"plot_type": "plot",
                                "x_data": np.arange(0.0, 0.05, 0.001),
                                "y_data": glob_pt_norm[key],
                                "x_title": x_axis_title,
                                "y_title": y_axis_title,
                                "color": color,
                                "linestyle": "solid"}
            
            plot_args["traces"]["111"][key] = info'''

        fig.add_trace(
            go.Scatter(
                x = np.arange(0.0, 0.05, 0.001),
                y = glob_pt_norm[key],
                name = key.capitalize()
            )
        )

    plot_args["hline"] = {"val": 1.0, "color": "black", "linestyle": "--"}
    
    cms_plot = cl_plot_funcs.cmsPlot(plot_title, plot_path, **plot_args)

    fig.add_hline(y=1.0, line_dash="dash", line_color="green")
    
    fig.update_xaxes(title_text=x_axis_title, minor=dict(showgrid=True, dtick=0.001))

    fig.update_layout(
        title_text="Normalized PT Distribution",
    )

    fig.update_yaxes(title_text=y_axis_title, minor=dict(showgrid=True, dtick=0.05))

    if download > 0:
        cms_plot.write_fig()

    return fig

if __name__=="__main__":
    cl_plot_funcs.update_particle_callback()
    cl_plot_funcs.update_pileup_callback()
