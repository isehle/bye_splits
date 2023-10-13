# pyright: reportUnboundVariable=false
# pyright: reportGeneralTypeIssues=false
# pyright: reportInvalidStringEscapeSequence=false

import os
import sys

from dash import dcc, html, Input, Output, callback, ctx
import dash
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
from scipy.stats import crystalball
from scipy.optimize import curve_fit, Bounds

from flask import send_file

import io

parent_dir = os.path.abspath(__file__ + 5 * "/..")
sys.path.insert(0, parent_dir)

import argparse
from bye_splits.utils import params, parsing, cl_helpers, cl_plot_funcs

parser = argparse.ArgumentParser(description="")
parsing.add_parameters(parser)
FLAGS = parser.parse_args()

cluster_data = cl_helpers.clusterData()

# Dash page setup
##############################################################################################################################

marks = {
    coef: {"label": format(coef, ".3f"), "style": {"transform": "rotate(-90deg)"}}
    for coef in np.arange(0.0, 0.05, 0.001)
}

dash.register_page(__name__, title="Distributions", name="Distributions")

layout = dbc.Container(
    [
        dbc.Row(
            [
                html.Div(
                    "Distributions and Resolutions",
                    style={"fontSize": 40, "textAlign": "center"},
                )
            ]
        ),
        html.Br(),
        dcc.Tabs(
            id = "particle",
            value = "photons",
            children = [
                dcc.Tab(label = "Photons", value = "photons"),
                dcc.Tab(label = "Electrons", value = "electrons"),
                dcc.Tab(label = "Pions", value = "pions")
                ]),
        html.Br(),
        dbc.Row(
            children = [
                dbc.Col(
                    dbc.Button("Pile Up", id="pileup", color="primary", n_clicks=0),
                    width = 2,
                ),
                dbc.Col(
                    children = [
                        html.Label("PT Cut: "),
                        dcc.Input(id="pt_cut", value=10.0),
                    ],
                    width = 2,
                ),
                dbc.Col(
                    dbc.Button("Download Figure", id="download", color="primary", n_clicks=0),
                    width = 2
                )
            ],
            justify="start",
            align="center",
        ),
        html.Br(),
        html.Label("Coef:"),
        dcc.Slider(id="coef", min=0.0, max=0.05, value=0.01, marks=marks),
        html.Label("EtaRange:"),
        dcc.RangeSlider(id="eta_range", min=1.7, max=2.9, step=0.1, value=[1.7, 2.7]),
        html.Label("Distribution Range (Eff_RMS):"),
        dcc.RangeSlider(id="sig_range", min=-5,max=5,step=1, value=[-2, 2]),
        dbc.Row(
            [
                dcc.Graph(id="pt_dists", mathjax=True),
                html.Div(id="stat_info")
            ]
        )
    ]
)

@callback(
    Output("pt_dists", "figure"),
    Output("stat_info", "children"),
    Input("particle", "value"),
    Input("coef", "value"),
    Input("eta_range", "value"),
    Input("sig_range", "value"),
    Input("pt_cut","value"),
    Input("pileup", "n_clicks"),
    Input("download", "n_clicks")
)
##############################################################################################################################

def plot_dists(particle, coef, eta_range, sig_range, pt_cut, pileup, download):
    global glob_res
    pileup_key = "PU0" if pileup%2==0 else "PU200"

    df_original, df_weighted = cluster_data.get_dataframes(pileup_key, particle, coef, eta_range, pt_cut)

    if particle != "pions":
        weighted_cols = {"PU0": ["pt_norm"],
                        "PU200": ["pt_norm", "pt_norm_eta_corr"]}
    else:
        weighted_cols = {"PU0": ["pt_norm", "pt_norm_en_corrected"],
                         "PU200": ["pt_norm", "pt_norm_en_corrected", "pt_norm_eta_corrected"]}
    
    dists = {"original": df_original,
             "weighted": df_weighted}
    
    fig = go.Figure()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Distributions", "Resolutions"))
    vals = {}
    diff_means = {}

    dist_args = {"traces": {"111": {}}}
    for key in dists.keys():
        df = dists[key]
        if key == "original":
            col_names = ["pt_norm"]
        else:
            col_names = weighted_cols[pileup_key]

        for col_name in col_names:            
            if key == "original":
                color = "blue"
                name = key.capitalize()
            else:
                if col_name == "pt_norm":
                    color = "red"
                    name = "Layer Weighted"
                elif col_name == "pt_norm_eta_corrected" or col_name == "pt_norm_eta_corr":
                    color = "green"
                    name = "Eta Corrected"
                elif col_name == "pt_norm_en_corrected":
                    color = "purple"
                    name = "Energy Corrected"

            upper_res = df[col_name].mean() + sig_range[1]*df[col_name].std()
            lower_res = df[col_name].mean() + sig_range[0]*df[col_name].std()
            
            upper_eff = df[col_name].mean() + sig_range[1]*cl_helpers.effrms(df[col_name].to_frame().rename({0: col_name}))
            lower_eff = df[col_name].mean() + sig_range[0]*cl_helpers.effrms(df[col_name].to_frame().rename({0: col_name}))

            display_vals = df[ (df[col_name] > lower_eff) & (df[col_name] < upper_eff) ]

            mean_total_val = df[col_name].mean() if particle != "electrons" else cluster_data._get_gaus_mean(df, col_name)
            mean_display_val = display_vals[col_name].mean() if particle != "electrons" else cluster_data._get_gaus_mean(display_vals, col_name)

            vals[name + " Total"] = {"Events"                                    : len(df[col_name]),
                                     "Mean" if particle!= "electrons" else "Peak": mean_total_val,
                                     "Res"                                       : df[col_name].std()/mean_total_val,
                                     "Eff Res"                                   : cl_helpers.effrms(df[col_name])/mean_total_val}
            
            vals[name + " Displayed"] = {"Events"                                    : len(display_vals[col_name]),
                                         "Mean" if particle!= "electrons" else "Peak": mean_display_val,
                                         "Res"                                       : display_vals[col_name].std()/mean_display_val,
                                         "Eff Res"                                   : cl_helpers.effrms(display_vals[col_name])/mean_display_val}
            
            x_title = r"$\frac{{p_T}^{Cl}}{{p_T}^{Gen}}$"
            y_title = r"$Events$"

            plot_info = {"plot_type": "hist",
                                        "data": display_vals[col_name],
                                        "bins": 1000 if particle != "pions" else 100,
                                        "label": name,
                                        "x_title": x_title,
                                        "y_title": y_title,
                                        "color": color,
                                        "vline": {"val": np.mean(display_vals[col_name]), "color": color, "linestyle": "--"},
                                        }

            fig.add_trace(go.Histogram(
                                x = display_vals[col_name],
                                nbinsx=1000 if particle != "pions" else 100,
                                autobinx=False,
                                name=name,
                                histnorm="probability density",
                                marker_color=color,
                            ),
                            row=1,
                            col=1
            )

            dist_args["traces"]["111"][name] = plot_info

                ####################################################################################################

    vline_key = "Mean" if particle != "electrons" else "Peak"

    fig.add_vline(x=vals["Original Displayed"][vline_key], row=1, col=1, line_dash="dash", line_color="blue")
    fig.add_vline(x=vals["Layer Weighted Displayed"][vline_key], row=1, col=1, line_dash="dash", line_color="red")
    
    if pileup_key=="PU200":
        if particle != "pions":
            fig.add_vline(x=vals["Eta Corrected Displayed"][vline_key], row=1, col=1, line_dash="dash", line_color="green")
        else:
            fig.add_vline(x=vals["Energy Corrected Displayed"][vline_key], row=1, col=1, line_dash="dash", line_color="green")
            fig.add_vline(x=vals["Eta Corrected Displayed"][vline_key], row=1, col=1, line_dash="dash", line_color="green")

    
    stat_df = pd.DataFrame(vals).reset_index()
    stat_df = stat_df.rename(columns={"index": ""})
    stat_table = dbc.Table.from_dataframe(stat_df)

    x_title = r"$\Huge{\frac{{p_T}^{Cl}}{{p_T}^{Gen}}}$"
    fig.update_layout(barmode="overlay")

    fig.update_xaxes(title_text=x_title,
                     row=1,
                     col=1)
    fig.update_yaxes(title_text=r"$\Large{{Events_{bin}}/{Events_{tot}}}$",
                     row=1,
                     col=1,
                     minor=dict(showgrid=True, dtick=10, gridwidth=1, gridcolor="darkgrey"),
                     type="log"
                     )
    
    fig.update_xaxes(title_text=r"$\Huge{r^{Cl}}$",
                     row=1,
                     col=2,
                     minor=dict(showgrid=True, dtick=0.005, gridwidth=1))
    fig.update_yaxes(title_text=r"$\Large{Res}$",
                     row=1,
                     col=2,
                     minor=dict(showgrid=True, dtick=0.01, gridwidth=1))

    fig.update_traces(opacity=0.5, row=1, col=1)

    if ctx.triggered_id != "coef":
        glob_res = cluster_data.get_global_res(particle, eta_range, pt_cut, pileup_key)

    res_args = {"traces": {"111": {}}}
    if particle != "pions":
        colors = ("blue", "red") if pileup_key == "PU0" else ("blue", "red", "green")
    else:
        colors = ("blue", "red", "purple") if pileup_key == "PU0" else ("blue", "red", "purple", "green")
    for key, color in zip(glob_res.keys(), colors):
        rms, eff = glob_res[key].values()

        name = lambda x: f"{key.capitalize()}  {x}"
            
        res_args["traces"]["111"][name("RMS")] = {"plot_type": "plot",
                                                "x_data": np.arange(0.001, 0.05, 0.001),
                                                "y_data": rms,
                                                "x_title": r"$r$",
                                                "y_title": r"${\sigma \: ({\sigma}_{Eff})}/{\langle {p_T}^{Norm} \rangle}$",
                                                "color": color,
                                                "linestyle": "solid",
                                                "yscale": "log"}

        res_args["traces"]["111"][name("Eff_RMS")] = {"plot_type": "plot",
                                                    "x_data": np.arange(0.001, 0.05, 0.001),
                                                    "y_data": eff,
                                                    "x_title": r"$r$",
                                                    "y_title": r"${\sigma \: ({\sigma}_{Eff})}/{\langle {p_T}^{Norm} \rangle}$",
                                                    "color": color,
                                                    "linestyle": "dashed",
                                                    "yscale": "log"}

        fig.add_trace(
            go.Scatter(
                x = np.arange(0.001, 0.05, 0.001),
                y = rms,
                name = name("RMS"),
                line = dict(color=color)
            ),
            row = 1,
            col = 2
        )
        fig.add_trace(
            go.Scatter(
                x = np.arange(0.001, 0.05, 0.001),
                y = eff,
                name = name("Eff_RMS"),
                line=dict(color=color,
                          dash="dash")
            ),
            row=1,
            col=2
        )

    fig.update_yaxes(type="log",
                     row=1,
                     col=2)
    
    if download > 0:

        if particle == "photons":
            part_sym = "\gamma"
        elif particle == "electrons":
            part_sym = "e"
        elif particle == "pions":
            part_sym = "\pi"

        hist_title = r"${} \: {} \: Distribution \: ({}, r={})$".format(part_sym, "{p_T}^{Norm}", pileup_key, coef)
        hist_path = "plots/png/pTNormDist_{}_eta_{}_{}_ptGtr_{}_log_{}_bc_stc_radius{}_TEST.png".format(pileup_key, cl_helpers.annot_str(eta_range[0]), cl_helpers.annot_str(eta_range[1]), pt_cut, particle, str(coef).replace(".","p"))

        hist_cms = cl_plot_funcs.cmsPlot(hist_title, hist_path, **dist_args)

        hist_cms.write_fig()

        res_title = r"${}$ Res vs. ".format(part_sym) + r"$r^{Cl}$, (" + r"{}, ".format(pileup_key) + r"${p_T}^{Gen} > 10 GeV)$"
        res_path = "plots/png/res_vs_radius_{}_eta_{}_{}_ptGtr_{}_{}_bc_stc_TEST.png".format(pileup_key, cl_helpers.annot_str(eta_range[0]), cl_helpers.annot_str(eta_range[1]), pt_cut, particle)

        res_cms = cl_plot_funcs.cmsPlot(res_title, res_path, **res_args)

        res_cms.write_fig()

    return fig, stat_table

if __name__ == "__main__":
    cl_plot_funcs.update_particle_callback()
    cl_plot_funcs.update_pileup_callback()