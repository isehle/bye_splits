import os
import sys
from dash import dcc, html, Input, Output, callback, ctx
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from flask import send_file

import io

parent_dir = os.path.abspath(__file__ + 5 * "/..")
sys.path.insert(0, parent_dir)

import argparse
from bye_splits.utils import params, parsing, cl_helpers, cl_plot_funcs

parser = argparse.ArgumentParser(description="")
parsing.add_parameters(parser)
FLAGS = parser.parse_args()

cfg = cl_helpers.read_cl_size_params()

if cfg["local"]:
    data_dir = cfg["localDir"]
else:
    data_dir = params.EOSStorage(FLAGS.user, "data/")

input_files = cfg["dashApp"]

'''def get_dataframes(init_files, particles, coef, eta_range, pt_cut):
    df_o, df_w = cl_helpers.get_dfs(init_files, coef, particles)
    pt_cut = float(pt_cut) if pt_cut!="PT Cut" else 0
    df_o, df_w = cl_helpers.filter_dfs(df_o, eta_range, pt_cut), cl_helpers.filter_dfs(df_w, eta_range, pt_cut)

    return df_o, df_w'''


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
            #value = "pions",
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
        #html.Label("PT Cut: "),
        #dcc.RangeSlider(id="pt_range", min=0, max=100, step=5, value=[10, 100]),
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
    #Input("pt_range","value"),
    Input("pileup", "n_clicks"),
    Input("download", "n_clicks")
)
##############################################################################################################################

def plot_dists(particle, coef, eta_range, sig_range, pt_cut, pileup, download):
#def plot_dists(particle, coef, eta_range, sig_range, pt_range, pileup):
    global glob_res
    pileup_key = "PU0" if pileup%2==0 else "PU200"

    init_files = input_files[pileup_key]

    df_original, df_weighted = cl_helpers.get_dataframes(init_files, particle, coef, eta_range, pt_cut)
    #df_original, df_weighted = cl_helpers.get_dataframes(init_files, particle, coef, eta_range, pt_range)

    weighted_cols = {"PU0": ["pt_norm"],
                     "PU200": ["pt_norm", "pt_norm_eta_corr"]}
    
    dists = {"original": df_original,
             "weighted": df_weighted}
    
    fig = go.Figure()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Distributions", "Resolutions"))
    vals = {}

    dist_args = {"traces": {"111": {}}}
    dist_args["vline"] = {"val": 1.0, "color": "black", "linestyle": "--"}
    for key in dists.keys():
        df = dists[key]
        if key == "original":
            col_names = ["pt_norm"]
        else:
            col_names = weighted_cols[pileup_key]

        for col_name in col_names:
            if key != "original":
                name = "Layer Weights" if col_name == "pt_norm" else "Eta Correction"
                color = "red" if col_name == "pt_norm" else "green"
            else:
                name = "Original"
                color = "blue"

            #upper_val = df[col_name].mean() + sig_range[1]*df[col_name].std()
            #lower_val = df[col_name].mean() + sig_range[0]*df[col_name].std()
            
            upper_val = df[col_name].mean() + sig_range[1]*cl_helpers.effrms(df[col_name].to_frame().rename({0: col_name}))
            lower_val = df[col_name].mean() + sig_range[0]*cl_helpers.effrms(df[col_name].to_frame().rename({0: col_name}))

            #display_vals = df[ (df[col_name] > lower_val) & (df[col_name] < upper_val) ]
            display_vals = df

            vals[name + " Total"] = {"Events": len(df[col_name]),
                          "Mean": df[col_name].mean(),
                          "Res": df[col_name].std()/df[col_name].mean(),
                          "Eff Res": cl_helpers.effrms(df[col_name])/df[col_name].mean()}
            
            vals[name + " Displayed"] = {"Events": len(display_vals[col_name]),
                                         "Mean": display_vals[col_name].mean(),
                                         "Res": display_vals[col_name].std()/display_vals[col_name].mean(),
                                         "Eff Res": cl_helpers.effrms(display_vals[col_name])/display_vals[col_name].mean()}
            
            x_title = r"$\frac{{p_T}^{Cl}}{{p_T}^{Gen}}$"
            y_title = r"$Events$"
            plot_info = {"plot_type": "hist",
                         "data": display_vals[col_name],
                         "bins": 1000 if particle != "pions" else 100,
                         "label": name,
                         "x_title": x_title,
                         "y_title": y_title,
                         "color": color,
                         "vline": {"val": np.mean(display_vals[col_name]), "color": color, "linestyle": "--"}}
            
            dist_args["traces"]["111"][name] = plot_info

            fig.add_trace(go.Histogram(
                                x = display_vals[col_name],
                                nbinsx=1000 if particle != "pions" else 100,
                                autobinx=False,
                                name=name,
                                histnorm="probability",
                            ),
                            #marker_color=color,
                            row=1,
                            col=1
            )

    fig.add_vline(x=1.0, row=1, col=1, line_dash="dashdot", line_color="black")
    fig.add_vline(x=vals["Original Displayed"]["Mean"], row=1, col=1, line_dash="dash", line_color="blue")
    #fig.add_vline(x=vals["Original Total"]["Mean"], row=1, col=1, line_dash="dash", line_color="blue")
    fig.add_vline(x=vals["Layer Weights Displayed"]["Mean"], row=1, col=1, line_dash="dash", line_color="red")
    #fig.add_vline(x=vals["Layer Weights Total"]["Mean"], row=1, col=1, line_dash="dash", line_color="red")
    if pileup_key=="PU200":
        fig.add_vline(x=vals["Eta Correction Displayed"]["Mean"], row=1, col=1, line_dash="dash", line_color="green")
        #fig.add_vline(x=vals["Eta Correction Total"]["Mean"], row=1, col=1, line_dash="dash", line_color="green")
    
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
                     type="log")
    
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
        glob_res = cl_helpers.get_global_res(init_files[particle], eta_range, pt_cut, pileup_key)
        #glob_res = cl_helpers.get_global_res(init_files[particle], eta_range, pt_range, pileup_key)

    res_args = {"traces": {"111": {}}}
    colors = ("blue", "red") if pileup_key == "PU0" else ("blue", "red", "green")
    for key, color in zip(glob_res.keys(), colors):
        rms, eff = glob_res[key].values()

        name = lambda x: f"{key.capitalize()}  {x}"

        res_args["traces"]["111"][name("RMS")] = {"plot_type": "plot",
                                                  "x_data": np.arange(0.001, 0.05, 0.001),
                                                  "y_data": rms,
                                                  "x_title": r"$r$",
                                                  "y_title": r"$\sigma \: ({\sigma}_{Eff})$",
                                                  "color": color,
                                                  "linestyle": "solid",
                                                  "yscale": "log"}

        res_args["traces"]["111"][name("Eff_RMS")] = {"plot_type": "plot",
                                                      "x_data": np.arange(0.001, 0.05, 0.001),
                                                      "y_data": eff,
                                                      "x_title": r"$r$",
                                                      "y_title": r"$\sigma \: ({\sigma}_{Eff})$",
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
        eta_text = cl_plot_funcs.eta_text(eta_range)
        pt_text = cl_plot_funcs.pt_text(pt_cut)
        radius_text = cl_plot_funcs.radius_text(coef)

        hist_title = r"${{p_T}^{Cl}}/{{p_T}^{Gen}} \: Distribution \: $" + r"$({})$".format(pileup_key)
        hist_path = "plots/png/pTNormDist_{}_eta_{}_{}_ptGtr_{}_log_{}_final.png".format(pileup_key, cl_helpers.annot_str(eta_range[0]), cl_helpers.annot_str(eta_range[1]), pt_cut, particle)

        dist_args["eta_text"] = eta_text
        dist_args["pt_text"] = pt_text
        dist_args["radius_text"] = radius_text

        hist_cms = cl_plot_funcs.cmsPlot(hist_title, hist_path, **dist_args)

        hist_cms.write_fig()

        res_title = r"$Resolution \: vs. \: r^{Cl}$" + r"$({})$".format(pileup_key)
        res_path = "plots/png/res_vs_radius_{}_eta_{}_{}_ptGtr_{}_{}_final.png".format(pileup_key, cl_helpers.annot_str(eta_range[0]), cl_helpers.annot_str(eta_range[1]), pt_cut, particle)

        res_args["eta_text"] = eta_text
        res_args["pt_text"] = pt_text
        res_args["radius_text"] = radius_text

        res_cms = cl_plot_funcs.cmsPlot(res_title, res_path, **res_args)

        res_cms.write_fig()

    return fig, stat_table

if __name__ == "__main__":
    cl_plot_funcs.update_particle_callback()
    cl_plot_funcs.update_pileup_callback()