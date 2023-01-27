import os
import sys
from dash import dcc, html, Input, Output, callback
import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import re
import pandas as pd

#parent_dir = os.path.abspath(__file__ + 4 * '/..')
parent_dir = '/home/llr/cms/ehle/git/bye_splits_final/'

sys.path.insert(0, parent_dir)

import argparse
from bye_splits.utils import params, parsing, common

# Set up to work with the rest of the files in the pipeline
parser = argparse.ArgumentParser(description='Clustering standalone step.')
parsing.add_parameters(parser)
FLAGS = parser.parse_args()
assert FLAGS.sel in ('splits_only',) or FLAGS.sel.startswith('above_eta_') or FLAGS.sel.startswith('below_eta_')

FLAGS.reg = 'All'
FLAGS.sel = 'below_eta_2.7'

input_files = params.fill_kw['FillInFiles']

# Find the element of a list containing strings ['coef_{float_1}', 'coef_{float_2}', ...] which is closest to some float_i
def_k = 0.0
def closest(list, k=def_k):
    try:
        list = np.reshape(np.asarray(list), 1)
    except:
        list = np.asarray(list)
    if isinstance(k, str):
        k_num = float(re.split('coef_',k)[1].replace('p','.'))
    else:
        k_num = k
    id = (np.abs(list-k_num)).argmin()
    return list[id]

def effrms(data, c=0.68):
    """Compute half-width of the shortest interval
    containing a fraction 'c' of items in a 1D array.
    """
    out = {}
    x = np.sort(data, kind='mergesort')
    m = int(c *len(x)) + 1
    out = [np.min(x[m:] - x[:-m]) / 2.0]

    return out

def get_str(coef, file):
    coef_str = 'coef_{}'.format(str(coef).replace('.','p'))
    if coef_str not in file.keys():
        coef_list = [float(re.split('coef_',key)[1].replace('p','.')) for key in file.keys()]
        new_coef = closest(coef_list, coef)
        coef_str = 'coef_{}'.format(str(new_coef).replace('.','p'))
    return coef_str

def get_dfs(init_files, coef, pars):

    file_dict = {}
    start = params.energy_kw['EnergyOut']
    df_dict = dict.fromkeys(init_files.keys(),[0.0])

    for key in init_files.keys():
        file_dict[key] = [start+re.split('gen_cl3d_tc',file)[1] for file in init_files[key]]
        file_dict[key] = [common.fill_path(file,**pars) for file in file_dict[key]]
        if len(file_dict[key])==1:

            File = pd.HDFStore(file_dict[key][0],'r')
            
            coef_str = get_str(coef, File)
            
            df = File[coef_str]
        else:
            file_list = [pd.HDFStore(val,'r') for val in file_dict[key]]
            coef_str = get_str(coef, file_list[0])
            df_list = [file_list[i][coef_str] for i in range(len(file_list))]
            df = pd.concat(df_list)
        df_dict[key] = df
    
    return df_dict

# Dash page setup
##############################################################################################################################

marks = {coef : {"label" : format(coef,'.3f'), "style": {"transform": "rotate(-90deg)"}} for coef in np.arange(0.0,0.05,0.001)}

dash.register_page(__name__, title='RMS', name='RMS')

layout = dbc.Container([
    dbc.Row([html.Div("Interactive Normal Distribution", style={'fontSize': 40, 'textAlign': 'center'})]),

    html.Hr(),

    dcc.Graph(id="histograms-x-graph",mathjax=True),
    html.P("Coef:"),
    dcc.Slider(id="coef", min=0.0, max=0.05, value=0.001,marks=marks),
    html.P("EtaRange:"),
    dcc.RangeSlider(id='eta_range',min=1.4,max=2.7,step=0.1,value=[1.4,2.7]),

    html.Hr(),

    dbc.Row([
        dcc.Markdown(r'Gaussianity := $\frac{|RMS-RMS_{Eff}|}{RMS}$', mathjax=True, style={'fontSize': 30, 'textAlign': 'center'})
    ]),

    html.Hr(),

    html.Div(id='my_table'),
])

@callback(
    Output("histograms-x-graph", "figure"),
    Output("my_table", "children"),
    Input("coef", "value"),
    Input("eta_range", "value"))

##############################################################################################################################

def display_color(coef, eta_range):
    dfs_by_particle = get_dfs(input_files, coef, vars(FLAGS))
    phot_df = dfs_by_particle['photon']
    pion_df = dfs_by_particle['pion']

    phot_df['normed_energies'] = phot_df['en']/phot_df['genpart_energy']
    pion_df['normed_energies'] = pion_df['en']/pion_df['genpart_energy']

    phot_df = phot_df[ phot_df['genpart_exeta'] > eta_range[0] ]
    pion_df = pion_df[ pion_df['genpart_exeta'] > eta_range[0] ]

    phot_df = phot_df[ phot_df['genpart_exeta'] < eta_range[1] ]
    pion_df = pion_df[ pion_df['genpart_exeta'] < eta_range[1] ]

    phot_mean_en = phot_df['normed_energies'].mean()
    pion_mean_en = pion_df['normed_energies'].mean()

    phot_rms = phot_df['normed_energies'].std()/phot_mean_en
    phot_eff_rms = effrms(phot_df['normed_energies'])/phot_mean_en
    phot_gaus_diff = np.abs(phot_eff_rms-phot_rms)/phot_rms

    pion_rms = pion_df['normed_energies'].std()/pion_mean_en
    pion_eff_rms = effrms(pion_df['normed_energies'])/pion_mean_en
    pion_gaus_diff = np.abs(pion_eff_rms-pion_rms)/pion_rms

    pion_gaus_str = format(pion_gaus_diff[0], '.3f')
    phot_gaus_str = format(phot_gaus_diff[0], '.3f')

    pion_rms_str = format(pion_rms, '.3f')
    phot_rms_str = format(phot_rms, '.3f')

    pion_eff_rms_str = format(pion_eff_rms[0], '.3f')
    phot_eff_rms_str = format(phot_eff_rms[0], '.3f')

    fig = go.Figure()

    fig.add_trace(go.Histogram(x=pion_df['normed_energies'], nbinsx=100, autobinx=False, name='Pion'))
    fig.add_trace(go.Histogram(x=phot_df['normed_energies'], nbinsx=100, autobinx=False, name='Photon'))

    fig.update_layout(barmode='overlay',title_text='Normalized Cluster Energy', xaxis_title=r'$\Huge{\frac{E_{Cl}}{E_{Gen}}}$', yaxis_title_text=r'$\Large{Events}$')

    fig.update_traces(opacity=0.5)

    my_vals = {
        'Photon': {
            'RMS': phot_rms_str,
            'Effective RMS': phot_eff_rms_str,
            'Gaussianity': phot_gaus_str,
        },
        'Pion': {
            'RMS': pion_rms_str,
            'Effective RMS': pion_eff_rms_str,
            'Gaussianity': pion_gaus_str,
        }
    }

    val_df = pd.DataFrame(my_vals).reset_index()
    val_df = val_df.rename(columns={'index': ''})

    return fig, dbc.Table.from_dataframe(val_df)
        