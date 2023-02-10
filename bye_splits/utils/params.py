# coding: utf-8

_all_ = [ ]

import os
from pathlib import Path
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd

user = 'ehle'
NbinsRz = 42
NbinsPhi = 216
MinROverZ = 0.076
MaxROverZ = 0.58
MinPhi = -np.pi
MaxPhi = +np.pi
PileUp = "PU0"
local = False
if local:   
    base_dir = "/grid_mnt/vol_home/llr/cms/ehle/Repos/bye_splits_final/"
else:
    base_dir = "/eos/user/i/iehle/"
DataFolder = 'data/{}/'.format(PileUp)
LocalDataFolder = 'data/new_algos'
EOSDataFolder = '/eos/user/b/bfontana/FPGAs/new_algos/'

viz_kw = {
    'DataPath': Path(EOSDataFolder),
    'OutPath': Path(EOSDataFolder),
    'LocalPath': Path(__file__).parents[2] / LocalDataFolder,
    'CfgProdPath': Path(__file__).parents[2] / 'bye_splits/production/prod_params.yaml',
    'CfgDataPath': Path(__file__).parents[2] / 'bye_splits/data_handle/config.yaml',
}

base_kw = {
    'NbinsRz': NbinsRz,
    'NbinsPhi': NbinsPhi,
    'MinROverZ': MinROverZ,
    'MaxROverZ': MaxROverZ,
    'MinPhi': MinPhi,
    'MaxPhi': MaxPhi,
    'RzBinEdges': np.linspace( MinROverZ, MaxROverZ, num=NbinsRz+1 ),
    'PhiBinEdges': np.linspace( MinPhi, MaxPhi, num=NbinsPhi+1 ),

    'LayerEdges': [0,42],
    'IsHCAL': False,

    'DataFolder': Path(f"{base_dir}{DataFolder}"),
    'FesAlgos': ['FloatingpointMixedbcstcrealsig4DummyHistomaxxydr015GenmatchGenclustersntuple/HGCalTriggerNtuple'],
    'BasePath': "{}{}".format(base_dir, DataFolder),
    'OutPath': "{}out".format(base_dir),

    'RzBinEdges': np.linspace( MinROverZ, MaxROverZ, num=NbinsRz+1 ),
    'PhiBinEdges': np.linspace( MinPhi, MaxPhi, num=NbinsPhi+1 ),

    'Placeholder': np.nan,
}

if user=='ehle':
    local = False
    pile_up = "PU0"
    particle = "photon"
    if local:
        base_dir = "/grid_mnt/vol_home/llr/cms/ehle/git/bye_splits_final/"
    else:
        base_dir = "/eos/user/i/iehle/"
    base_kw['DataFolder'] = f"data/{pile_up}/{particle}"
    base_kw['FesAlgos'] = ['FloatingpointMixedbcstcrealsig4DummyHistomaxxydr015GenmatchGenclustersntuple/HGCalTriggerNtuple']
    base_kw['BasePath'] = f"{base_dir}{base_kw['DataFolder']}"
input_files = {'photons':[], 'pions': []}
input_directory = base_kw['BasePath']
for filename in os.listdir(input_directory):
    if filename.startswith('energy_out'):
        path = os.path.join(input_directory, filename)
        with pd.HDFStore(path, "r") as File:
            if len(File.keys())>0:
                if 'photon' in filename:
                    input_files['photons'].append(path)
                else:
                    input_files['pions'].append(path)

def set_dictionary(adict):
    adict.update(base_kw)
    return adict
    
if len(base_kw['FesAlgos'])!=1:
    raise ValueError('The event number in the cluster task'
                     ' assumes there is only on algo.\n'
                     'The script must be adapted.')

# fill task
fill_kw = set_dictionary(
    {'FillIn'      : None, # Set per file, see common.FileDict
     'FillOut'     : 'fill',
     'FillOutComp' : 'fill_comp',
     'FillOutPlot' : 'fill_plot' }
     )

# optimization task
opt_kw = set_dictionary(
    { 'Epochs': 99999,
      'KernelSize': 10,
      'WindowSize': 3,
      'OptIn': 'triggergeom_condensed',
      'OptEnResOut': 'opt_enres',
      'OptPosResOut': 'opt_posres',
      'OptCSVOut': 'stats',
      'FillOutPlot': fill_kw['FillOutPlot'],
      'Pretrained': False,
    }
)

# smooth task
smooth_kw = set_dictionary(
    { #copied from L1Trigger/L1THGCal/python/hgcalBackEndLayer2Producer_cfi.py
        'BinSums': (13,               # 0
                    11, 11, 11,       # 1 - 3
                    9, 9, 9,          # 4 - 6
                    7, 7, 7, 7, 7, 7,  # 7 - 12
                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  # 13 - 27
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3  # 28 - 41
                    ),
        'SeedsNormByArea': False,
        'AreaPerTriggerCell': 4.91E-05,
        'SmoothIn': fill_kw['FillOut'],
        'SmoothOut': 'smooth' }
    )

# seed task
seed_kw = set_dictionary(
    { 'SeedIn': smooth_kw['SmoothOut'],
      'SeedOut': 'seed',
      'histoThreshold': 20.,
      'WindowPhiDim': 1}
    )

# cluster task
cluster_kw = set_dictionary(
    { 'ClusterInTC': fill_kw['FillOut'],
      'ClusterInSeeds': seed_kw['SeedOut'],
      'ClusterOutPlot': 'cluster_validation',
      'ClusterOutValidation': 'cluster_plot',
      'CoeffA': ( (0.015,)*7 + (0.020,)*7 + (0.030,)*7 + (0.040,)*7 + #EM
                  (0.040,)*6 + (0.050,)*6 + # FH
                  (0.050,)*12 ), # BH
      'CoeffB': 0,
      'MidRadius': 2.3,
      'PtC3dThreshold': 0.5,
      'ForEnergy': False,
      'EnergyOut': 'cluster_energy',
      'GenPart': fill_kw['FillIn']}
)

# validation task
validation_kw = set_dictionary(
    { 'ClusterOutValidation': cluster_kw['ClusterOutValidation'],
      'FillOutComp' : fill_kw['FillOutComp'],
      'FillOut': fill_kw['FillOut'] }
)

# energy task
energy_kw = set_dictionary(
    { 'ClusterIn': cluster_kw['ClusterOutValidation'],
      'Coeff': cluster_kw['CoeffA'],
      'ReInit': False,
      'Coeffs': (0.0, 0.05, 50), #tuple containing (coeff_start, coeff_end, num_coeffs)
      'EnergyIn': cluster_kw['EnergyOut'],
      'EnergyOut': 'energy_out',
      'BestMatch': True,
      'MatchFile': False,
      'MakePlot': True}
)

disconnectedTriggerLayers = [
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28
]
