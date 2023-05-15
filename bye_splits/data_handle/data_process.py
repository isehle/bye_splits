# coding: utf-8

_all_ = ['baseline_selection', 'EventDataParticle']

import os
import sys

parent_dir = os.path.abspath(__file__ + 2 * "/..")
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import yaml

import bye_splits
from bye_splits.utils import common

from utils import params
from data_handle.geometry import GeometryData
from data_handle.event import EventData
from data_handle.data_input import InputData
from plotly.express.colors import sample_colorscale, qualitative

def baseline_selection(df_gen, df_cl, sel, **kw):
    data = pd.merge(left=df_gen, right=df_cl, how='inner', on='event')
    nin = data.shape[0]
    data = data[(data.gen_eta>kw['EtaMin']) & (data.gen_eta<kw['EtaMax'])]

    if sel.startswith('above_eta_'):
        data = data[data.gen_eta > float(sel.split('above_eta_')[1])]
        return data
    
    with common.SupressSettingWithCopyWarning():
        data['enres'] = data.cl3d_en - data.gen_en
        data.enres /= data.gen_en

    nansel = pd.isna(data['enres'])
    nandf = data[nansel]
    nandf['enres'] = 1.1
    data = data[~nansel]
    data = pd.concat([data,nandf], sort=False)
        
    if sel == 'splits_only':
        # select events with splitted clusters (enres < energy cut)
        # if an event has at least one cluster satisfying the enres condition,
        # all of its clusters are kept (this eases comparison with CMSSW)
        evgrp = data.groupby(['event'], sort=False)
        multiplicity = evgrp.size()
        bad_res = (evgrp.apply(lambda grp: np.any(grp['enres'] < kw['EnResSplits']))).values
        bad_res_mask = np.repeat(bad_res, multiplicity.values)
        data = data[bad_res_mask]
  
    elif sel == 'no_splits':
        data = data[(data.gen_eta > kw['EtaMinStrict']) &
                    (data.gen_eta < kw['EtaMaxStrict'])]
        evgrp = data.groupby(['event'], sort=False)
        multiplicity = evgrp.size()
        good_res = (evgrp.apply(lambda grp: np.all(grp['enres'] > kw['EnResNoSplits']))).values
        good_res_mask = np.repeat(good_res, multiplicity.values)
        data = data[good_res_mask]
        
    elif sel == 'all':
        pass
    
    else:
        m = 'Selection {} is not supported.'.format(sel)
        raise ValueError(m)

    nout = data.shape[0]
    eff = (nout / nin) * 100
    print("The baseline selection has a {}% efficiency: {}/{}".format(np.round(eff,2), nout, nin))
    return data

def get_data_reco_chain_start(nevents=500, reprocess=False, tag='chain', particles='photons', event=None):
    """Access event data."""
    data_part_opt = dict(tag=tag, reprocess=reprocess, debug=True)
    data_particle = EventDataParticle(particles=particles, **data_part_opt)
    if event == None:
        ds_all, events = data_particle.provide_random_events(n=nevents) #, seed=42)
        # ds_all = data_particle.provide_events(events=[170004, 170015, 170017, 170014])
    else:
        ds_all = data_particle.provide_event(event,merge=False)
        events = event

    if ds_all["gen"].empty:
        mes = "No events in the parquet file."
        raise RuntimeError(mes)

    tc_keep = {
        "event": "event",
        "good_tc_waferu": "tc_wu",
        "good_tc_waferv": "tc_wv",
        "good_tc_cellu": "tc_cu",
        "good_tc_cellv": "tc_cv",
        "good_tc_layer": "tc_layer",
        "good_tc_pt": "tc_pt",
        "good_tc_mipPt": "tc_mipPt",
        "good_tc_energy": "tc_energy",
        "good_tc_x": "tc_x",
        "good_tc_y": "tc_y",
        "good_tc_z": "tc_z",
        "good_tc_eta": "tc_eta",
        "good_tc_phi": "tc_phi",
        "good_tc_multicluster_id": "tc_multicluster_id",
    }

    ds_tc = ds_all["tc"]
    ds_tc = ds_tc[tc_keep.keys()]
    ds_tc = ds_tc.rename(columns=tc_keep)

    gen_keep = {
        "event": "event",
        "good_genpart_exeta":  "gen_eta",
        "good_genpart_exphi":  "gen_phi",
        "good_genpart_energy": "gen_en",
        "good_genpart_pt":     "gen_pt",
    }
    ds_gen = ds_all["gen"]
    ds_gen = ds_gen.rename(columns=gen_keep)

    cl_keep = {
        "event": "event",
        "good_cl3d_eta":    "cl3d_eta",
        "good_cl3d_phi":    "cl3d_phi",
        "good_cl3d_id":     "cl3d_id",
        "good_cl3d_energy": "cl3d_en",
        "good_cl3d_pt":     "cl3d_pt",
    }
    ds_cl = ds_all["cl"]
    ds_cl = ds_cl.rename(columns=cl_keep)
    return ds_gen, ds_cl, ds_tc

def colorscale(df, variable, scale, saturate=False):
    norm_points = (df[variable]-df[variable].min())/(df[variable].max()-df[variable].min())
    colorscale = sample_colorscale(scale, norm_points) 
    if saturate:
        ["rgb(255,255,255)" if norm_points[i] == 1 else s for i, s in enumerate(colorscale)]
    return colorscale

def get_event(filename, event, coefs):
    """   Load dataframes of a particular event from the hdf5 file   """
    dic = {}
    for coef in coefs:
        key = 'ev_'+str(event)+'/coef_'+str(coef)[2:]
        dic[coef] = pd.read_hdf(filename, key)    
    dic['gen'] = pd.read_hdf(filename, 'ev_'+str(event)+'/gen_info')
    return dic

def store_event(h5file, path, data):
    """   Save dictionary in a h5file. A dictionary corresponds to one event.
    The key is a selected radius, while the values is the corresponding dataframe  """

    if isinstance(data, dict):
        for key, item in data.items():
            if isinstance(item, pd.DataFrame):
                item.to_hdf(h5file, path + str(key))
            else:
                raise ValueError('Cannot save %s type'%type(item))
    else:
        data.to_hdf(h5file, path)

def process_and_store(dict_events, gen_info, ds_geom, h5file):

    tc_keep = {'seed_idx'  : 'seed_idx',
               'tc_mipPt'  : 'mipPt',
               'tc_eta'    : 'tc_eta',
               'tc_wu'     : 'waferu',
               'tc_wv'     : 'waferv',
               'tc_cu'     : 'triggercellu',
               'tc_cv'     : 'triggercellv',
               'tc_layer'  : 'layer'}

    sci_update = {'triggercellieta': 'triggercellu',
                  'triggercelliphi': 'triggercellv',
                  'tc_wu'          : 'waferu'}
    ds_geom['sci'] = ds_geom['sci'].rename(columns=sci_update)
    
    for event in dict_events.keys():
        print('Procesing event '+str(event))
        dict_event = dict_events[event]
        for coef in dict_event.keys():
            dict_event[coef] = dict_event[coef].rename(columns=tc_keep)
            silicon_df = pd.merge(left=dict_event[coef], right=ds_geom['si'], how='inner',
                                  on=['layer', 'waferu', 'waferv', 'triggercellu', 'triggercellv'])
            silicon_df = silicon_df.drop(['waferorient', 'waferpart'], axis=1)
            scintillator_df = pd.merge(left=dict_event[coef], right=ds_geom['sci'], how='inner',
                                       on=['layer', 'waferu', 'triggercellu', 'triggercellv'])

            dict_event[coef] = pd.concat([silicon_df, scintillator_df]).reset_index(drop=True)
            color_continuous = colorscale(dict_event[coef], 'mipPt', 'viridis')
            color_discrete   = colorscale(dict_event[coef], 'seed_idx', qualitative.Light24, True)
            dict_event[coef] = dict_event[coef].assign(color_energy=color_continuous, color_clusters=color_discrete)
            
        store_event(h5file, '/ev_'+str(event)+'/coef_', dict_event)
        store_event(h5file, '/ev_'+str(event)+'/gen_info', gen_info[gen_info.event == event])

            
def EventDataParticle(tag, reprocess, logger=None, debug=False, particles=None):
    """Factory for EventData instances of different particle types"""
    with open(params.CfgPath, "r") as afile:
        cfg = yaml.safe_load(afile)
        if particles is None:
            particles = cfg["selection"]["particles"]
        if particles not in ("photons", "photons_PU", "electrons", "pions"):
            raise ValueError("{} are not supported.".format(particles))
        defevents = cfg["defaultEvents"][particles]

        indata = InputData()
        indata.path = cfg["io"]["file" + particles]
        indata.adir = cfg["io"]["dir" + particles]
        indata.tree = cfg["io"]["tree" + particles]

    tag = particles + "_" + tag
    tag += "_debug" * debug


    return EventData(indata, tag, defevents, reprocess, logger)
