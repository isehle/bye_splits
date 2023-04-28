# coding: utf-8
"""
Plots TC distributions and particular events, before and after:
- smoothing
- trigger cell movement
The full TC distributions show the distributions *before* TC movement.

There is some code duplication with `_full/_f` and `_sel/_s` suffixes
to describe the full phase space and the one defined ("selected") by `pars["region"]`
"""

_all_ = [ ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

from bye_splits.utils import params, common, parsing

import argparse
import numpy as np
import pandas as pd
import uproot as up
import random
import h5py
import yaml

from bye_splits.data_handle import data_process
from bye_splits.data_handle.geometry import GeometryData

from bokeh.io import output_file, show, save
from bokeh.layouts import layout
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource,
                          LogColorMapper, LogTicker,
                          LinearColorMapper, BasicTicker,
                          PrintfTickFormatter,
                          Range1d,
                          TabPanel, Tabs)
from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.palettes import viridis as _palette

colors = ('orange', 'red', 'black')
def set_figure_props(p, hide_legend=True):
    """set figure properties"""
    p.axis.axis_line_color = 'black'
    p.axis.major_tick_line_color = 'black'
    p.axis.major_label_text_font_size = '10px'
    p.axis.major_label_standoff = 2
    p.xaxis.axis_label = r"$$\color{black} \phi$$"
    p.yaxis.axis_label = '$$R/z$$'
    if hide_legend:
        p.legend.click_policy='hide'
    
def plot_trigger_cells_occupancy(pars, **cfg):
    particles = pars.particles
    pileup_dir = "PU0" if not pars.pileup else "PU200"
    particle_dir = "{}/{}/{}".format(params.LocalStorage, pileup_dir, particles)
    filename_template = lambda name: "{}_{}_{}".format(particles, pileup_dir, name)

    #_, _, tcData = data_process.get_data_reco_chain_start(nevents=pars.nevents, reprocess=False)
    _, _, tcData = data_process.get_data_reco_chain_start(nevents=-1, reprocess=False)

    rz_bins = np.linspace(cfg["base"]["MinROverZ"], cfg["base"]["MaxROverZ"], cfg["base"]["NbinsRz"])
    phi_bins = np.linspace(cfg["base"]["MinPhi"], cfg["base"]["MaxPhi"], cfg["base"]["NbinsPhi"])
    
    # Requires regular binning
    binDistRz = rz_bins[1] - rz_bins[0]
    binDistPhi = phi_bins[1] - phi_bins[0]
    binConv = lambda vals,dist,amin : (vals*dist) + (dist/2) + amin

    SHIFTH, SHIFTV = 3*binDistPhi, binDistRz

    simDataPath = cfg["plot"][pileup_dir][particles]["files"][0]
    _, simAlgoFiles, simAlgoPlots = ({} for _ in range(3))

    fe = cfg["base"]["FesAlgo"]
    simAlgoFiles[fe] = [ os.path.join(simDataPath) ]

    title_common = r"{} vs {} bins".format(cfg["base"]["NbinsPhi"], cfg["base"]["NbinsRz"])
    if pars.pos_endcap:
        title_common += '; Positive end-cap only'
    title_common += '; Min(R/z)={} and Max(R/z)={}'.format(cfg["base"]['MinROverZ'],
                                                           cfg["base"]['MaxROverZ'])

    mypalette = _palette(50)

    # Inputs: Trigger Cells
    tcVariables = {"tc_layer", "tc_phi", "tc_eta", "tc_x", "tc_y", "tc_z", "tc_multicluster_id"}
    assert(tcVariables.issubset(tcData.keys()))

    # Inputs: Cluster After Custom Iterative Algorithm
    cluster_dir = "{}/cluster/".format(particle_dir)
    if not pars.pileup:
        file_name = filename_template(cfg["plot"]["baseName"])
        file_name += "_coef_{}".format(str(cfg["plot"]["radius"]).replace(".","p"))
    else:
        file_name = "{}_coef_{}".format(cfg["plot"]["baseName"],str(cfg["plot"]["radius"]).replace(".","p"))
    
    outclusterplot = common.fill_path(file_name, data_dir=cluster_dir, **pars)
    with pd.HDFStore(outclusterplot, mode='r') as store:
        splittedClusters_3d_local = store['data']

    tcData= common.tc_base_selection(tcData,
                                     range_rz=(cfg["base"]['MinROverZ'],
                                              cfg["base"]['MaxROverZ']))

    tcData_full = tcData[:]
    #tcData_sel = tcData[subdetCond]

    copt = dict(labels=False)
    with common.SupressSettingWithCopyWarning():
        tcData_full['Rz_bin'] = pd.cut(tcData_full['Rz'],
                                    bins=rz_bins,
                                    **copt)
        tcData_full['phi_bin'] = pd.cut(tcData_full['tc_phi'],
                                        bins=phi_bins,
                                        **copt)

        '''tcData_sel['Rz_bin'] = pd.cut(tcData_sel['Rz'],
                                    bins=rz_bins,
                                    **copt)
        tcData_sel['phi_bin'] = pd.cut(tcData_sel['phi'],
                                    bins=phi_bins,
                                    **copt)'''
        
        # Convert bin ids back to values (central values in each bin)
        tcData_full['Rz_center'] = binConv(tcData_full.Rz_bin, binDistRz, cfg["base"]['MinROverZ'])
        tcData_full['phi_center'] = binConv(tcData_full.phi_bin, binDistPhi, cfg["base"]['MinPhi'])
        '''tcData_sel['Rz_center'] = binConv(tcData_sel.Rz_bin, binDistRz, cfg["base"]['MinROverZ'])
        tcData_sel['phi_center'] = binConv(tcData_sel.phi_bin, binDistPhi, cfg["base"]['MinPhi'])'''

        _cols_drop = ['Rz_bin', 'phi_bin', 'Rz', 'tc_phi']
        tcData_full = tcData_full.drop(_cols_drop, axis=1)
        #tcData_sel = tcData_sel.drop(_cols_drop, axis=1)

    # if `-1` is included in pars["ledges"], the full selection is also drawn
    try:
        pars.ledges.remove(-1)
        leftLayerEdges, rightLayerEdges = pars.ledges[:-1], pars.ledges[1:]
        leftLayerEdges.insert(0, 0)
        rightLayerEdges.insert(0, tcData_full.layer.max())
    except ValueError:
        leftLayerEdges, rightLayerEdges = pars.ledges[:-1], pars.ledges[1:]

    ledgeszip = tuple(zip(leftLayerEdges,rightLayerEdges))
    tcSelections = ['layer>{}, layer<={}'.format(x,y) for x,y in ledgeszip]
    grps_f, grps_s = ([] for _ in range(2))
    for lmin,lmax in ledgeszip:
        #full
        grps_f.append( tcData_full[ (tcData_full.tc_layer>lmin) &
                                    (tcData_full.tc_layer<=lmax) ] )
        groupby_full = grps_f[-1].groupby(['Rz_center', 'phi_center'],
                                          as_index=False)
        grps_f[-1] = groupby_full.count()
        eta_mins = groupby_full.min()['tc_eta']
        eta_maxs = groupby_full.max()['tc_eta']
        grps_f[-1].insert(0, 'min_eta', eta_mins)
        grps_f[-1].insert(0, 'max_eta', eta_maxs)
        grps_f[-1] = grps_f[-1].rename(columns={'tc_z': 'ntc'})
        _cols_keep = ['phi_center', 'ntc', 'Rz_center',
                      'min_eta', 'max_eta']
        grps_f[-1] = grps_f[-1][_cols_keep]

        #sel
        '''grps_s.append( tcData_sel[ (tcData_sel.layer>lmin) &
                                    (tcData_sel.layer<=lmax) ] )
        groupby_sel = grps_s[-1].groupby(['Rz_center', 'phi_center'],
                                          as_index=False)
        grps_s[-1] = groupby_sel.count()
        eta_mins = groupby_sel.min()['eta']
        eta_maxs = groupby_sel.max()['eta']
        grps_s[-1].insert(0, 'min_eta', eta_mins)
        grps_s[-1].insert(0, 'max_eta', eta_maxs)
        grps_s[-1] = grps_s[-1].rename(columns={'z': 'ntc'})
        _cols_keep = ['phi_center', 'ntc', 'Rz_center',
                      'min_eta', 'max_eta']
        grps_s[-1] = grps_s[-1][_cols_keep]'''
    

    #########################################################################
    ################### DATA ANALYSIS: SIMULATION ###########################
    #########################################################################

    #fillplot_template = "{}_{}_{}".format(particles, pileup_dir, cfg["fill"]["FillOutPlot"])
    fillplot_template = filename_template(cfg["fill"]["FillOutPlot"])
    outfillplot = common.fill_path(fillplot_template, data_dir=particle_dir, **pars)
    with pd.HDFStore(outfillplot, mode='r') as store:
        splittedClusters_3d_cmssw = store[fe + '_3d']
        splittedClusters_tc = store[fe + '_tc']

    simAlgoPlots[fe] = (splittedClusters_3d_cmssw,
                        splittedClusters_tc,
                        splittedClusters_3d_local )

    #########################################################################
    ################### PLOTTING: TRIGGER CELLS #############################
    #########################################################################
    bckg_full, bckg_sel = ([] for _ in range(2))
    for idx,(grp_full,grp_sel) in enumerate(zip(grps_f,grps_s)):
        source_full = ColumnDataSource(grp_full)
        #source_sel  = ColumnDataSource(grp_sel)

        mapper_class = LogColorMapper if pars.log_scale else LinearColorMapper
        mapper = mapper_class(palette=mypalette,
                              low=grp_full['ntc'].min(),
                              high=grp_full['ntc'].max())

        title = title_common + '; {}'.format(tcSelections[idx])
        '''fig_opt = dict(title=title,
                       width=1800,
                       height=600,
                       x_range=Range1d(tcData_full.phi_center.min()-SHIFTH,
                                       tcData_full.phi_center.max()+SHIFTH),
                       y_range=Range1d(tcData_full.Rz_center.min()-SHIFTV,
                                       tcData_full.Rz_center.max().max()+SHIFTV),
                       tools="hover,box_select,box_zoom,reset,save",
                       x_axis_location='below',
                       x_axis_type='linear',
                       y_axis_type='linear')'''
        
        fig_opt = dict(title=title,
                       width=1800,
                       height=600,
                       tools="hover,box_select,box_zoom,reset,save",
                       x_axis_location='below',
                       x_axis_type='linear',
                       y_axis_type='linear')

        p_full = figure(**fig_opt)
        p_full.output_backend = 'svg'
        p_full.toolbar.logo = None
        '''p_sel  = figure(**fig_opt)
        p_sel.output_backend = 'svg'
        p_sel.toolbar.logo = None'''

        rect_opt = dict(x='phi_center', y='Rz_center',
                        width=binDistPhi, height=binDistRz,
                        width_units='data', height_units='data',
                        line_color='black',
                        fill_color=transform('ntc', mapper))
        p_full.rect(source=source_full, **rect_opt)
        #p_sel.rect(source=source_sel, **rect_opt)

        ticker = ( LogTicker(desired_num_ticks=len(mypalette))
                   if pars.log_scale
                   else BasicTicker(desired_num_ticks=int(len(mypalette)/4)) )
        color_bar = ColorBar(color_mapper=mapper,
                             title='#Hits',
                             ticker=ticker,
                             formatter=PrintfTickFormatter(format="%d"))
        p_full.add_layout(color_bar, 'right')
        #p_sel.add_layout(color_bar, 'right')

        set_figure_props(p_full, hide_legend=False)
        #set_figure_props(p_sel, hide_legend=False)        

        tooltips = [ ("#TriggerCells", "@{ntc}"),
                     ("min(eta)", "@{min_eta}"),
                     ("max(eta)", "@{max_eta}") ]
        p_full.hover.tooltips = tooltips
        #p_sel.hover.tooltips = tooltips

        bckg_full.append( p_full )
        #bckg_sel.append( p_sel )

    #########################################################################
    ################### PLOTTING: SIMULATION ################################
    #########################################################################
    # Already true
    #assert len(kw['FesAlgos'])==1
    ev_panels = [] #pics = []

    for _k,(df_3d_cmssw,df_tc,df_3d_local) in simAlgoPlots.items():

        if pars.nevents > len(df_tc['event'].unique()):
            nev_data = len(df_tc['event'].unique())
            m = ( 'You are trying to plot more events ({}) than '.format(nev_data) +
                  'those available in the dataset ({}).'.format(pars.nevents) )
            raise ValueError(m)
        
        event_list = cfg["plot"][pileup_dir][particles]["events"]
        if len(event_list) == 0:
            event_sample = ( random.sample(df_tc["event"].unique().astype("int").tolist(),pars.nevents) )
        else:
            #event_sample = ( df_tc["event"].unique().astype("int").tolist() )
            event_sample = event_list
        for ev in event_sample:
            # Inputs: Energy 2D histogram after smoothing but before clustering

            outsmooth_template = filename_template(cfg["smooth"]["SmoothOut"])
            outsmooth = common.fill_path(outsmooth_template, data_dir=particle_dir, **pars)
            with h5py.File(outsmooth, mode='r') as storeSmoothIn:
                k = fe+'_'+str(ev)+'_group'
                try:
                    energies_post_smooth, _, _ = storeSmoothIn[k]
                except KeyError:
                    continue

            # convert 2D numpy array to (rz_bin, phi_bin) pandas dataframe
            df_smooth = ( pd.DataFrame(energies_post_smooth)
                              .reset_index()
                              .rename(columns={'index': 'Rz_bin'}) )
            df_smooth = ( pd.melt(df_smooth,
                                      id_vars='Rz_bin',
                                      value_vars=[x for x in range(0,216)])
                             .rename(columns={'variable': 'phi_bin', 'value': 'energy_post_smooth'}) )
            df_smooth['Rz_center']  = binConv(df_smooth.Rz_bin,  binDistRz,  cfg["base"]['MinROverZ'])
            df_smooth['phi_center'] = binConv(df_smooth.phi_bin, binDistPhi, cfg["base"]['MinPhi'])

            # do not display empty (or almost empty) bins
            df_smooth = df_smooth[ df_smooth.energy_post_smooth > 0.1 ]
            
            
            tools = "hover,box_select,box_zoom,reset,save"
            '''fig_opt = dict(width=900,
                           height=300,
                           x_range=Range1d(cfg["base"]['MinPhi']-2*SHIFTH,
                                           cfg["base"]['MaxPhi']+2*SHIFTH),
                           y_range=Range1d(cfg["base"]['MinROverZ']-SHIFTV,
                                           cfg["base"]['MaxROverZ']+SHIFTV),
                           tools=tools,
                           x_axis_location='below',
                           x_axis_type='linear',
                           y_axis_type='linear')'''
            
            fig_opt = dict(width=900,
                           height=300,
                           tools=tools,
                           x_axis_location='below',
                           x_axis_type='linear',
                           y_axis_type='linear')

            ev_tc       = df_tc[ df_tc.event == ev ]
            ev_3d_cmssw = df_3d_cmssw[ df_3d_cmssw.event == ev ]
            ev_3d_local = df_3d_local[ df_3d_local.event == ev ]

            tc_cols = [ 'tc_mipPt', 'tc_z', 'tc_multicluster_id',
                        'tc_eta',
                        'tc_phi',
                        'Rz', 'Rz_bin', 'tc_phi_bin',
                      ]
            ev_tc = ev_tc.filter(items=tc_cols)

            copt = dict(labels=False)

            # Convert bin ids back to values (central values in each bin)
            ev_tc['Rz_center'] = binConv(ev_tc.Rz_bin, binDistRz, cfg["base"]['MinROverZ'])
            ev_tc['phi_center'] = binConv(ev_tc.tc_phi_bin, binDistPhi,
                                              cfg["base"]['MinPhi'])


            _cols_drop = ['Rz_bin', 'tc_phi_bin', 'Rz']
            ev_tc = ev_tc.drop(_cols_drop, axis=1)

            with common.SupressSettingWithCopyWarning():
                ev_3d_cmssw['cl3d_Roverz']=common.calcRzFromEta(ev_3d_cmssw.cl3d_eta)
                ev_3d_cmssw['gen_Roverz']=common.calcRzFromEta(ev_3d_cmssw.gen_eta)

            cl3d_pos_rz  = ev_3d_cmssw['cl3d_Roverz'].unique()
            cl3d_pos_phi = ev_3d_cmssw['cl3d_phi'].unique()
            gen_pos_rz   = ev_3d_cmssw['gen_Roverz'].unique()
            gen_pos_phi  = ev_3d_cmssw['gen_phi'].unique()
            drop_cols = ['cl3d_Roverz', 'cl3d_eta', 'cl3d_phi']
            ev_3d_cmssw = ev_3d_cmssw.drop(drop_cols, axis=1)
            assert( len(gen_pos_rz) == 1 and len(gen_pos_phi) == 1 )

            groupby = ev_tc.groupby(['Rz_center',
                                         'phi_center'],
                                        as_index=False)
            group = groupby.count()

            _ensum = groupby.sum()['tc_mipPt']
            _etamins = groupby.min()['tc_eta']
            _etamaxs = groupby.max()['tc_eta']

            group = group.rename(columns={'tc_z': 'nhits'})
            group.insert(0, 'min_eta', _etamins)
            group.insert(0, 'max_eta', _etamaxs)
            group.insert(0, 'sum_en', _ensum)

            mapper_class = LogColorMapper if pars.log_scale else LinearColorMapper
            ticker = ( LogTicker(desired_num_ticks=len(mypalette))
                      if pars.log_scale
                      else BasicTicker(desired_num_ticks=int(len(mypalette)/4)) )
            base_bar_opt = dict(ticker=ticker,
                                formatter=PrintfTickFormatter(format="%d"))

            rect_opt = dict( y='Rz_center',
                             width=binDistPhi, height=binDistRz,
                             width_units='data', height_units='data',
                             line_color='black' )

            seed_window = ( 'phi seeding step window: {}'
                           .format(cfg["optimization"]["WindowSize"]) )
            figs = []
            t_d = {0: ( 'Energy Density (before smoothing step, ' +
                        'before algo, {})'.format(seed_window) ),
                   1: ( 'Energy Density (after smoothing step, ' +
                        'before algo, {})'.format(seed_window) ),
                   2: ( 'Hit Density (before smoothing step, ' +
                        'before algo, {})'.format(seed_window) ), }
            group_d = {0: group,
                       1: df_smooth,
                       2: group,}
            hvar_d = {0: 'sum_en',
                      1: 'energy_post_smooth',
                      2: 'nhits',}
            bvar_d = {0: 'Energy [mipPt]',
                      1: 'Energy [mipPt]',
                      2: '#Hits',}
            toolvar_d = {0: ('EnSum', '@{sum_en}'),
                         1: ('EnSum', '@{energy_post_smooth}'),
                         2: ('#hits', '@{nhits}'),}
            rec_opt_d = {0: dict(x='phi_center',
                                 source=ColumnDataSource(group),
                                 **rect_opt),
                         1: dict(x='phi_center',
                                 source=ColumnDataSource(df_smooth),
                                 **rect_opt),
                         2: dict(x='phi_center',
                                 source=ColumnDataSource(group),
                                 **rect_opt),}

            base_cross_opt = dict(size=25, angle=np.pi/4, line_width=4)
            gen_label = 'Gen Particle Position'
            gen_cross_opt = dict(x=gen_pos_phi, y=gen_pos_rz,
                                 color=colors[0],
                                 legend_label=gen_label,
                                 **base_cross_opt)
            cmssw_label = 'CMSSW Cluster Position'
            cmssw_cross_opt = dict(x=cl3d_pos_phi, y=cl3d_pos_rz,
                                   color=colors[1],
                                   legend_label=cmssw_label,
                                   **base_cross_opt)
            local_label = 'Custom Cluster Position'
            local_cross_opt = dict(x=ev_3d_local.phi,
                                   y=ev_3d_local.Rz,
                                   color=colors[2],
                                   legend_label=local_label,
                                    **base_cross_opt)

            for it in range(len(t_d.keys())):
                figs.append( figure(title=t_d[it], **fig_opt) )
                figs[-1].output_backend = 'svg'
                figs[-1].toolbar.logo = None

                map_opt = dict( low= group_d[it][ hvar_d[it] ].min(),
                                high=group_d[it][ hvar_d[it] ].max() )
                mapper = mapper_class(palette=mypalette, **map_opt)

                bar_opt = dict(title=bvar_d[it], **base_bar_opt)
                bar = ColorBar(color_mapper=mapper, **bar_opt)
                figs[-1].add_layout(bar, 'right')

                figs[-1].rect(fill_color=transform(hvar_d[it], mapper),
                              **rec_opt_d[it] )

                figs[-1].hover.tooltips = [ toolvar_d[it] ]
                figs[-1].cross(**gen_cross_opt)
                figs[-1].cross(**local_cross_opt)
                figs[-1].cross(**cmssw_cross_opt)

                set_figure_props(figs[-1])

            cross1_opt = dict(x=gen_pos_phi, y=gen_pos_rz,
                              color=colors[0], **base_cross_opt)
            cross2_opt = dict(x=cl3d_pos_phi, y=cl3d_pos_rz,
                              color=colors[1], **base_cross_opt)
            for bkg1,bkg2 in zip(bckg_full,bckg_sel):
                bkg1.cross(**cross1_opt)
                bkg1.cross(**cross2_opt)
                bkg2.cross(**cross1_opt)
                bkg2.cross(**cross2_opt)

            #pics.append( (p,ev) )
            _lay = layout( [[figs[2]], [figs[0]], [figs[1]]] )
            ev_panels.append( TabPanel(child=_lay,
                                    title='{}'.format(ev)) )

    plot_template_name = filename_template(cfg["plot"]["plotName"])
    plot_path = common.fill_path(plot_template_name, ext="html", **pars)

    output_file(plot_path)

    tc_panels_full, tc_panels_sel = ([] for _ in range(2))
    for i,(bkg1,bkg2) in enumerate(zip(bckg_full,bckg_sel)):
        tc_panels_full.append( TabPanel(child=bkg1,
                                     title='Full | Selection {}'.format(i)) )
        tc_panels_sel.append( TabPanel(child=bkg2,
                                    title='Region {} | Selection {}'.format(pars.reg,i)) )

    lay = layout([[Tabs(tabs=ev_panels)],
                  [Tabs(tabs=tc_panels_sel)],
                  [Tabs(tabs=tc_panels_full)]])
    
    show(lay) if pars.show_html else save(lay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot trigger cells occupancy.')
    parser.add_argument('--ledges', help='layer edges (if -1 is added the full range is also included)', default=[0,42], nargs='+', type=int)
    parser.add_argument('--pos_endcap', help='Use only the positive endcap.',
                        default=True, type=bool)
    parser.add_argument('--show_html', help="Display plot instead of saving", default=False, type=bool)
    parser.add_argument('--hcal', help='Consider HCAL instead of default ECAL.', action='store_true')
    parser.add_argument('-n', '--nevents', help='number of events to process', type=int, default=5)
    parser.add_argument('--particles', choices=("photons", "electrons", "pions"), default="photons")
    parser.add_argument('--pileup', help='plot PU200 instead of PU0', action='store_true')
    parser.add_argument('-l', '--log', help='use color log scale', action='store_true')
    parser.add_argument('--show_html', type=bool, default=False)
    parsing.add_parameters(parser)

    FLAGS = parser.parse_args()
    pars = common.dot_dict(vars(FLAGS))

    with open(params.CfgPath, "r") as file:
        cfg = yaml.safe_load(file)

    plot_trigger_cells_occupancy(pars, **cfg)

