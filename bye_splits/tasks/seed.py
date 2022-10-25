# coding: utf-8

_all_ = [ 'seed' ]

import os
import sys
parent_dir = os.path.abspath(__file__ + 3 * '/..')
sys.path.insert(0, parent_dir)

import bye_splits
from bye_splits.utils import common, params

import re
import numpy as np
import h5py
import yaml

import yaml

def validation(mipPts, event, infile, outfile, nbinsrz, nbinsphi):
    """Compares all values of 2d histogram between local and CMSSW versions."""
    with open(infile, 'w') as flocal, open(outfile, 'r') as fremote:
        lines = fremote.readlines()

        for line in lines:
            l = line.split('\t')
            if l[0]=='\n' or '#' in l[0]:
                continue
            bin1 = int(l[0])
            bin2 = int(l[1])
            val_remote = float(l[2].replace('\n', ''))
            val_local = mipPts[bin1,bin2]
            if abs(val_remote-val_local)>0.0001:
                print('Diff found! Bin1={}\t Bin2={}\tRemote={}\tLocal={}'.format(bin1, bin2, val_remote, val_local))
                
        for bin1 in range(nbinsrz):
            for bin2 in range(nbinsphi):
                flocal.write('{}\t{}\t{}\n'.format(bin1, bin2, np.around(mipPts[bin1,bin2], 6)))

def fill_nans(weights, seeds):
    """Fills NaN values in bin with the mean of adjacent bins."""
    for bin_pair in zip(seeds[0],seeds[1]): # (R/z, phi) bins
        val = weights[bin_pair]
        if np.isnan(val):
            lft, rgt = (bin_pair[0], bin_pair[1]-1), (bin_pair[0], bin_pair[1]+1)
            weights[bin_pair] = 0.5*(weights[lft]+weights[rgt])
            
def seed(pars, debug=False, **kw):
    inseeding = common.fill_path(kw['SeedIn'], **pars)
    outseeding = common.fill_path(kw['SeedOut'], **pars)
    with open(params.CfgPath, 'r') as afile:
        cfg = yaml.safe_load(afile)
    with h5py.File(inseeding,  mode='r') as storeIn, h5py.File(outseeding, mode='w') as storeOut:
        bad_seeds=0
        for key in storeIn.keys():
            energies, wght_x, wght_y = storeIn[key]        
            window_size_phi = pars['seed_window']
            window_size_Rz  = 1
            surroundings = []
     
            # add unphysical top and bottom R/z rows for edge cases
            # fill the rows with negative (unphysical) energy values
            # boundary conditions on the phi axis are satisfied by 'np.roll'
            phiPad = -1 * np.ones((1,kw['NbinsPhi']))
            energies = np.concatenate((phiPad,energies,phiPad))
     
            #remove padding
            slc = slice(1,energies.shape[0]-1)
     
            # note: energies is by definition larger or equal to itself
            for iRz in range(-window_size_Rz, window_size_Rz+1):
                for iphi in range(-window_size_phi, window_size_phi+1):
                    surroundings.append(np.roll(energies, shift=(iRz,iphi), axis=(0,1))[slc])
     
            energies = energies[slc]
     
            # maxima = ( (energies > kw['histoThreshold'] ) &
            #            (energies >= south) & (energies > north) &
            #            (energies >= east) & (energies > west) &
            #            (energies >= northeast) & (energies > northwest) &
            #            (energies >= southeast) & (energies > southwest) )
            # TO DO: UPDATE THE >= WITH SOME >
            maxima = (energies > kw['histoThreshold'] )
            for surr in surroundings:
                maxima = maxima & (energies >= surr)

            seeds_idx = np.nonzero(maxima)
            res = [energies[seeds_idx], wght_x[seeds_idx], wght_y[seeds_idx]]
     
            # The 'flat_top' kernel might create a seed in a bin without any firing TC.
            # This happens when the two phi-adjacent bins would create two (split) clusters
            # had we used a default smoothing kernel.
            # The seed position cannot threfore be defined based on firing TC.
            # We thus perform the energy weighted average of the TC of the phi-adjacent bins.
            # Note: the first check avoids an error when an event has no seeds
            if res[0].shape[0]!=0 and np.isnan(res[1])[0] and np.isnan(res[2])[0]:
                if pars['smooth_kernel'] != 'flat_top':
                    #mes = 'Seeds with NaN values should appear only with flat_top smoothing.'
                    #raise ValueError(mes)
                    bad_seeds+=1
                    fill_nans(wght_x, seeds_idx)
                    fill_nans(wght_y, seeds_idx)
                    res[1], res[2] = wght_x[seeds_idx], wght_y[seeds_idx]
                elif len(res[1]) > 1:
                    mes = 'Only one cluster is expected in this scenario.'
                    raise ValueError(mes)
     
                lft = (seeds_idx[0][0], seeds_idx[1][0]-1)
                rgt = (seeds_idx[0][0], seeds_idx[1][0]+1)
                enboth = energies[lft] + energies[rgt]
                res[0] = np.array([enboth])
                res[1] = np.array([(wght_x[lft]*energies[lft]+wght_x[rgt]*energies[rgt])/enboth])
                res[2] = np.array([(wght_y[lft]*energies[lft]+wght_y[rgt]*energies[rgt])/enboth])
                    
            search_str = '{}_([0-9]{{1,7}})_group'.format(kw['FesAlgo'])
            event_number = re.search(search_str, key).group(1)
     
            if debug:
                print('Ev: {}'.format(event_number))
                print('Seeds bins: {}'.format(seeds_idx))
                print('NSeeds={}\tMipPt={}\tX={}\tY={}'.format(len(res[0]),res[0],res[1],res[2])) 

            storeOut[key] = res
            storeOut[key].attrs['columns'] = ['seedEn', 'seedXdivZ', 'seedYdivZ']
            storeOut[key].attrs['doc'] = 'Smoothed energies and projected bin positions of seeds'
        
        bad_perc = bad_seeds/len(storeIn.keys())
        print(f"\nPercentage of Bad Seeds: {bad_perc}\n")

if __name__ == "__main__":
    import argparse
    from bye_splits.utils import parsing

    parser = argparse.ArgumentParser(description='Seeding standalone step.')
    parsing.add_parameters(parser)
    FLAGS = parser.parse_args()

    seed_d = params.read_task_params('seed')
    seed(vars(FLAGS), **seed_d)
