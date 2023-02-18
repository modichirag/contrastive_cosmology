import numpy as np
import sys, os
sys.path.append('../../src/')
import sbitools, sbiplots


def hodells_path(cfgd):
    cuts = cfgd['datacuts']
    args = {}
    for i in cfgd.keys():
        args.update(**cfgd[i])
    cfgd = sbitools.Objectify(**args)

    #
    datapath = f'/mnt/ceph/users/cmodi/contrastive/data/{cfgd.simulation}/{cfgd.finder}/z{int(cfgd.z*10):02d}-N{int(cfgd.nbar/1e-4):04d}/{cfgd.hodmodel}/'
    analysis_path = datapath.replace("data", "analysis")
    
    #folder name is decided by data-cuts imposed
    folder = ''
    for key in sorted(cuts):
        if cuts[key]:
            print(key, str(cuts[key]))
            if type(cuts[key]) == bool: folder = folder + f"{key}"
            else: folder = folder + f'{key}{cuts[key]}'
            folder += '-'
    folder = folder[:-1] + f'{cfgd.suffix}/'

    return analysis_path + folder


def bispec_path(cfgd):
    cuts = cfgd['datacuts']
    args = {}
    for i in cfgd.keys():
        args.update(**cfgd[i])
    cfgd = sbitools.Objectify(**args)

    #
    datapath = f'/mnt/ceph/users/cmodi/contrastive/data/{cfgd.simulation}/{cfgd.finder}/z{int(cfgd.z*10):02d}-N{int(cfgd.nbar/1e-4):04d}/{cfgd.hodmodel}/'
    analysis_path = datapath.replace("data", "analysis")
    
    #folder name is decided by data-cuts imposed
    folder = 'bk-'
    for key in sorted(cuts):
        if cuts[key]:
            print(key, str(cuts[key]))
            if type(cuts[key]) == bool: folder = folder + f"{key}"
            else: folder = folder + f'{key}{cuts[key]}'
            folder += '-'
    folder = folder[:-1] + f'{cfgd.suffix}/'

    return analysis_path + folder
