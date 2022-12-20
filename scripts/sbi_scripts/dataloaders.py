import numpy as np
import sys, os
sys.path.append('../../src/')
import sbitools
import argparse
import pickle

#####
def matter_pk(args):
    pk = np.load('../../sbi-scattering/data/quijote/latin_hypercube/pkmatter.npy')
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')
    #
    ikmin = np.where(k>args.kmin)[0][0]
    print(ikmin)
    ikmax = np.where(k>args.kmax)[0][0]
    pk = pk[..., ikmin:ikmax, :]

    features = pk.copy()
    print("Features shape : ", features.shape)
    #
    cosmo_params = sbitools.quijote_params()[0]
    ncosmop = cosmo_params.shape[-1]
    params = cosmo_params
    print("Parameters shape : ", params.shape)

    return features, params

#####
def hod_pk(args):
    datapath = "/mnt/ceph/users/cmodi/contrastive/data/" + args.datapath
    pk = np.load(datapath + '/power.npy')
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')
    ngal = np.load(datapath + '/gals.npy')[..., 0]
    nhod = pk.shape[1]
    print("Number of HODs per cosmology : ", nhod)
    #
    ikmin = np.where(k>args.kmin)[0][0]
    print(ikmin)
    ikmax = np.where(k>args.kmax)[0][0]
    pk = pk[..., ikmin:ikmax, :]

    if args.ampnorm: pk /= pk[..., 1:2] # Normalize at large sacles
    features = np.concatenate([pk, np.expand_dims(ngal, -1)], axis=-1)
    print("Features shape : ", features.shape)
    #
    cosmo_params = sbitools.quijote_params()[0]
    ncosmop = cosmo_params.shape[-1]
    cosmo_params = np.repeat(cosmo_params, nhod, axis=0).reshape(-1, nhod, ncosmop)
    if args.fithod == 1: 
        hod_params = np.load(datapath + 'hodp.npy')
        nhodp = hod_params.shape[-1]
        hod_params = sbitools.minmax(hod_params.reshape(-1, nhodp), log_transform=False)[0]
        hod_params = hod_params.reshape(-1, nhod, nhodp)
        params = np.concatenate([cosmo_params, hod_params], axis=-1)
    else: 
        params = cosmo_params
    print("Parameters shape : ", params.shape)

    return features, params



#####
def hod_ells(args, testing=False):
    datapath = "/mnt/ceph/users/cmodi/contrastive/data/" + args.datapath
    pk = np.load(datapath + '/power_ell.npy')
    pk += 1e8 #add offset to make it positive
    print("Shape of raw power spectra : ", pk.shape)
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')
    ngal = np.load(datapath + '/gals.npy')[..., 0]
    nhod = pk.shape[1]
    print("Number of HODs per cosmology : ", nhod)
    #
    ikmin = np.where(k>args.kmin)[0][0]
    ikmax = np.where(k>args.kmax)[0][0]
    pk = pk[..., ikmin:ikmax, :]
    
    #pk = np.swapaxes(pk, 2, 3)
    #pk = pk.reshape([pk.shape[0], pk.shape[1], -1])
    #pk = pk[..., 0]
    #if args.ampnorm: pk /= pk[..., 1:2] # Normalize at large sacles

    if len(args.ells) > 1:
        if args.ells == "02": pk = pk[..., [0, 1]]
        if args.ells == "04": pk = pk[..., [0, 2]]
        if args.ells == "24": pk = pk[..., [1, 2]]
        if args.ells == "024": pass
        pk = np.swapaxes(pk, 2, 3)
        pk = pk.reshape([pk.shape[0], pk.shape[1], -1])
    else:
        if args.ells == "0": pk = pk[..., 0]
        if args.ells == "2": pk = pk[..., 1]
        if args.ells == "4": pk = pk[..., 2]
    print("final pk shape : ", pk.shape)
    if args.ampnorm: pk /= pk[..., 1:2] # Normalize at large sacles
    
    features = np.concatenate([pk, np.expand_dims(ngal, -1)], axis=-1)
    print("Features shape : ", features.shape)
    #
    try:
        if args.fiducial: 
            cosmo_params = np.array([0.3175, 0.049, 0.6711,0.9624, 0.834]).reshape(1, -1)
            cosmo_params = np.repeat(cosmo_params, pk.shape[0], axis=0)
        else: cosmo_params = sbitools.quijote_params()[0]
    except:
        cosmo_params = sbitools.quijote_params()[0]
    
    ncosmop = cosmo_params.shape[-1]
    cosmo_params = np.repeat(cosmo_params, nhod, axis=0).reshape(-1, nhod, ncosmop)
    if args.fithod == 1: 
        hod_params = np.load(datapath + 'hodp.npy')
        nhodp = hod_params.shape[-1]
        print(cosmo_params.shape, hod_params.shape)
        hod_params = sbitools.minmax(hod_params.reshape(-1, nhodp), log_transform=False)[0]
        hod_params = hod_params.reshape(-1, nhod, nhodp)
        params = np.concatenate([cosmo_params, hod_params], axis=-1)
    else: 
        params = cosmo_params
    print("Parameters shape : ", params.shape)

    return features, params



#####
def hod_ells_combine(args):
    datapath = "/mnt/ceph/users/cmodi/contrastive/data/" + args.datapath
    datapath2 = "/mnt/ceph/users/cmodi/contrastive/data/" + args.datapath2
    paths = [datapath, datapath2]
    print(paths)
    pk, hod_params, ngal = [], [], []
    for path in paths:
        pk.append(np.load(path + '/power_ell.npy'))
        print("Shape of raw power spectra : ", pk[-1].shape)
        hod_params.append(np.load(path + 'hodp.npy'))
        ngal.append(np.load(path + '/gals.npy')[..., 0])
    pk = np.concatenate(pk, axis=1)
    ngal = np.concatenate(ngal, axis=1)
    hod_params = np.concatenate(hod_params, axis=1)
    print(pk.shape, ngal.shape, hod_params.shape)
    pk += 1e8 #add offset to make it positive
    nhod = pk.shape[1]
    print("Number of HODs per cosmology : ", nhod)
    #
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')
    ikmin = np.where(k>args.kmin)[0][0]
    ikmax = np.where(k>args.kmax)[0][0]
    pk = pk[..., ikmin:ikmax, :]
    
    #pk = np.swapaxes(pk, 2, 3)
    #pk = pk.reshape([pk.shape[0], pk.shape[1], -1])
    #pk = pk[..., 0]
    #if args.ampnorm: pk /= pk[..., 1:2] # Normalize at large sacles

    if len(args.ells) > 1:
        if args.ells == "02": pk = pk[..., [0, 1]]
        if args.ells == "04": pk = pk[..., [0, 2]]
        if args.ells == "24": pk = pk[..., [1, 2]]
        if args.ells == "024": pass
        pk = np.swapaxes(pk, 2, 3)
        pk = pk.reshape([pk.shape[0], pk.shape[1], -1])
    else:
        if args.ells == "0": pk = pk[..., 0]
        if args.ells == "2": pk = pk[..., 1]
        if args.ells == "4": pk = pk[..., 2]
    print("final pk shape : ", pk.shape)
    if args.ampnorm: pk /= pk[..., 1:2] # Normalize at large sacles
    
    features = np.concatenate([pk, np.expand_dims(ngal, -1)], axis=-1)
    print("Features shape : ", features.shape)
    #
    try:
        if args.fiducial: 
            cosmo_params = np.array([0.3175, 0.049, 0.6711,0.9624, 0.834]).reshape(1, -1)
            cosmo_params = np.repeat(cosmo_params, pk.shape[0], axis=0)
        else: cosmo_params = sbitools.quijote_params()[0]
    except:
        cosmo_params = sbitools.quijote_params()[0]
    
    ncosmop = cosmo_params.shape[-1]
    cosmo_params = np.repeat(cosmo_params, nhod, axis=0).reshape(-1, nhod, ncosmop)
    if args.fithod == 1: 
        nhodp = hod_params.shape[-1]
        print(cosmo_params.shape, hod_params.shape)
        hod_params = sbitools.minmax(hod_params.reshape(-1, nhodp), log_transform=False)[0]
        hod_params = hod_params.reshape(-1, nhod, nhodp)
        params = np.concatenate([cosmo_params, hod_params], axis=-1)
    else: 
        params = cosmo_params
    print("Parameters shape : ", params.shape)

    return features, params



#####
def hod_ells_offset(args, testing=False):
    datapath = "/mnt/ceph/users/cmodi/contrastive/data/" + args.datapath
    pk = np.load(datapath + '/power_ell.npy')
    #pk += 1e8 #add offset to make it positive
    offset = 1e4*np.random.uniform(1, 10, np.prod(pk.shape[:2])).reshape(pk.shape[0], pk.shape[1])
    #if testing: offset = offset*0. + 1e4
    pk = pk + offset[..., None, None]
    print("Shape of raw power spectra : ", pk.shape)
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')
    ngal = np.load(datapath + '/gals.npy')[..., 0]
    nhod = pk.shape[1]
    print("Number of HODs per cosmology : ", nhod)
    #
    ikmin = np.where(k>args.kmin)[0][0]
    ikmax = np.where(k>args.kmax)[0][0]
    pk = pk[..., ikmin:ikmax, :]
    
    if len(args.ells) > 1:
        if args.ells == "02": pk = pk[..., [0, 1]]
        if args.ells == "04": pk = pk[..., [0, 2]]
        if args.ells == "24": pk = pk[..., [1, 2]]
        if args.ells == "024": pass
        pk = np.swapaxes(pk, 2, 3)
        pk = pk.reshape([pk.shape[0], pk.shape[1], -1])
    else:
        if args.ells == "0": pk = pk[..., 0]
        if args.ells == "2": pk = pk[..., 1]
        if args.ells == "4": pk = pk[..., 2]
    print("final pk shape : ", pk.shape)
    if args.ampnorm: pk /= pk[..., 1:2] # Normalize at large sacles
    
    features = np.concatenate([pk, np.expand_dims(ngal, -1)], axis=-1)
    print("Features shape : ", features.shape)
    #
    try:
        if args.fiducial: 
            cosmo_params = np.array([0.3175, 0.049, 0.6711,0.9624, 0.834]).reshape(1, -1)
            cosmo_params = np.repeat(cosmo_params, pk.shape[0], axis=0)
        else: cosmo_params = sbitools.quijote_params()[0]
    except:
        cosmo_params = sbitools.quijote_params()[0]
    
    ncosmop = cosmo_params.shape[-1]
    cosmo_params = np.repeat(cosmo_params, nhod, axis=0).reshape(-1, nhod, ncosmop)
    if args.fithod == 1: 
        hod_params = np.load(datapath + 'hodp.npy')
        print(cosmo_params.shape, hod_params.shape)
        #if testing: hod_params = np.concatenate([hod_params, np.log10(1. + offset)[..., None]], axis=-1)
        hod_params = np.concatenate([hod_params, np.log10(offset)[..., None]], axis=-1)
        nhodp = hod_params.shape[-1]
        print(cosmo_params.shape, hod_params.shape)
        hod_params = sbitools.minmax(hod_params.reshape(-1, nhodp), log_transform=False)[0]
        hod_params = hod_params.reshape(-1, nhod, nhodp)
        params = np.concatenate([cosmo_params, hod_params], axis=-1)
    else: 
        params = cosmo_params
    print("Parameters shape : ", params.shape)

    return features, params


