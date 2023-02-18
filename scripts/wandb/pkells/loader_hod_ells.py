import numpy as np
import sys, os
sys.path.append('../../../src/')
import sbitools
import argparse
import pickle


#####
def k_cuts(args, k, pk):
    ikmin = np.where(k>args.kmin)[0][0]
    ikmax = np.where(k>args.kmax)[0][0]
    pk = pk[..., ikmin:ikmax, :]
    print("pk shape after k-cuts : ", pk.shape)
    return pk


def add_offset(args, pk):
    if args.offset_amp:
        print(f"Offset power spectra with amplitude: {args.offset_amp}")
        offset = args.offset_amp*np.random.uniform(1, 10, np.prod(pk.shape[:2]))
        offset = offset.reshape(pk.shape[0], pk.shape[1]) # different offset for sim & HOD realization
        pk = pk + offset[..., None, None]
    else:
        offset = None
    return pk, offset


def normalize_amplitude(args, pk):
    if args.ampnorm:
        pk /= pk[..., args.ampnorm:args.ampnorm+1]
        print(f"Normalize amplitude at scale-index: {args.ampnorm}")
    return pk


def subset_pk_ells(args, pk):
    
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
    print("pk shape after selecting ells : ", pk.shape)
    return pk



def hod_ells_lh_features(datapath, args):
    pk = np.load(datapath + '/power_ell.npy')
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')
    ngal = np.load(datapath + '/gals.npy')[..., 0] # read centrals only
    nsims, nhod = pk.shape[0], pk.shape[1]
    print("Loaded power spectrum data with shape : ", pk.shape)
    
    #Offset here
    pk, offset = add_offset(args, pk)

    #k cut
    pk = k_cuts(args, k, pk)

    #ells
    pk = subset_pk_ells(args, pk)
    
    # Normalize at large sacles
    pk = normalize_amplitude(args, pk)
    
    if args.ngals:
        print("Add ngals to data")
        features = np.concatenate([pk, np.expand_dims(ngal, -1)], axis=-1)
        
    print("Final features shape : ", features.shape)

    return features, offset



def hod_ells_lh_params(datapath, args, features):
    #
    cosmo_params = sbitools.quijote_params()[0]
    ncosmop = cosmo_params.shape[-1]
    nhod = features.shape[1]
    cosmo_params = np.repeat(cosmo_params, nhod, axis=0).reshape(-1, nhod, ncosmop)
    print("Cosmology parameter shape after nhod repeats", cosmo_params.shape)
    
    if args.fithod == 1: 
        hod_params = np.load(datapath + 'hodp.npy')
        print("HOD parameters shape", hod_params.shape)
        nhodp = hod_params.shape[-1] # number of hod parameters
        if args.standardize_hod:
            hod_params = sbitools.minmax(hod_params.reshape(-1, nhodp), log_transform=False)[0]
            
        hod_params = hod_params.reshape(-1, nhod, nhodp)
        params = np.concatenate([cosmo_params, hod_params], axis=-1)
        
    else: 
        params = cosmo_params
        
    print("Parameters shape : ", params.shape)
    return params



def hod_ells_lh(datapath, args):
    """
    Data:
    power spectrum multipoles and ngals
    Offset multipoles with a random constant amplitude scaled with offset_amp.
    """
    
    features, offset = hod_ells_lh_features(datapath, args)
    params = hod_ells_lh_params(datapath, args, features)
    
    if offset is not None:
        print("offset shape: ", offset.shape)
        if args.standardize_hod:
            offset = sbitools.minmax(offset, log_transform=False)[0]
        params = np.concatenate([params, np.expand_dims(offset, -1)], axis=-1)
        print("Params shape after adding offset: ", params.shape)
                    
    return features, params


