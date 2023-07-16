import numpy as np
import sys, os
sys.path.append('../../../src/')
import sbitools
import argparse
import pickle


#####
def k_cuts(args, bk, k=None, verbose=True):
    if k is None:
        k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-bispec.npy')
    ikmin = np.where(k[:, 0]>args.kmin)[0][0]
    if k[:, 0].max()>args.kmax :
        ikmax = np.where(k[:, 0]>args.kmax)[0][0]
        if verbose: print(f"kmax cut at {ikmax}")
        bk = bk[..., ikmin:ikmax]
    else:
        if verbose: print(f"kmax cut at {args.kmax} is on smaller scales than the data at {k[:, 0].max()}")
        bk = bk[..., ikmin:, :]
    if verbose: print("bk shape after k-cuts : ", bk.shape)
    return bk


def _add_offset(offset_amp, bk, seed):
    offset = offset_amp*np.random.uniform(1, 10, np.prod(bk.shape[:2]))
    offset = offset.reshape(bk.shape[0], bk.shape[1]) # different offset for sim & HOD realization
    bk = bk + offset[..., None]
    return bk, offset


def add_offset(args, bk, seed=None, verbose=True):
    if seed is not None: np.random.seed(seed)
    if args.offset_amp:
        if verbose: print(f"Offset power spectra with amplitude: {args.offset_amp}")
        bk, offset = _add_offset(args.offset_amp, bk, seed)
    else:
        offset = None
    return bk, offset


def _normalize_amplitude(args, bk):
    if args.ampnorm:
        raise NotImplementedError
    return bk 

def normalize_amplitude(args, bk, verbose=True):
    if args.ampnorm:
        bk = _normalize_amplitude(args, bk) 
    return bk 


def hod_bispec_lh_features(datapath, args, verbose=True):
    if args.reduced: bk = np.load(datapath + '/qspec.npy')
    else: bk = np.load(datapath + '/bspec.npy')
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-bispec.npy')
    if verbose: print("k shape : ", k.shape)
    ngal = np.load(datapath + '/gals.npy')[..., 0] # read centrals only
    nsims, nhod = bk.shape[0], bk.shape[1]
    if verbose: print("Loaded bispectrum data with shape : ", bk.shape)
    
    #Offset here
    bk, offset = add_offset(args, bk, verbose=verbose)

    #k cut
    bk = k_cuts(args, bk, k, verbose=verbose)

    # Normalize at large sacles
    bk = normalize_amplitude(args, bk, verbose=verbose)
    
    if args.ngals:
        print("Add ngals to data")
        if ngal.shape[1] != bk.shape[1]:
            print("Ngal shape is not consistent : ", ngal.shape, bk.shape)
            print("Subsizing galaxy catalog for bisepctrum, make sure it is consistent")
            ngal = ngal[:, :bk.shape[1]]
        features = np.concatenate([bk, np.expand_dims(ngal, -1)], axis=-1)
    else:
        features = bk.copy()
        
    print("Final features shape : ", features.shape)
    return features, offset



def hod_lh_params(datapath, args, features):
    #
    cosmo_params = sbitools.quijote_params()[0]
    ncosmop = cosmo_params.shape[-1]
    nhod = features.shape[1]
    cosmo_params = np.repeat(cosmo_params, nhod, axis=0).reshape(-1, nhod, ncosmop)
    print("Cosmology parameter shape after nhod repeats", cosmo_params.shape)
    
    if args.fithod == 1: 
        hod_params = np.load(datapath + 'hodp.npy')
        if hod_params.shape[1] != nhod:
            print("HOD shape is not consistent : ", hod_params.shape, nhod)
            print("Subsizing number of HOD runs for bisepctrum, make sure it is consistent")
            hod_params = hod_params[:, :nhod]

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



def hod_bispec_lh(datapath, args):
    """
    Data:
    power spectrum multipoles and ngals
    Offset multipoles with a random constant amplitude scaled with offset_amp.
    """
    
    features, offset = hod_bispec_lh_features(datapath, args)
    params = hod_lh_params(datapath, args, features)
    
    if offset is not None:
        print("offset shape: ", offset.shape)
        if args.standardize_hod:
            offset = sbitools.minmax(offset, log_transform=False)[0]
        params = np.concatenate([params, np.expand_dims(offset, -1)], axis=-1)
        print("Params shape after adding offset: ", params.shape)
                    
    return features, params



