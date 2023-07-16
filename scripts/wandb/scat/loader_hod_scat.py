import numpy as np
import sys, os
sys.path.append('../../../src/')
import sbitools
import argparse
import pickle


#####
def k_cuts(args, scat, k=None, verbose=True):
    return scat


def _add_offset(offset_amp, scat, seed):
    pass


def add_offset(args, scat, seed=None, verbose=True):
    return scat, None


def _normalize_amplitude(args, scat):
    if args.ampnorm:
        raise NotImplementedError
    return scat 

def normalize_amplitude(args, scat, verbose=True):
    if args.ampnorm:
        scat = _normalize_amplitude(args, scat) 
    return scat 


def hod_scat_lh_features(datapath, args, verbose=True):
    s0 = np.load(datapath + '/scat_0.npy')
    s1 = np.load(datapath + '/scat_1.npy')
    s2 = np.load(datapath + '/scat_2.npy')
    ngal = np.load(datapath + '/gals.npy')[..., 0] # read centrals only
    k = None
    #
    nsims = s0.shape[0]
    nhod = s0.shape[1]
    if verbose:
        print("Loaded scattering data with shapes : ", s0.shape, s1.shape, s2.shape)

    #parse args and combine coefficients
    ellindex = [int(i) for i in args.ells]
    expindex = [int(i) for i in args.exps]
    orderindex = [int(i) for i in args.orders]
    s0 = s0[..., expindex].reshape(nsims, nhod, -1)
    s1 = s1[:, :, :,  ellindex, :][..., expindex].reshape(nsims, nhod, -1)
    s2 = s2[:, :, :,  ellindex, :][..., expindex].reshape(nsims, nhod, -1)
    ss = [s0, s1, s2]
    scat = np.concatenate([ss[i] for i in orderindex], axis=-1)
    if verbose:
        print("Combined scattering data to shape : ", scat.shape)

    #Offset here
    scat, offset = add_offset(args, scat, verbose=verbose)

    #k cut
    scat = k_cuts(args, scat, k, verbose=verbose)

    # Normalize at large sacles
    scat = normalize_amplitude(args, scat, verbose=verbose)
    
    if args.ngals:
        print("Add ngals to data")
        if ngal.shape[1] != scat.shape[1]:
            print("Ngal shape is not consistent : ", ngal.shape, scat.shape)
            print("Subsizing galaxy catalog for bisepctrum, make sure it is consistent")
            ngal = ngal[:, :scat.shape[1]]
        features = np.concatenate([scat, np.expand_dims(ngal, -1)], axis=-1)
        
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
            print("Subsizing number of HOD runs for scattering, make sure it is consistent")
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



def hod_scat_lh(datapath, args):
    """
    Data:
    power spectrum multipoles and ngals
    Offset multipoles with a random constant amplitude scaled with offset_amp.
    """
    
    features, offset = hod_scat_lh_features(datapath, args)
    params = hod_lh_params(datapath, args, features)
    
    if offset is not None:
        print("offset shape: ", offset.shape)
        if args.standardize_hod:
            offset = sbitools.minmax(offset, log_transform=False)[0]
        params = np.concatenate([params, np.expand_dims(offset, -1)], axis=-1)
        print("Params shape after adding offset: ", params.shape)
                    
    return features, params



