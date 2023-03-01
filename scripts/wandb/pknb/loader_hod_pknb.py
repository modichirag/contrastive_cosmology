import numpy as np
import sys, os
sys.path.append('../../../src/')
sys.path.append('../pkells/')
import loader_hod_ells as loader_pk
sys.path.append('../bispec/')
import loader_hod_bispec as loader_bk
import sbitools
import argparse
import pickle


##Re-write it as imports from pkells and bispce file
def k_cuts_pk(args, pk, k=None):
    if k is None:
        k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')
    ikmin = np.where(k>args.kmin_pk)[0][0]
    ikmax = np.where(k>args.kmax_pk)[0][0]
    pk = pk[..., ikmin:ikmax, :]
    print("pk shape after k-cuts : ", pk.shape)
    return pk

def k_cuts_bk(args, bk, k=None):
    if k is None:
        k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-bispec.npy')
    ikmin = np.where(k[:, 0]>args.kmin_bk)[0][0]
    if k[:, 0].max()>args.kmax_bk :
        ikmax = np.where(k[:, 0]>args.kmax_bk)[0][0]
        print(f"kmax cut at {ikmax}")
        bk = bk[..., ikmin:ikmax]
    else:
        print(f"kmax cut at {args.kmax_bk} is on smaller scales than the data at {k[:, 0].max()}")
        bk = bk[..., ikmin:, :]
    print("bk shape after k-cuts : ", bk.shape)
    return bk


def add_offset_pk(args, pk, seed=None):
    if seed is not None: np.random.seed(seed)
    if args.offset_amp_pk:
        pk, offset = loader_pk._add_offset(args.offset_amp_pk, pk, seed)                
        print(f"Offset power spectra with amplitude: {args.offset_amp_pk}")
        # offset = args.offset_amp_pk*np.random.uniform(1, 10, np.prod(pk.shape[:2]))
        # offset = offset.reshape(pk.shape[0], pk.shape[1]) # different offset for sim & HOD realization
        # pk = pk + offset[..., None, None] #add k and ells dimension
    else:
        offset = None
    return pk, offset


def add_offset_bk(args, bk, seed=None):
    if seed is not None: np.random.seed(seed)
    if args.offset_amp_bk:
        bk, offset = loader_bk._add_offset(args.offset_amp_bk, bk, seed)                
        print(f"Offset power spectra with amplitude: {args.offset_amp_bk}")
        # offset = args.offset_amp_bk*np.random.uniform(1, 10, np.prod(bk.shape[:2]))
        # offset = offset.reshape(bk.shape[0], bk.shape[1]) # different offset for sim & HOD realization
        # bk = bk + offset[..., None]
    else:
        offset = None
    return bk, offset


def normalize_amplitude_bk(args, bk):
    if args.ampnorm_bk:
        raise NotImplementedError
    return bk 

def normalize_amplitude_pk(args, pk):
    if args.ampnorm_pk:
        pk = loader_pk._ampnorm(args.ampnorm_pk, pk)
        # pk /= pk[..., args.ampnorm_pk:args.ampnorm_pk+1]
        print(f"Normalize amplitude at scale-index: {args.ampnorm_pk}")
    return pk


def subset_pk_ells(args, pk):
    pk = loader_pk.subset_pk_ells(args, pk)
    # if len(args.ells) > 1:
    #     if args.ells == "02": pk = pk[..., [0, 1]]
    #     if args.ells == "04": pk = pk[..., [0, 2]]
    #     if args.ells == "24": pk = pk[..., [1, 2]]
    #     if args.ells == "024": pass
    #     pk = np.swapaxes(pk, 2, 3)
    #     pk = pk.reshape([pk.shape[0], pk.shape[1], -1])
    # else:
    #     if args.ells == "0": pk = pk[..., 0]
    #     if args.ells == "2": pk = pk[..., 1]
    #     if args.ells == "4": pk = pk[..., 2]
    # print("pk shape after selecting ells : ", pk.shape)
    return pk


def hod_bispec_lh_features(datapath, args):
    if args.reduced: bk = np.load(datapath + '/qspec.npy')
    else: bk = np.load(datapath + '/bspec.npy')
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-bispec.npy')
    print("k shape : ", k.shape)
    ngal = np.load(datapath + '/gals.npy')[..., 0] # read centrals only
    nsims, nhod = bk.shape[0], bk.shape[1]
    print("Loaded bispectrum data with shape : ", bk.shape)
    
    #Offset here
    bk, offset = add_offset_bk(args, bk)

    #k cut
    bk = k_cuts_bk(args, bk, k)

    # Normalize at large sacles
    bk = normalize_amplitude_bk(args, bk)
    
    if args.ngals:
        print("Add ngals to data")
        if ngal.shape[1] != bk.shape[1]:
            print("Ngal shape is not consistent : ", ngal.shape, bk.shape)
            print("Subsizing galaxy catalog for bisepctrum, make sure it is consistent")
            ngal = ngal[:, :bk.shape[1]]
        features = np.concatenate([bk, np.expand_dims(ngal, -1)], axis=-1)
        
    print("Final features shape : ", features.shape)
    return features, offset



# def hod_ells_lh_features(datapath, args):
#     pk = np.load(datapath + '/power_ell.npy')[:, :10] #bispectrum has 10 HOD realizations only
#     k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')
#     ngal = np.load(datapath + '/gals.npy')[..., 0] # read centrals only
#     nsims, nhod = pk.shape[0], pk.shape[1]
#     print("Loaded power spectrum data with shape : ", pk.shape)
    
#     #Offset here
#     pk, offset = add_offset_pk(args, pk)

#     #k cut
#     pk = k_cuts_pk(args, pk, k)

#     #ells
#     pk = subset_pk_ells(args, pk)
    
#     # Normalize at large sacles
#     pk = normalize_amplitude_pk(args, pk)

#     features = pk*1.
#     print("Final features shape : ", features.shape)

#     return features, offset


# def hod_bispec_lh_features(datapath, args):
#     if args.reduced: bk = np.load(datapath + '/qspec.npy')
#     else: bk = np.load(datapath + '/bspec.npy')
#     k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-bispec.npy')
#     print("k shape : ", k.shape)
#     ngal = np.load(datapath + '/gals.npy')[..., 0] # read centrals only
#     nsims, nhod = bk.shape[0], bk.shape[1]
#     print("Loaded bispectrum data with shape : ", bk.shape)
    
#     #Offset here
#     bk, offset = loader_bk.add_offset(args, bk)

#     #k cut
#     bk = loader_bk.k_cuts(args, bk, k)

#     # Normalize at large sacles
#     bk = loader_bk.normalize_amplitude(args, bk)
    
#     if args.ngals:
#         print("Add ngals to data")
#         if ngal.shape[1] != bk.shape[1]:
#             print("Ngal shape is not consistent : ", ngal.shape, bk.shape)
#             print("Subsizing galaxy catalog for bisepctrum, make sure it is consistent")
#             ngal = ngal[:, :bk.shape[1]]
#         features = np.concatenate([bk, np.expand_dims(ngal, -1)], axis=-1)
        
#     print("Final features shape : ", features.shape)
#     return features, offset



# def hod_ells_lh_features(datapath, args):
#     pk = np.load(datapath + '/power_ell.npy')[:, :10] #bispectrum has 10 HOD realizations only
#     k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')
#     ngal = np.load(datapath + '/gals.npy')[..., 0] # read centrals only
#     nsims, nhod = pk.shape[0], pk.shape[1]
#     print("Loaded power spectrum data with shape : ", pk.shape)
    
#     #Offset here
#     pk, offset = loader_pk.add_offset(args, pk)

#     #k cut
#     pk = loader_pk.k_cuts(args, pk, k)

#     #ells
#     pk = loader_pk.subset_pk_ells(args, pk) 
    
#     # Normalize at large sacles
#     pk = loader_pk.normalize_amplitude(args, pk)

#     features = pk*1.
#     print("Final features shape : ", features.shape)

#     return features, offset



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



def hod_pknb_lh(datapath, args):
    """
    Data:
    power spectrum multipoles and ngals
    Offset multipoles with a random constant amplitude scaled with offset_amp.
    """
    
    features_pk, offset_pk = hod_ells_lh_features(datapath, args)
    features_bk, offset_bk = hod_bispec_lh_features(datapath, args)
    print(features_pk.shape, features_bk.shape)
    features = np.concatenate([features_pk, features_bk], axis=-1)
    params = hod_lh_params(datapath, args, features)
    
    if offset_pk is not None:
        print("offset shape: ", offset_pk.shape)
        if args.standardize_hod:
            offset_pk = sbitools.minmax(offset_pk, log_transform=False)[0]
        params = np.concatenate([params, np.expand_dims(offset_pk, -1)], axis=-1)
        print("Params shape after adding offset: ", params.shape)
                    
    if offset_bk is not None:
        print("offset shape: ", offset_bk.shape)
        if args.standardize_hod:
            offset_bk = sbitools.minmax(offset_bk, log_transform=False)[0]
        params = np.concatenate([params, np.expand_dims(offset_bk, -1)], axis=-1)
        print("Params shape after adding offset: ", params.shape)
                    
    return features, params



