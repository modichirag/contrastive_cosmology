import numpy as np
import sys, os
import matplotlib.pyplot as plt
import argparse
import pickle, json
from io import StringIO
import wandb
import yaml
import torch
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

selfpath = sys.path[0]
print(selfpath)
sys.path.append(f'{selfpath}/../../src/')
sys.path.append(f'{selfpath}/pkells/')
sys.path.append(f'{selfpath}/bispec/')
sys.path.append(f'{selfpath}/pknb/')
import sbitools, sbiplots
import loader_hod_ells as loader_ells
import loader_hod_bispec as loader_bk
import loader_hod_pknb as loader_pknb
import folder_path

api = wandb.Api()

#Which simulation to test
train_idx, test_idx = sbitools.test_train_split(np.arange(2000), None, train_size_frac=0.85, retindex=True)
isim = test_idx[int(sys.argv[1])]
ihod = int(sys.argv[2]) # load the simulation number from command line ##isim, ihod = test_idx[0], 0
finder_data = str(sys.argv[3]) #'Rockstar'
hodmodel_lh = str(sys.argv[4]) #'zheng07'
hodmodel_data = sys.argv[5]     # hod of test dataset
simulation='quijote'
z=0.5
nbar=0.0004
nposterior = 10
print(f"Predicting for LHC number {isim}, HOD realization {ihod}")


datapath = f'/mnt/ceph/users/cmodi/contrastive/data/{simulation}/{finder_data}/z{int(z*10):02d}-N{int(nbar/1e-4):04d}/{hodmodel_data}/'
print(f"data path : {datapath}")
savepath = f'/mnt/ceph/users/cmodi/contrastive/diagnostics/z{int(z*10):02d}-N{int(nbar/1e-4):04d}/ens{nposterior}/{simulation}-{finder_data}-{hodmodel_data}/tmpdata/'
print(f"save path : {savepath}")
savepath = f'{savepath}/S{isim:04d}-H{ihod:02d}'
os.makedirs(savepath, exist_ok=True)



def get_params(isim, ihod=None):
    cosmo_params = sbitools.quijote_params()[0]
    #print("cosmo params shape : ", cosmo_params.shape)

    if ihod is not None:
        hod_params = np.load(datapath + 'hodp.npy')
        #print("hod params shape : ", hod_params.shape)
        hodp = hod_params[isim, ihod]
        return np.concatenate([cosmo_params[isim],
                               hod_params[isim, ihod]])
    else:
        return cosmo_params[isim]
        

def get_pells(isim, ihod):
    pk = np.load(datapath + '/power_ell.npy')
    ngal = np.load(datapath + '/gals.npy')[..., 0]
    #print(pk.shape, ngal.shape)
    return pk[isim, ihod], ngal[isim, ihod]


def get_bispec(isim, ihod):
    qk = np.load(datapath + '/qspec.npy')
    bk = np.load(datapath + '/bspec.npy')
    #print(bk.shape)
    return bk[isim, ihod], qk[isim, ihod]

def get_scat(isim, ihod):
    sc0 = np.load(datapath + '/scat_0.npy')[isim, ihod]
    sc1 = np.load(datapath + '/scat_1.npy')[isim, ihod]
    sc2 = np.load(datapath + '/scat_2.npy')[isim, ihod]
    return sc0, sc1, sc2


def setup_cfg(cfg_path, verbose=False):
    
    cfg_path = insert_sweep_name(cfg_path)
    cfg_dict = yaml.load(open(f'{cfg_path}'), Loader=yaml.Loader)
    sweep_id = cfg_dict['sweep']['id']
    sweep = api.sweep(f'modichirag92/quijote-hodells/{sweep_id}')
    #sort in the order of validation log prob                                                                                                                     
    names, log_prob = [], []
    for run in sweep.runs:
        if run.state == 'finished':
            # print(run.name, run.summary['best_validation_log_prob'])
            try:
                model_path = run.summary['output_directory']
                names.append(run.name)
                log_prob.append(run.summary['best_validation_log_prob'])
            except Exception as e:
                print('Exception in checking state of run : ', e)
    idx = np.argsort(log_prob)[::-1]
    names = np.array(names)[idx]

    args = {}
    for i in cfg_dict.keys():
        args.update(**cfg_dict[i])
    cfg = sbitools.Objectify(**args)
    if ('scat' in cfg_path):
        cfg.analysis_path = folder_path.scat_path(cfg_dict, verbose=verbose)
    elif ('pknb' in cfg_path):
        cfg.analysis_path = folder_path.pknb_path(cfg_dict, verbose=verbose)
    elif 'ells' in cfg_path:
        cfg.analysis_path = folder_path.hodells_path(cfg_dict, verbose=verbose)
    elif ('bspec' in cfg_path) or ('qspec' in cfg_path):
        cfg.analysis_path = folder_path.bispec_path(cfg_dict, verbose=verbose)
    scaler = sbitools.load_scaler(cfg.analysis_path)

    toret = {'sweepid':sweep_id, 'cfg':cfg, 'idx':idx, 'scaler':scaler, 'names':names, 'cfg_dict':cfg_dict}
    return toret


def setup_bispec(bk, sweepdict, ngal=None, verbose=False):
    
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-bispec.npy')    
    cfg = sweepdict['cfg']
    scaler = sweepdict['scaler']
    bk = np.expand_dims(np.expand_dims(bk, 0), 0) #add batch and hod shape
    bk, offset = loader_bk.add_offset(cfg, bk.copy(), verbose=verbose)
    bk = loader_bk.k_cuts(cfg, bk, k, verbose=verbose)
    bk =  loader_bk.normalize_amplitude(cfg, bk, verbose=verbose)
    if cfg.ngals:
        # print("Add ngals to data")
        if ngal is None: raise NotImplementedError         
        bk = np.concatenate([bk, np.reshape([ngal], (1, 1, 1))], axis=-1)
    bk = bk[:, 0] #remove HOD axis
    features = sbitools.standardize(bk, scaler=scaler, log_transform=cfg.logit)[0]
    return features


def setup_pells(pells, sweepdict, ngal=None, verbose=False):
    
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')    
    cfg = sweepdict['cfg']
    scaler = sweepdict['scaler']
    pells = np.expand_dims(np.expand_dims(pells, 0), 0) #add batch and hod shape
    pells, offset = loader_ells.add_offset(cfg, pells.copy(), verbose=verbose)
    pells = loader_ells.k_cuts(cfg, pells, k, verbose=verbose)
    pells = loader_ells.subset_pk_ells(cfg, pells, verbose=verbose)
    pells =  loader_ells.normalize_amplitude(cfg, pells, verbose=verbose)
    if cfg.ngals:
        # print("Add ngals to data")
        if ngal is None: raise NotImplementedError         
        pells = np.concatenate([pells, np.reshape([ngal], (1, 1, 1))], axis=-1)
    pells = pells[:, 0]
    #print("pells shape : ", pells.shape)
    features = sbitools.standardize(pells, scaler=scaler, log_transform=cfg.logit)[0]
    return features


def setup_pknb(pknb, sweepdict, ngal=None, verbose=False):
    
    pells, bk = pknb
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')    
    kb = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-bispec.npy')    
    cfg = sweepdict['cfg']
    scaler = sweepdict['scaler']
    #
    pells = np.expand_dims(np.expand_dims(pells, 0), 0) #add batch and hod shape
    pells, offset_pk = loader_pknb.add_offset_pk(cfg, pells.copy(), verbose=verbose)
    pells = loader_pknb.k_cuts_pk(cfg, pells, k, verbose=verbose)
    pells = loader_pknb.subset_pk_ells(cfg, pells, verbose=verbose)
    pells =  loader_pknb.normalize_amplitude_pk(cfg, pells, verbose=verbose)
    
    bk = np.expand_dims(np.expand_dims(bk, 0), 0) #add batch and hod shape
    bk, offset_bk = loader_pknb.add_offset_bk(cfg, bk.copy(), verbose=verbose)
    bk = loader_pknb.k_cuts_bk(cfg, bk, kb, verbose=verbose)
    bk =  loader_pknb.normalize_amplitude_bk(cfg, bk, verbose=verbose)

    pknb = np.concatenate([pells, bk], axis=-1)
    if cfg.ngals:
        # print("Add ngals to data")
        if ngal is None: raise NotImplementedError         
        pknb = np.concatenate([pknb, np.reshape([ngal], (1, 1, 1))], axis=-1)
    pknb = pknb[:, 0]
    #print("pknb shape : ", pknb.shape)
    features = sbitools.standardize(pknb, scaler=scaler, log_transform=cfg.logit)[0]
    return features


def setup_scat(scs, sweepdict, ngal=None, verbose=False):
    
    s0, s1, s2 = scs
    cfg = sweepdict['cfg']
    scaler = sweepdict['scaler']

    #parse cfg and combine coefficients
    ellindex = [int(i) for i in cfg.ells]
    expindex = [int(i) for i in cfg.exps]
    orderindex = [int(i) for i in cfg.orders]
    s0 = s0[..., expindex].reshape(-1)
    s1 = s1[...,  ellindex, :][..., expindex].reshape(-1)
    s2 = s2[...,  ellindex, :][..., expindex].reshape(-1)

    ss = [s0, s1, s2]
    scat = np.concatenate([ss[i] for i in orderindex], axis=-1)
    if verbose:
        print("Combined scattering data to shape : ", scat.shape)

    if cfg.ngals:
        if ngal is None: raise NotImplementedError         
        scat = np.concatenate([scat, [ngal]], axis=0)
    scat = np.expand_dims(scat, 0) # add batch dims
    if verbose:
        print(scat.shape)
        
    features = sbitools.standardize(scat, scaler=scaler, log_transform=cfg.logit)[0]
    return features


def get_samples(summary, data, ngal, sweepdict, nposterior=1, nsamples=1000, cosmoonly=True, verbose=False):
    if summary == 'pells': features = setup_pells(data.copy(), sweepdict, ngal=ngal)
    elif summary == 'bispec': features = setup_bispec(data.copy(), sweepdict, ngal=ngal)
    elif summary == 'pknb': features = setup_pknb(data.copy(), sweepdict, ngal=ngal)
    elif summary == 'scat': features = setup_scat(data, sweepdict, ngal=ngal)
    sweepid = sweepdict['sweepid']
    posteriors = []
    for j in range(nposterior):
        name = sweepdict['names'][j]
        if verbose: print(name)
        model_path = f"{sweepdict['cfg'].analysis_path}/{sweepid}/{name}/"
        posteriors.append(sbitools.load_posterior(model_path))
    posterior = NeuralPosteriorEnsemble(posteriors=posteriors)
    samples = posterior.sample((nsamples,), x=torch.from_numpy(features.astype('float32')), show_progress_bars=verbose).detach().numpy()
    if cosmoonly: return np.array(samples)[:, :5]
    else: return np.array(samples)


#load data
pells, ngal = get_pells(isim, ihod)
bspec, qspec = get_bispec(isim, ihod)
pkb = [pells, bspec]
#scs = get_scat(isim, ihod)
params = get_params(isim, ihod)
np.save(f'{savepath}/params', params)


def parse_name(cfg_path, suffix=''):
    
    if cfg_path.find('quijote') + 1 :
        msim = 'quijote'
        if cfg_path.find('Rockstar') + 1 :
            msim = 'quijote-rockstar'
    else: msim = 'fastpm' 
    if cfg_path.find('scat') + 1 :
        name = f'{msim}-scat-ells012-exps0123-orders012'
        print(f"saving at {name}")
        return name
    if cfg_path.find('pknb') + 1 :
        summ = 'pknb'
        kmax = cfg_path.split("kmax")[1][:6] + cfg_path.split("kmax")[2][:6]
    elif cfg_path.find('ells') + 1 :
        summ = 'ells'
        kmax = cfg_path.split("kmax")[1][:3]
    else:
        summ = 'bk'
        kmax = cfg_path.split("kmax")[1][:3]
    name = f'{msim}-{hodmodel_lh}-{summ}-kmax{kmax}{suffix}'
    print(f"saving at {name}")
    return name


def insert_sweep_name(path):
    if '%s' in path: 
        dirname = '/'.join(cfg_path.split('/')[:-2])
    else: 
        dirname = path
    print(f"In directory {dirname}")
    for root, dirs, files in os.walk(dirname):
        if len(dirs) > 1 : 
            print('More than 1 sweeps, abort!')
            raise
        break
    print(f"Sweep found : {dirs[0]}")
    if '%s' in path: 
        return path%dirs[0]
    else:
        return path + f'/{dirs[0]}/'
    

print()
basepath = '/mnt/ceph/users/cmodi/contrastive/analysis/'
numd = 'z05-N0004'
hod = hodmodel_lh
suffix = '-train94'
suffixsave = '-train94-nogals'

############################################
## Quijote + FoF
sim = 'quijote'
finder = 'FoF'
print(f"\nFor LH with {sim}+{finder}")

## pkells
for kmax in [0.3, 0.5]:
    print()
    try:
        cfg_path = f'{basepath}/{sim}/{finder}/{numd}/{hod}/ells024-kmax{kmax:0.1f}-kmin0.005-offset_amp10000.0-standardize{suffix}/%s/sweep_config_hodells_{sim}.yaml'
        #cfg_path = f'{basepath}/{sim}/{finder}/{numd}/{hod}/ells024-kmax{kmax:0.1f}-kmin0.005-ngals-offset_amp10000.0-standardize/%s/sweep_config_hodells_{sim}.yaml'
        sweepdict = setup_cfg(cfg_path)
        samples = get_samples('pells', pells.copy(), ngal, sweepdict, nposterior=nposterior)
        name = parse_name(cfg_path, suffixsave)
        np.save(f'{savepath}/{name}', samples)
    except Exception as e:
        print(f"Exception occured when working for path\n{cfg_path}\n{e}")

## bispectrum
for kmax in [0.5]:
    print()
    try:
        cfg_path = f'{basepath}/{sim}/{finder}/{numd}/{hod}/bk-kmax{kmax:0.1f}-kmin0.005-standardize{suffix}/%s/sweep_config_bspec_{sim}.yaml'
        #cfg_path = f'{basepath}/{sim}/{finder}/{numd}/{hod}/bk-kmax{kmax:0.1f}-kmin0.005-ngals-standardize/%s/sweep_config_bspec_{sim}.yaml'
        sweepdict = setup_cfg(cfg_path)
        samples = get_samples('bispec', bspec.copy(), ngal, sweepdict, nposterior=nposterior)
        name = parse_name(cfg_path, suffixsave)
        np.save(f'{savepath}/{name}', samples)
    except Exception as e:
        print(f"Exception occured when working for path\n{cfg_path}\n{e}")

        
# ## pknb
# for bkmax in [0.3, 0.5]:
#     for pbkmax in [0.3, 0.5]:
#         try:
#             runpath = f'pknb-ells024-kmax_bk{bkmax:0.1f}-kmax_pk{bkmax:0.1f}-kmin_bk0.005-kmin_pk0.005-ngals-offset_amp_pk10000.0-standardize'        
#             cfg_path = f'{basepath}/{sim}/{finder}/{numd}/{hodmodel}/{runpath}/%s/sweep_config_pknb_{sim}.yaml'
#             sweepdict = setup_cfg(cfg_path)
#             samples = get_samples('pknb', pkb.copy(), ngal, sweepdict, nposterior=nposterior)
#             name = parse_name(cfg_path)
#             np.save(f'{savepath}/{name}', samples)
#         except Exception as e:
#             print(f"Exception occured when working for path\n{cfg_path}\n{e}")


# ############################################
# ## Quijote + Rockstar
sim = 'quijote'
finder = 'Rockstar'
print(f"\nFor LH with {sim}+{finder}")


## pkells
for kmax in [0.3, 0.5]:
    print()
    try:
        #cfg_path = f'{basepath}/{sim}/{finder}/{numd}/{hod}/ells024-kmax{kmax:0.1f}-kmin0.005-ngals-offset_amp10000.0-standardize/%s/sweep_config_hodells_{sim}.yaml'
        cfg_path = f'{basepath}/{sim}/{finder}/{numd}/{hod}/ells024-kmax{kmax:0.1f}-kmin0.005-offset_amp10000.0-standardize{suffix}/%s/sweep_config_hodells_{sim}.yaml'
        sweepdict = setup_cfg(cfg_path)
        samples = get_samples('pells', pells.copy(), ngal, sweepdict, nposterior=nposterior)
        name = parse_name(cfg_path, suffixsave)
        np.save(f'{savepath}/{name}', samples)
    except Exception as e:
        print(f"Exception occured when working for path\n{cfg_path}\n{e}")


## bispectrum
for kmax in [0.2, 0.3, 0.5]:
    print()
    try:
        #cfg_path = f'{basepath}/{sim}/{finder}/{numd}/{hod}/bk-kmax{kmax:0.1f}-kmin0.005-ngals-standardize/%s/sweep_config_bspec_{sim}.yaml'
        cfg_path = f'{basepath}/{sim}/{finder}/{numd}/{hod}/bk-kmax{kmax:0.1f}-kmin0.005-standardize{suffix}/%s/sweep_config_bspec_{sim}.yaml'
        sweepdict = setup_cfg(cfg_path)
        samples = get_samples('bispec', bspec.copy(), ngal, sweepdict, nposterior=nposterior)
        name = parse_name(cfg_path, suffixsave)
        np.save(f'{savepath}/{name}', samples)
    except Exception as e:
        print(f"Exception occured when working for path\n{cfg_path}\n{e}")


# ############################################
# ## FastPM + FOF
sim = 'fastpm'
finder = 'FoF'
print(f"\nFor LH with {sim}+{finder}")


## pkells
for kmax in [0.3, 0.5]:
    print()
    try:
        #cfg_path = f'{basepath}/{sim}/{finder}/{numd}/{hod}/ells024-kmax{kmax:0.1f}-kmin0.005-ngals-offset_amp10000.0-standardize/%s/sweep_config_hodells_{sim}.yaml'
        cfg_path = f'{basepath}/{sim}/{finder}/{numd}/{hod}/ells024-kmax{kmax:0.1f}-kmin0.005-offset_amp10000.0-standardize{suffix}/%s/sweep_config_hodells_{sim}.yaml'
        sweepdict = setup_cfg(cfg_path)
        samples = get_samples('pells', pells.copy(), ngal, sweepdict, nposterior=nposterior)
        name = parse_name(cfg_path, suffixsave)
        np.save(f'{savepath}/{name}', samples)
    except Exception as e:
        print(f"Exception occured when working for path\n{cfg_path}\n{e}")



## bispectrum
for kmax in [0.2, 0.3, 0.5]:
    print()
    try:
        #cfg_path = f'{basepath}/{sim}/{finder}/{numd}/{hod}/bk-kmax{kmax:0.1f}-kmin0.005-ngals-standardize/%s/sweep_config_bspec_{sim}.yaml'
        cfg_path = f'{basepath}/{sim}/{finder}/{numd}/{hod}/bk-kmax{kmax:0.1f}-kmin0.005-standardize{suffix}/%s/sweep_config_bspec_{sim}.yaml'
        sweepdict = setup_cfg(cfg_path)
        samples = get_samples('bispec', bspec.copy(), ngal, sweepdict, nposterior=nposterior)
        name = parse_name(cfg_path, suffixsave)
        np.save(f'{savepath}/{name}', samples)
    except Exception as e:
        print(f"Exception occured when working for path\n{cfg_path}\n{e}")



        
