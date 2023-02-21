import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.append('../../src/')
sys.path.append('./pkells/')
sys.path.append('./bispec/')
import sbitools, sbiplots
import argparse
import pickle, json
from io import StringIO
import wandb
import yaml
import torch
import loader_hod_ells as loader_ells
import loader_hod_bispec as loader_bk
import folder_path
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

api = wandb.Api()

simulation='fastpm'
finder='FoF'
z=0.5
nbar=0.0004
hodmodel='zheng07_velab'
datapath = f'/mnt/ceph/users/cmodi/contrastive/data/{simulation}/{finder}/z{int(z*10):02d}-N{int(nbar/1e-4):04d}/{hodmodel}/'
print(f"data path : {datapath}")

def get_params(isim, ihod=None):
    cosmo_params = sbitools.quijote_params()[0]
    print(cosmo_params.shape)

    if ihod is not None:
        hod_params = np.load(datapath + 'hodp.npy')
        print(hod_params.shape)
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


def setup_cfg(cfg_path):
    
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
                print(e)
    idx = np.argsort(log_prob)[::-1]
    names = np.array(names)[idx]

    args = {}
    for i in cfg_dict.keys():
        args.update(**cfg_dict[i])
    cfg = sbitools.Objectify(**args)
    if 'ells' in cfg_path:
        cfg.analysis_path = folder_path.hodells_path(cfg_dict)
    elif ('bspec' in cfg_path) or ('qspec' in cfg_path):
        cfg.analysis_path = folder_path.bispec_path(cfg_dict)
    scaler = sbitools.load_scaler(cfg.analysis_path)

    toret = {'sweepid':sweep_id, 'cfg':cfg, 'idx':idx, 'scaler':scaler, 'names':names}
    return toret

def setup_bispec(bk, sweepdict, ngal=None):
    
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-bispec.npy')    
    cfg = sweepdict['cfg']
    scaler = sweepdict['scaler']
    bk = np.expand_dims(np.expand_dims(bk, 0), 0) #add batch and hod shape
    bk, offset = loader_bk.add_offset(cfg, bk.copy(), seed=1)
    bk = loader_bk.k_cuts(cfg, k, bk)
    bk =  loader_bk.normalize_amplitude(cfg, bk)
    if cfg.ngals:
        # print("Add ngals to data")
        if ngal is None: raise NotImplementedError         
        bk = np.concatenate([bk, np.reshape([ngal], (1, 1, 1))], axis=-1)
    bk = bk[:, 0] #remove HOD axis
    features = sbitools.standardize(bk, scaler=scaler, log_transform=cfg.logit)[0]
    return features


def setup_pells(pells, sweepdict, ngal=None):
    
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')    
    cfg = sweepdict['cfg']
    scaler = sweepdict['scaler']
    pells = np.expand_dims(np.expand_dims(pells, 0), 0) #add batch and hod shape
    pells, offset = loader_ells.add_offset(cfg, pells.copy(), seed=2)
    pells = loader_ells.k_cuts(cfg, k, pells)
    pells = loader_ells.subset_pk_ells(cfg, pells)
    pells =  loader_ells.normalize_amplitude(cfg, pells)
    if cfg.ngals:
        # print("Add ngals to data")
        if ngal is None: raise NotImplementedError         
        pells = np.concatenate([pells, np.reshape([ngal], (1, 1, 1))], axis=-1)
    pells = pells[:, 0]
    print("pells shape : ", pells.shape)
    features = sbitools.standardize(pells, scaler=scaler, log_transform=cfg.logit)[0]
    return features



def get_samples(summary, data, ngal, sweepdict, nposterior=1, nsamples=1000, verbose=True):
    if summary == 'pells': features = setup_pells(data.copy(), sweepdict, ngal=ngal)
    elif summary == 'bispec': features = setup_bispec(data.copy(), sweepdict, ngal=ngal)
    sweepid = sweepdict['sweepid']
    posteriors = []
    for j in range(nposterior):
        name = sweepdict['names'][j]
        if verbose: print(name)
        model_path = f"{sweepdict['cfg'].analysis_path}/{sweepid}/{name}/"
        posteriors.append(sbitools.load_posterior(model_path))
    posterior = NeuralPosteriorEnsemble(posteriors=posteriors)
    samples = posterior.sample((nsamples,), x=torch.from_numpy(features.astype('float32')), show_progress_bars=verbose).detach().numpy()
    return np.array(samples)


#load data
train_idx, test_idx = sbitools.test_train_split(np.arange(2000), None, train_size_frac=0.85, retindex=True)
isim = test_idx[1]
ihod = 5
pells, ngal = get_pells(isim, ihod)
bspec, qspec = get_bispec(isim, ihod)
params = get_params(isim, ihod)
np.save(f'diagnostics/{simulation}/tmpdata/S{isim:04d}-H{ihod:02d}-params', params)


def parse_name(cfg_path):
    if cfg_path.find('quijote') + 1 : msim = 'quijote'
    else:msim = 'fastpm' 
    if cfg_path.find('ells') + 1 : summ = 'ells'
    else: summ = 'bk'
    kmax = cfg_path.split("kmax")[1][:3]
    suffix = ''
    name = f'{msim}-{summ}-kmax{kmax}{suffix}'
    return name


nposterior = 5
cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//quijote/FoF/z05-N0004/zheng07_velab/ells024-kmax0.5-kmin0.005-ngals-offset_amp10000.0-standardize/28a1ot12/sweep_config_hodells_quijote.yaml'
sweepdict = setup_cfg(cfg_path)
samples = get_samples('pells', pells.copy(), ngal, sweepdict, nposterior=nposterior)
name = parse_name(cfg_path)
np.save(f'diagnostics//{simulation}/tmpdata/S{isim:04d}-H{ihod:02d}-{name}', samples)

cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//fastpm/FoF/z05-N0004/zheng07_velab/ells024-kmax0.5-kmin0.005-ngals-offset_amp10000.0-standardize/y1nzq7gl/sweep_config_hodells_fastpm.yaml'
sweepdict = setup_cfg(cfg_path)
samples = get_samples('pells', pells.copy(), ngal, sweepdict, nposterior=nposterior)
name = parse_name(cfg_path)
np.save(f'diagnostics//{simulation}/tmpdata/S{isim:04d}-H{ihod:02d}-{name}', samples)

cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//fastpm/FoF/z05-N0004/zheng07_velab/bk-kmax0.5-kmin0.005-ngals-standardize/3ba457rn/sweep_config_bspec_fastpm.yaml'
sweepdict = setup_cfg(cfg_path)
samples = get_samples('bispec', bspec.copy(), ngal, sweepdict, nposterior=nposterior)
name = parse_name(cfg_path)
np.save(f'diagnostics//{simulation}/tmpdata/S{isim:04d}-H{ihod:02d}-{name}', samples)

cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//fastpm/FoF/z05-N0004/zheng07_velab/bk-kmax0.3-kmin0.005-ngals-standardize/9tlu63w4/sweep_config_bspec_fastpm.yaml'
sweepdict = setup_cfg(cfg_path)
samples = get_samples('bispec', bspec.copy(), ngal, sweepdict, nposterior=nposterior)
name = parse_name(cfg_path)
np.save(f'diagnostics//{simulation}/tmpdata/S{isim:04d}-H{ihod:02d}-{name}', samples)

cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//quijote/FoF/z05-N0004/zheng07_velab/bk-kmax0.3-kmin0.005-ngals-standardize/hhglght1/sweep_config_bspec_quijote.yaml'
sweepdict = setup_cfg(cfg_path)
samples = get_samples('bispec', bspec.copy(), ngal, sweepdict, nposterior=nposterior)
name = parse_name(cfg_path)
np.save(f'diagnostics//{simulation}/tmpdata/S{isim:04d}-H{ihod:02d}-{name}', samples)

cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//quijote/FoF/z05-N0004/zheng07_velab/bk-kmax0.5-kmin0.005-ngals-standardize/d8075uar/sweep_config_bspec_quijote.yaml'
sweepdict = setup_cfg(cfg_path)
samples = get_samples('bispec', bspec.copy(), ngal, sweepdict, nposterior=nposterior)
name = parse_name(cfg_path)
np.save(f'diagnostics//{simulation}/tmpdata/S{isim:04d}-H{ihod:02d}-{name}', samples)

