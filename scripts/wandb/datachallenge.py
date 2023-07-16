import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.append('../../src/')
sys.path.append('../')
import sbitools, sbiplots
import argparse
import pickle, json
from io import StringIO
import wandb
import yaml
import torch
import loader_hod_ells as loader
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble
from nbodykit.lab import FFTPower, BigFileCatalog
from pmesh.pm import ParticleMesh, RealField

nc = 512
bs = 2000
pm = ParticleMesh(Nmesh=[nc, nc, nc], BoxSize=bs, dtype='f8')
import h5py as h5

def setup_cfg(cfg_path):
    
    cfg_dict = yaml.load(open(f'{cfg_path}'), Loader=yaml.Loader)
    sweep_id = cfg_dict['sweep']['id']


    sweep = api.sweep(f'modichirag92/quijote-hodells/{sweep_id}')
    #sort in the order of validation log prob                                                                                                                     
    names, log_prob = [], []
    for run in sweep.runs:
        if run.state == 'finished':
            # print(run.name, run.summary['best_validation_log_prob'])
            model_path = run.summary['output_directory']
            names.append(run.name)
            log_prob.append(run.summary['best_validation_log_prob'])

    idx = np.argsort(log_prob)[::-1]
    names = np.array(names)[idx]


    args = {}
    for i in cfg_dict.keys():
        args.update(**cfg_dict[i])
    cfg = sbitools.Objectify(**args)
    cfg.analysis_path = folder_path.hodells_path(cfg_dict)
    scaler = sbitools.load_scaler(cfg.analysis_path)

    toret = {'sweepid':sweep_id, 'cfg':cfg, 'idx':idx, 'scaler':scaler, 'names':names}
    return toret

def setup_features(pells, sweepdict, ngal=None):
    
    cfg = sweepdict['cfg']
    scaler = sweepdict['scaler']
    pells = np.expand_dims(pells, 0)
    pells = np.expand_dims(pells, 0)
    # print(pells.shape)
    pells, offset = loader.add_offset(cfg, pells.copy())
    pells = loader.k_cuts(cfg, kfid, pells)
    pells = loader.subset_pk_ells(cfg, pells)
    pells =  loader.normalize_amplitude(cfg, pells)
    if cfg.ngals:
        # print("Add ngals to data")
        if ngal is None: raise NotImplementedError         
        pells = np.concatenate([pells, np.reshape([ngal], (1, 1, 1))], axis=-1)
    pells = pells[:, 0]
    # print(pells.shape)
    features = sbitools.standardize(pells, scaler=scaler, log_transform=cfg.logit)[0]
    return features


def get_samples(features, sweepdict, nposterior=1, nsamples=1000, verbose=True):

    samples = []
    sweepid = sweepdict['sweepid']
    for j in range(nposterior):
        name = sweepdict['names'][j]
        if verbose: print(name)
        model_path = f"{sweepdict['cfg'].analysis_path}/{sweepid}/{name}/"
        posterior = sbitools.load_posterior(model_path)
        samples.append(posterior.sample((nsamples,), x=torch.from_numpy(features.astype('float32')), show_progress_bars=verbose).detach().numpy())
    return np.array(samples)


def get_samples_ens(features, sweepdict, nposterior=5, nsamples=1000, verbose=True):

    posteriors = []
    sweepid = sweepdict['sweepid']
    for j in range(nposterior):
        name = sweepdict['names'][j]
        if verbose: print(name)
        model_path = f"{sweepdict['cfg'].analysis_path}/{sweepid}/{name}/"
        posteriors.append(sbitools.load_posterior(model_path))
    posterior = NeuralPosteriorEnsemble(posteriors=posteriors)
    samples = posterior.sample((nsamples,), x=torch.from_numpy(features.astype('float32')), show_progress_bars=verbose).detach().numpy()
    return np.array(samples)



cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/Rockstar/z10-N0004/zheng07_velab/ells024-kmax0.5-kmin0.005-ngals-offset_amp10000.0-standardize/lked4v64/sweep_config_hodells_quijote.yaml'
sweepdict0 = setup_cfg(cfg_path)
cfg_path1 = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/Rockstar/z10-N0004/zheng07_velab/ells024-kmax0.3-kmin0.005-ngals-offset_amp10000.0-standardize/tb2ojwm3/sweep_config_hodells_quijote.yaml'
sweepdict1 = setup_cfg(cfg_path1)


#read in data
allpells = []
allngals = []
for i in range(1, 11):
    print(i)
    infile = h5.File(f"../data/mock_lcdm_redshift-space_{i}.h5", 'r')
    galaxies = infile['galaxies']
    pos = np.stack([galaxies['x'], galaxies['y'], galaxies['z_los']], axis=1)
    infile.close()
    
    ngal = pos.shape[0]/ 2**3 #rescale by volume, Quijote vs mock data

    mesh = pm.paint(pos)
    pkrsd = FFTPower(mesh/mesh.cmean(), mode='2d', Nmu=12, poles=[0, 2, 4], dk=2*np.pi/1000)
    pells = np.array([pkrsd.poles.data['power_%d'%i].real for i in [0, 2, 4]]).T
    allpells.append(pells.copy())
    allngals.append(ngal)




features = []
for j in range(10):
    pells = allpells[j]*1.
    features0 = setup_features(pells.copy(), sweepdict0, ngal=allngals[j])
    features1 = setup_features(pells.copy(), sweepdict1, ngal=allngals[j])
    features.append([features0, features1])


      
samplesens = []
for j in range(10):
    samples0 = get_samples_ens(features[j][0], sweepdict0, nposterior=5, verbose=False)
    samples1 = get_samples_ens(features[j][1], sweepdict1, nposterior=5, verbose=False)
    samplesens.append([samples0, samples1])
