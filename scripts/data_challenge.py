import numpy as np
import sys, os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ruamel import yaml
import wandb
api = wandb.Api()

import sys
sys.path.append('../src/')
sys.path.append('../scripts/wandb/')
sys.path.append('../scripts/wandb/pkells/')
sys.path.append('../scripts/wandb/bispec//')
sys.path.append('../scripts/wandb/pknb///')
# import pickle
import sbitools
import loader_hod_ells as loader_ells
import loader_hod_bispec as loader_bispec
import loader_hod_pknb as loader_pknb
import folder_path

import torch
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

# from nbodykit.lab import FFTPower, BigFileCatalog
# from pmesh.pm import ParticleMesh, RealField

nc = 512
bs = 2000

kfid = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')

def setup_cfg(cfg_path, verbose=True):
    
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
    
    if ('pknb' in cfg_path) :
        cfg.analysis_path = folder_path.pknb_path(cfg_dict, verbose=verbose)
    elif 'ells' in cfg_path:
        cfg.analysis_path = folder_path.hodells_path(cfg_dict, verbose=verbose)
    elif ('bspec' in cfg_path) or ('qspec' in cfg_path):
        cfg.analysis_path = folder_path.bispec_path(cfg_dict, verbose=verbose)
    scaler = sbitools.load_scaler(cfg.analysis_path)
                                  
    toret = {'sweepid':sweep_id, 'cfg':cfg, 'idx':idx, 'scaler':scaler, 'names':names,  'cfg_dict':cfg_dict}
    return toret

def setup_features_pells(pells, sweepdict, ngal=None, verbose=True, standardize=True):
    
    cfg = sweepdict['cfg']
    scaler = sweepdict['scaler']
    pells = np.expand_dims(pells, 0)
    pells = np.expand_dims(pells, 0)
    # print(pells.shape)
    pells, offset = loader_ells.add_offset(cfg, pells.copy(), verbose=verbose)
    pells = loader_ells.k_cuts(cfg, pk=pells, k=kfid, verbose=verbose)
    pells = loader_ells.subset_pk_ells(cfg, pells, verbose=verbose)
    pells =  loader_ells.normalize_amplitude(cfg, pells, verbose=verbose)
    if cfg.ngals:
        # print("Add ngals to data")
        if ngal is None: raise NotImplementedError         
        pells = np.concatenate([pells, np.reshape([ngal], (1, 1, 1))], axis=-1)
    pells = pells[:, 0]
    # print(pells.shape)
    
    if standardize:
        features = sbitools.standardize(pells, scaler=scaler, log_transform=cfg.logit)[0]
    else:
        features = pells.copy()
    return features


def setup_features_bispec(bk, sweepdict, ngal=None, verbose=True, standardize=True):
    
    cfg = sweepdict['cfg']
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-bispec.npy')
    if verbose: print("k shape : ", k.shape)
    
    bk = np.expand_dims(bk, 0) #expand nsim dimension
    bk = np.expand_dims(bk, 0) #expand nhod dimension
    bk, offset = loader_bispec.add_offset(cfg, bk, verbose=verbose)
    bk = loader_bispec.k_cuts(cfg, bk, k, verbose=verbose)
    bk = loader_bispec.normalize_amplitude(cfg, bk, verbose=verbose)

    if cfg.ngals:
        if ngal is None: raise NotImplementedError         
        bk = np.concatenate([bk, np.reshape([ngal], (1, 1, 1))], axis=-1)

    bk = bk[:, 0]
    if standardize:
        features = sbitools.standardize(bk, scaler=sweepdict['scaler'], \
                                    log_transform=cfg.logit)[0]
    else:
        features = bk.copy()
    return features
    

def setup_features_pknb(pells, bk, sweepdict, ngal=None, verbose=True):
    
    cfg = sweepdict['cfg']
    scaler = sweepdict['scaler']
    
    #
    pells = np.expand_dims(pells, 0) #expand nsim dimension
    pells = np.expand_dims(pells, 0) #expand nhod dimension
    pells, offset = loader_pknb.add_offset_pk(cfg, pells.copy(), verbose=verbose)
    pells = loader_pknb.k_cuts_pk(cfg, pk=pells, k=kfid, verbose=verbose)
    pells = loader_pknb.subset_pk_ells(cfg, pells, verbose=verbose)
    pells =  loader_pknb.normalize_amplitude_pk(cfg, pells, verbose=verbose)
    pells = pells[:, 0]
    
    bk = np.expand_dims(bk, 0) #expand nsim dimension
    bk = np.expand_dims(bk, 0) #expand nhod dimension
    bk, offset = loader_pknb.add_offset_bk(cfg, bk, verbose=verbose)
    bk = loader_pknb.k_cuts_bk(cfg, bk, verbose=verbose)
    bk = loader_pknb.normalize_amplitude_bk(cfg, bk, verbose=verbose)
    if cfg.ngals:
        if ngal is None: raise NotImplementedError         
        bk = np.concatenate([bk, np.reshape([ngal], (1, 1, 1))], axis=-1)

    bk = bk[:, 0]
        
    features = np.concatenate([pells, bk], axis=-1)
    features = sbitools.standardize(features, scaler=sweepdict['scaler'], \
                                    log_transform=cfg.logit)[0]
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


def plot_samples(key):

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for i in range(nsims):
        ax[0].hist(samples[key][i][:, 0], alpha=0.5, range=(0.2, 0.5), bins=20)
        ax[1].hist(samples[key][i][:, 4], alpha=0.5, range=(0.55, 0.95), bins=20)
    ax[0].hist(samples[key+'-mean'][:, 0], alpha=0.5, range=(0.2, 0.5), bins=20, histtype='step', color='k', lw=2)
    ax[1].hist(samples[key+'-mean'][:, 4], alpha=0.5, range=(0.55, 0.95), bins=20, histtype='step', color='k', lw=2)

    for axis in ax: axis.grid()
    ax[0].set_xlabel('$\Omega_m$', fontsize=12)
    ax[1].set_xlabel('$\sigma_8$', fontsize=12)
    ax[0].set_xlim(0.2, 0.5)
    ax[1].set_xlim(0.55, 0.95)
    plt.suptitle(key, fontsize=12)
    plt.show()



def get_sweeps():
    sweepdicts = {}
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/Rockstar///z10-N0004/zheng07_velab/pknb-ells024-kmax_bk0.5-kmax_pk0.5-kmin_bk0.005-kmin_pk0.005-ngals-offset_amp_pk10000.0-standardize/8osyakp3/sweep_config_pknb_quijote.yaml'
    sweepdicts['pknb'] = setup_cfg(cfg_path)
    print()
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/Rockstar///z10-N0004/zheng07_velab/pknb-ells024-kmax_bk0.5-kmax_pk0.5-kmin_bk0.005-kmin_pk0.005-ngals-offset_amp_pk10000.0-standardize-train94/lgv5oqw0/sweep_config_pknb_quijote.yaml'
    sweepdicts['pknb-94'] = setup_cfg(cfg_path)
    print()
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/FoF/z10-N0004/zheng07_velab/pknb-ells024-kmax_bk0.5-kmax_pk0.5-kmin_bk0.005-kmin_pk0.005-ngals-offset_amp_pk10000.0-standardize/9u35a910/sweep_config_pknb_quijote.yaml'
    sweepdicts['pknb-fof'] = setup_cfg(cfg_path)
    print()

    ##########
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/Rockstar/z10-N0004/zheng07_velab/ells024-kmax0.5-kmin0.005-ngals-offset_amp10000.0-standardize-train94/5qtqq2hq/sweep_config_hodells_quijote.yaml'
    sweepdicts['pk05-94'] = setup_cfg(cfg_path)
    print()
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/Rockstar/z10-N0004/zheng07_velab/ells024-kmax0.5-kmin0.005-ngals-offset_amp10000.0-standardize/lked4v64/sweep_config_hodells_quijote.yaml'
    sweepdicts['pk05'] = setup_cfg(cfg_path)
    print()
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/Rockstar/z10-N0004/zheng07_velab/ells024-kmax0.3-kmin0.005-ngals-offset_amp10000.0-standardize/tb2ojwm3/sweep_config_hodells_quijote.yaml'
    sweepdicts['pk03'] = setup_cfg(cfg_path)
    print()
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/combine//z10-N0004/zheng07_velab/ells024-kmax0.5-kmin0.005-ngals-offset_amp10000.0-standardize/puqf4oyh/sweep_config_hodells_quijote_combine.yaml'
    sweepdicts['pk05-combine'] = setup_cfg(cfg_path)
    print()
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/FoF/z10-N0004/zheng07_velab/ells024-kmax0.5-kmin0.005-ngals-offset_amp10000.0-standardize/ogm6mii2/sweep_config_hodells_quijote.yaml'
    sweepdicts['pk05-fof'] = setup_cfg(cfg_path)
    print()
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/FoF/z10-N0004/zheng07_velab/ells024-kmax0.5-kmin0.005-ngals-offset_amp10000.0-standardize-train94/3vq2ao0y//sweep_config_hodells_quijote.yaml'
    sweepdicts['pk05-fof-94'] = setup_cfg(cfg_path)
    print()

    ########
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/Rockstar/z10-N0004/zheng07_velab/bk-kmax0.5-kmin0.005-ngals-standardize-train94/894e431j//sweep_config_bspec_quijote.yaml'
    sweepdicts['bk05-94'] = setup_cfg(cfg_path)
    print()
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/Rockstar/z10-N0004/zheng07_velab/bk-kmax0.5-kmin0.005-ngals-standardize/dzoaafd9/sweep_config_bspec_quijote.yaml'
    sweepdicts['bk05'] = setup_cfg(cfg_path)
    print()
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/Rockstar/z10-N0004/zheng07_velab/bk-kmax0.3-kmin0.005-ngals-standardize/uj68xhdv/sweep_config_bspec_quijote.yaml'
    sweepdicts['bk03'] = setup_cfg(cfg_path)
    print()
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/FoF//z10-N0004/zheng07_velab/bk-kmax0.5-kmin0.005-ngals-standardize/uhch7yrv/sweep_config_bspec_quijote.yaml'
    sweepdicts['bk05-fof'] = setup_cfg(cfg_path)
    print()
    # cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/FoF//z10-N0004/zheng07_velab/bk-kmax0.5-kmin0.005-ngals-standardize//uhch7yrv/sweep_config_bspec_quijote.yaml'
    # sweepdicts['bk05-fof-94'] = setup_cfg(cfg_path)
    cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis/quijote/combine//z10-N0004/zheng07_velab/bk-kmax0.5-kmin0.005-ngals-standardize/3pr8tlwj//sweep_config_bspec_quijote_combine.yaml'
    sweepdicts['bk05-combine'] = setup_cfg(cfg_path)
    print()

    return sweepdicts


    
def main():

    allpells = []
    allbks = []
    allngals = []
    print("Load data")
    for i in range(1, 11):
        print(i)
        pells = np.load(f"../data/stats/power_ell_{i}.npy")
        bk = np.load(f"../data/stats/bispec_{i}.npy")
        ngal = np.load(f"../data/stats/ngal_{i}.npy") * 8
        ngal_rescaled = ngal/2**3
        allbks.append(bk[3].copy())
        allpells.append(pells.copy())
        allngals.append([ngal_rescaled])

    allpells = np.array(allpells)
    allbks = np.array(allbks)

    print("Get sweeps")
    sweepdicts = get_sweeps()

    print()
    print("Get features")
    nsims = 10
    verbose = False
    features = {}
    # for key in ['pk05-combine', 'pk05-fof', 'pk05', 'pk03']:
    for key in sweepdicts.keys():
        print(key)
        if  'pknb' in key: 
            features[key] = [setup_features_pknb(allpells[j].copy(), allbks[j].copy(), sweepdicts[key], ngal=allngals[j], verbose=verbose) for j in range(nsims)]
        elif 'pk' in key: 
            features[key] = [setup_features_pells(allpells[j].copy(), sweepdicts[key], ngal=allngals[j], verbose=verbose) for j in range(nsims)]
        elif 'bk' in key:
            features[key] = [setup_features_bispec(allbks[j].copy(), sweepdicts[key], ngal=allngals[j], verbose=verbose) for j in range(nsims)]
        else:
            print('Not implelemnted')

    print()
    print("Get samples now")
    nsims = 10
    nens = 10

    for ens in [10, 20, 5]:
        samples = {}
        for key in sweepdicts.keys():
            print(key)
            samples[key] = [get_samples_ens(features[key][j].copy(), sweepdicts[key], nposterior=nens, verbose=False) \
                        for j in range(nsims)]    
            np.save(f'../data/results/{key}-ens{ens}', samples[key])
            samples[key+'-mean'] = get_samples_ens(np.array(features[key]).mean(axis=0).copy(), sweepdicts[key], nposterior=nens, verbose=False)     
            np.save(f'../data/results/{key}-mean-ens{ens}', samples[key+'-mean'])



if __name__=="__main__":

    main()
