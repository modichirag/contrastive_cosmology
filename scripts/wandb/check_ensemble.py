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

cfg_data = sys.argv[1]
cfgd = yaml.load(open(f'{cfg_data}'), Loader=yaml.Loader)
sweep_id = cfgd['sweep']['id']
cuts = cfgd['datacuts']
args = {}
for i in cfgd.keys():
    args.update(**cfgd[i])
cfgd = sbitools.Objectify(**cfgd)
cfgd = sbitools.Objectify(**args)
np.random.seed(cfgd.seed)

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

cfgd.analysis_path = analysis_path + folder
scaler = sbitools.load_scaler(cfgd.analysis_path)
features, params = loader.hod_ells_lh(datapath, cfgd)
data = sbitools.test_train_split(features, params, train_size_frac=cfgd.train_fraction)
data.trainx = sbitools.standardize(data.trainx, scaler=scaler, log_transform=cfgd.logit)[0]
data.testx = sbitools.standardize(data.testx, scaler=scaler, log_transform=cfgd.logit)[0]

api = wandb.Api()
sweep = api.sweep(f'modichirag92/quijote-hodells/{sweep_id}')
sweep_runs = sweep.runs

names, log_prob = [], []
for run in sweep_runs:
    if run.state == 'finished':
        print(run.name, run.summary['best_validation_log_prob'])
        model_path = run.summary['output_directory']
        if (bool(1+model_path.find(cfgd.analysis_path))):
            names.append(run.name)
            log_prob.append(run.summary['best_validation_log_prob'])
        else:
            print("inconsistency in analysis path and model path")
            sys.exit()

idx = np.argsort(log_prob)[::-1]
names = np.array(names)[idx]
log_prob = np.array(log_prob)[idx]
#print(idx)
#print(list(zip(names, log_prob)))


posteriors = []
for iname, name in enumerate(names):
    model_path = f'{cfgd.analysis_path}/{sweep_id}/{name}/'
    posteriors.append(sbitools.load_posterior(model_path))

    

#figpath = f'{cfgd.analysis_path}/{sweep_id}/figs'
#os.makedirs(figpath)

# for ien in [10]:
#     ensemble = NeuralPosteriorEnsemble(posteriors=posteriors[:ien])

    
#     fig, ax = plt.subplots(1, 2, figsize=(9, 4))

#     x = data.testx[0].copy()
#     posterior_samples = ensemble.sample((500,), x=torch.from_numpy(x.astype('float32'))).detach().numpy()
#     ax[0].hist(posterior_samples[:, 0], histtype='step', color='k', lw=2, density=True)
#     ax[1].hist(posterior_samples[:, 4], histtype='step', color='k', lw=2, density=True)    
#     for i in range(ien):
#         posterior_samples = posteriors[i].sample((500,), x=torch.from_numpy(x.astype('float32'))).detach().numpy()
#         ax[0].hist(posterior_samples[:, 0], alpha=0.2, density=True)
#         ax[1].hist(posterior_samples[:, 4], alpha=0.2, density=True)
        
#     plt.savefig('tmp.png')



# fig, ax = plt.subplots(1, 2, figsize=(9, 4))
# for ien in range(1, 5):

#     ensemble = NeuralPosteriorEnsemble(posteriors=posteriors[:ien])
    
#     x = data.testx[0].copy()
#     posterior_samples = ensemble.sample((500,), x=torch.from_numpy(x.astype('float32'))).detach().numpy()
#     ax[0].hist(posterior_samples[:, 0], histtype='step', lw=2, density=True)
#     ax[1].hist(posterior_samples[:, 4], histtype='step', lw=2, density=True)
        
# plt.savefig('tmp2.png')
    


oneranks = []
enranks = []
nbins = 10
for ien in range(1, 5):

    ensemble = NeuralPosteriorEnsemble(posteriors=posteriors[:ien])
    ranks = sbiplots.get_ranks(data.testx, data.testy, ensemble, test_frac=0.05, nsamples=100, ndim=5)[3]
    enranks.append(ranks)
    ranks = sbiplots.get_ranks(data.testx, data.testy, posteriors[ien], test_frac=0.05, nsamples=100, ndim=5)[3]
    oneranks.append(ranks)
    ncounts = ranks.shape[0]/nbins

    
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for i in range(ien):
        ax[0].hist(oneranks[i][:, 0], bins=nbins, alpha=0.5, histtype='step', lw=2)
        ax[1].hist(oneranks[i][:, 4], bins=nbins, alpha=0.5, histtype='step', lw=2)
    for axis in ax:
        axis.grid(which='both')
        axis.axhline(ncounts, color='k')
        axis.axhline(ncounts - ncounts**0.5, color='k', ls="--")
        axis.axhline(ncounts + ncounts**0.5, color='k', ls="--")
    plt.savefig('tmpranks.png')
    plt.close()
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    for i in range(ien):
        ax[0].hist(enranks[i][:, 0], bins=nbins, alpha=0.5, histtype='step', lw=2)
        ax[1].hist(enranks[i][:, 4], bins=nbins, alpha=0.5, histtype='step', lw=2)
    for axis in ax:
        axis.grid(which='both')
        axis.axhline(ncounts, color='k')
        axis.axhline(ncounts - ncounts**0.5, color='k', ls="--")
        axis.axhline(ncounts + ncounts**0.5, color='k', ls="--")
    plt.savefig('tmpranks2.png')
    plt.close()


