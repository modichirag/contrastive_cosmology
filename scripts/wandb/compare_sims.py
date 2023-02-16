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
import folder_path
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

cfg_data = sys.argv[1]
cfg_model = sys.argv[2]
cfgd_dict = yaml.load(open(f'{cfg_data}'), Loader=yaml.Loader)
cfgm_dict = yaml.load(open(f'{cfg_model}'), Loader=yaml.Loader)
sweep_id = cfgm_dict['sweep']['id'] # Sweep id is taken from model config
cuts = cfgd_dict['datacuts']
#check configs
for key in cfgd_dict['data'].keys():
    if key != 'simulation': assert cfgd_dict['data'][key] == cfgm_dict['data'][key]
for key in cfgd_dict['datacuts'].keys():
    assert cfgd_dict['datacuts'][key] == cfgm_dict['datacuts'][key]
print("Both the config files are compatible")

args, args2 = {}, {}
for i in cfgd_dict.keys():
    args.update(**cfgd_dict[i])
    args2.update(**cfgm_dict[i])
cfgd = sbitools.Objectify(**args)
cfgm = sbitools.Objectify(**args2)
np.random.seed(cfgd.seed)

# Datapath is taken from data config
datapath = f'/mnt/ceph/users/cmodi/contrastive/data/{cfgd.simulation}/{cfgd.finder}/z{int(cfgd.z*10):02d}-N{int(cfgd.nbar/1e-4):04d}/{cfgd.hodmodel}/'
analysis_path = datapath.replace("data", "analysis")

#load the files from the data folder
features, params = loader.hod_ells_lh(datapath, cfgm) # datacuts are used from model config
data = sbitools.test_train_split(features, params, train_size_frac=cfgd.train_fraction)
figpath = f'./diagnostics/{cfgd.simulation}/{cfgm.simulation}-{sweep_id}/'
os.makedirs(figpath, exist_ok=True)
#os.makedirs(f'{figpath}/figs/', exist_ok=True)
os.makedirs(f'{figpath}/data/', exist_ok=True)

#however use the scaler from the model
cfgm.analysis_path = folder_path.hodells_path(cfgm_dict)
scaler = sbitools.load_scaler(cfgm.analysis_path)
data.trainx = sbitools.standardize(data.trainx, scaler=scaler, log_transform=cfgm.logit)[0]
data.testx = sbitools.standardize(data.testx, scaler=scaler, log_transform=cfgm.logit)[0]


#load model
api = wandb.Api()
sweep = api.sweep(f'modichirag92/quijote-hodells/{sweep_id}')
sweep_runs = sweep.runs

#sort in the order of validation log prob
names, log_prob = [], []
for run in sweep_runs:
    if run.state == 'finished':
        print(run.name, run.summary['best_validation_log_prob'])
        model_path = run.summary['output_directory']
        if (bool(1+model_path.find(cfgm.analysis_path))):
            names.append(run.name)
            log_prob.append(run.summary['best_validation_log_prob'])
        else:
            print("inconsistency in analysis path and model path")
            sys.exit()

idx = np.argsort(log_prob)[::-1]
names = np.array(names)[idx]
log_prob = np.array(log_prob)[idx]


####################################
#Check for individual models

nmodels = 10
test_frac = 0.1
nsamples = 200

posteriors = []
for name in names[:nmodels]:
    model_path = f'{cfgm.analysis_path}/{sweep_id}/{name}/'
    print(model_path)

    posterior = sbitools.load_posterior(model_path)
    posteriors.append(posterior)
    
    # for i in range(3):
    #     print(i)
    #     fig, ax = sbiplots.plot_posterior(data.testx[i], data.testy[i], posterior, savename=f'{figpath}/corner-{name}{i}.png')

    # try:
    #     trues, mus, stds, ranks = sbiplots.get_ranks(data.testx, data.testy, posterior, test_frac=test_frac, nsamples=nsamples, ndim=5)
    #     tosave = np.stack([trues, mus, stds, ranks], axis=-1)
    #     np.save(f'{figpath}/data/ranks-{name}', tosave)
        
    #     fig, ax = sbiplots.plot_coverage(ranks, titles=sbiplots.cosmonames)
    #     plt.savefig(f'{figpath}/coverage-{name}.png')
    #     plt.close()
        
    #     fig, ax = sbiplots.plot_ranks_histogram(ranks, titles=sbiplots.cosmonames)
    #     plt.savefig(f'{figpath}/rankplot-{name}.png')
    #     plt.close()
        
    #     fig, ax = sbiplots.plot_predictions(trues, mus, stds,  titles=sbiplots.cosmonames)
    #     plt.savefig(f'{figpath}/predictions-{name}.png')
    #     plt.close()
        
    # except: continue
    
   

    #Check for ensembles
    ien = len(posteriors)
    #for ien in range(1, nmodels+1):

    name = f'ens{ien}'
    print(f"For ensemble : {name}")
    posterior = NeuralPosteriorEnsemble(posteriors=posteriors[:ien])

    try:
        trues, mus, stds, ranks = sbiplots.get_ranks(data.testx, data.testy, posterior, test_frac=test_frac, nsamples=nsamples, ndim=5)
        tosave = np.stack([trues, mus, stds, ranks], axis=-1)
        np.save(f'{figpath}/data/ranks-{name}', tosave)
        
        fig, ax = sbiplots.plot_coverage(ranks, titles=sbiplots.cosmonames)
        plt.savefig(f'{figpath}/coverage-{name}.png')
        plt.close()
        
        fig, ax = sbiplots.plot_ranks_histogram(ranks, titles=sbiplots.cosmonames)
        plt.savefig(f'{figpath}/rankplot-{name}.png')
        plt.close()
        
        fig, ax = sbiplots.plot_predictions(trues, mus, stds,  titles=sbiplots.cosmonames)
        plt.savefig(f'{figpath}/predictions-{name}.png')
        plt.close()
    except: continue

