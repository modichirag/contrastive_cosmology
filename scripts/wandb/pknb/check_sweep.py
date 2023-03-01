'''Create diagnostic figures for individial posteriors (and ensembles) of the sweep
The figures are saved in their respective folder (sweep/figs/) path
'''
import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.append('../../../src/')
sys.path.append('../')
import sbitools, sbiplots
import argparse
import pickle, json
from io import StringIO
import wandb
import yaml
import torch
import loader_hod_pknb as loader
import folder_path
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble


cfg_data = sys.argv[1]
cfgd_dict = yaml.load(open(f'{cfg_data}'), Loader=yaml.Loader)
sweep_id = cfgd_dict['sweep']['id']
cuts = cfgd_dict['datacuts']
args = {}
for i in cfgd_dict.keys():
    args.update(**cfgd_dict[i])
cfgd = sbitools.Objectify(**args)
np.random.seed(cfgd.seed)

#
datapath = f'/mnt/ceph/users/cmodi/contrastive/data/{cfgd.simulation}/{cfgd.finder}/z{int(cfgd.z*10):02d}-N{int(cfgd.nbar/1e-4):04d}/{cfgd.hodmodel}/'

cfgd.analysis_path = folder_path.pknb_path(cfgd_dict)
scaler = sbitools.load_scaler(cfgd.analysis_path)
features, params = loader.hod_pknb_lh(datapath, cfgd)
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
print(idx)
print(list(zip(names, log_prob)))

###
posteriors = []
figpath = f'{cfgd.analysis_path}/{sweep_id}/figs/'
os.makedirs(figpath, exist_ok=True)
test_frac, nsamples = 0.2, 500


for name in names:
    model_path = f'{cfgd.analysis_path}/{sweep_id}/{name}/'
    print(model_path)

    posterior = sbitools.load_posterior(model_path)
    posteriors.append(posterior)
    for i in range(3):
        print(i)
        fig, ax = sbiplots.plot_posterior(data.trainx[i], data.trainy[i], posterior, savename=f'{model_path}/corner-train{i}.png')
        fig, ax = sbiplots.plot_posterior(data.testx[i], data.testy[i], posterior, savename=f'{model_path}/corner{i}.png')

    try:
        trues, mus, stds, ranks = sbiplots.get_ranks(data.testx, data.testy, posterior, test_frac=0.05, nsamples=100, ndim=5)
    except Exception as e: 
        print('Exception occured : ', e)
        continue
    tosave = np.stack([trues, mus, stds, ranks], axis=-1)
    #np.save(f'{model_path}/ranks', tosave)

    fig, ax = sbiplots.plot_coverage(ranks, titles=sbiplots.cosmonames)
    plt.savefig(f'{model_path}/coverage.png')
    plt.close()

    fig, ax = sbiplots.plot_ranks_histogram(ranks, titles=sbiplots.cosmonames)
    plt.savefig(f'{model_path}/rankplot.png')
    plt.close()

    fig, ax = sbiplots.plot_predictions(trues, mus, stds,  titles=sbiplots.cosmonames)
    plt.savefig(f'{model_path}/predictions.png')
    plt.close()


    #Check for ensembles
    ien = len(posteriors)
    #for ien in range(1, nmodels+1):

    name = f'ens{ien}'
    print(f"For ensemble : {name}")
    posterior = NeuralPosteriorEnsemble(posteriors=posteriors[:ien])

    trues, mus, stds, ranks = sbiplots.get_ranks(data.testx, data.testy, posterior, test_frac=test_frac, nsamples=nsamples, ndim=5)
    tosave = np.stack([trues, mus, stds, ranks], axis=-1)
    #np.save(f'{figpath}/data/ranks-{name}', tosave)
    
    fig, ax = sbiplots.plot_coverage(ranks, titles=sbiplots.cosmonames)
    plt.savefig(f'{figpath}/coverage-{name}.png')
    plt.close()
    
    fig, ax = sbiplots.plot_ranks_histogram(ranks, titles=sbiplots.cosmonames)
    plt.savefig(f'{figpath}/rankplot-{name}.png')
    plt.close()
    
    fig, ax = sbiplots.plot_predictions(trues, mus, stds,  titles=sbiplots.cosmonames)
    plt.savefig(f'{figpath}/predictions-{name}.png')
    plt.close()
#except: continue

