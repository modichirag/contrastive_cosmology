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
print(idx)
print(list(zip(names, log_prob)))

for name in names:
    model_path = f'{cfgd.analysis_path}/{sweep_id}/{name}/'
    print(model_path)

    posterior = sbitools.load_posterior(model_path)
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
    np.save(f'{model_path}/ranks', tosave)

    fig, ax = sbiplots.plot_coverage(ranks, titles=sbiplots.cosmonames)
    plt.savefig(f'{model_path}/coverage.png')
    plt.close()

    fig, ax = sbiplots.plot_ranks_histogram(ranks, titles=sbiplots.cosmonames)
    plt.savefig(f'{model_path}/rankplot.png')
    plt.close()

    fig, ax = sbiplots.plot_predictions(trues, mus, stds,  titles=sbiplots.cosmonames)
    plt.savefig(f'{model_path}/predictions.png')
    plt.close()


