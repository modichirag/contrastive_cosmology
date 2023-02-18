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
import loader_hod_bispec as loader
from folder_path import bispec_path


cfg_data = sys.argv[1]
cfgd_dict = yaml.load(open(f'{cfg_data}'), Loader=yaml.Loader)
cuts = cfgd_dict['datacuts']
args = {}
for i in cfgd_dict.keys():
    args.update(**cfgd_dict[i])
cfgd = sbitools.Objectify(**args)
np.random.seed(cfgd.seed)

#
datapath = f'/mnt/ceph/users/cmodi/contrastive/data/{cfgd.simulation}/{cfgd.finder}/z{int(cfgd.z*10):02d}-N{int(cfgd.nbar/1e-4):04d}/{cfgd.hodmodel}/'

#analysis_path = datapath.replace("data", "analysis")
#if cfgd.reduced: folder = 'qspec/'
#else: folder = 'bspec/'
#cfgd.analysis_path = analysis_path + folder

cfgd.analysis_path = bispec_path(cfgd_dict)
scaler = sbitools.load_scaler(cfgd.analysis_path)
features, params = loader.hod_bispec_lh(datapath, cfgd)
data = sbitools.test_train_split(features, params, train_size_frac=cfgd.train_fraction)
data.trainx = sbitools.standardize(data.trainx, scaler=scaler, log_transform=cfgd.logit)[0]
data.testx = sbitools.standardize(data.testx, scaler=scaler, log_transform=cfgd.logit)[0]

netpath = 'maf-nt5-32-b64-b64-lr0.001/'
model_path = f'{cfgd.analysis_path}/{netpath}'
print(model_path)

nsamples = 2000
posterior = sbitools.load_posterior(model_path)
for i in range(10):
    print(i)
    fig, ax = sbiplots.plot_posterior(data.trainx[i], data.trainy[i], posterior, savename=f'{model_path}/corner-train{i}.png')
    fig, ax = sbiplots.plot_posterior(data.testx[i], data.testy[i], posterior, savename=f'{model_path}/corner{i}.png')
    posterior_samples = posterior.sample((nsamples,), x=torch.from_numpy(data.testx[i].astype('float32'))).detach().numpy()
    np.save(f'{model_path}/samples{i}', posterior_samples)

    
try:
    trues, mus, stds, ranks = sbiplots.get_ranks(data.testx, data.testy, posterior, test_frac=0.05, nsamples=100, ndim=5)
except Exception as e: 
    print('Exception occured : ', e)
    
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


