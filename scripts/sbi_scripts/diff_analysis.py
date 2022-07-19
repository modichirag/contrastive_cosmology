import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../src/')
import sbitools, sbiplots
import argparse
import pickle, json
import dataloaders
import torch
from dataclasses import dataclass

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--datapath', type=str, help='datapath in contrastive data folder')
parser.add_argument('--modelpath', type=str, help='modelpath in contrastive analysis folder')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999, default=0')
parser.add_argument('--suffix', type=str, default="test", help='suffix, default=""')
parser.add_argument('--fiducial', type=int, default=0, help='running for fiducial simulations')
args = parser.parse_args()

np.random.seed(args.seed)
#

modelpath = "/mnt/ceph/users/cmodi/contrastive/analysis/" + args.modelpath
datapath = "/mnt/ceph/users/cmodi/contrastive/data/" + args.datapath

with open(modelpath + 'args.json') as f:
    args_model = json.load(f)
print(args_model)


class Objectify(object):
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

args_model = Objectify(**args_model)
print(args_model)

dataloader = getattr(dataloaders, args_model.dataloader)


def diagnostics(posterior, scaler):
    print("Diagnostics for rockstar data")
    import copy
    argsf = copy.deepcopy(args_model)
    argsf.datapath = args.datapath
    print(argsf.datapath)
    features, params = dataloader(argsf)
    nsim = features.shape[1]
    features = features.reshape(-1, features.shape[-1])
    params = params.reshape(-1, params.shape[-1])
    features = sbitools.standardize(features, scaler=scaler, log_transform=argsf.logit)[0]

    cosmonames = r'$\Omega_m$,$\Omega_b$,$h$,$n_s$,$\sigma_8$'.split(",")
    cosmonames = cosmonames + ["Mcut", "sigma", "M0", "M1", "alpha"]
    if params.shape[-1] > len(cosmonames): 
        cosmonames = cosmonames + ["a_%d"%i for i in range(params.shape[-1] - len(cosmonames))]

    for _ in range(argsf.nposterior):
        ii = np.random.randint(0, features.shape[0], 1)[0]
        savename = modelpath + 'posterior-%s%04d.png'%(args.suffix, ii/nsim)
        fig, ax = sbiplots.plot_posterior(features[ii], params[ii], posterior, titles=cosmonames, savename=savename, ndim=5)
    sbiplots.test_diagnostics(features, params, posterior, titles=cosmonames, savepath=modelpath, test_frac=0.05, nsamples=500, suffix='-%s'%args.suffix)

scaler = sbitools.load_scaler(modelpath)
posterior = sbitools.load_posterior(modelpath)
diagnostics(posterior, scaler)

