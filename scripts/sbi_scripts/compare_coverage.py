import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../src/')
import sbitools, sbiplots
import argparse
import pickle, json
import dataloaders
import torch
import copy

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--modelpath', type=str, help='modelpath in contrastive analysis folder')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999, default=0')
parser.add_argument('--suffix', type=str, default="test", help='suffix, default=""')
parser.add_argument('--fiducial', type=int, default=0, help='running for fiducial simulations')
args = parser.parse_args()

np.random.seed(args.seed)
#

modelpath = "/mnt/ceph/users/cmodi/contrastive/analysis/" + args.modelpath
basepath = "/mnt/ceph/users/cmodi/contrastive/data/"

with open(modelpath + 'args.json') as f:
    args_model = json.load(f)
print(args_model)
args_model = sbitools.Objectify(**args_model)
print(args_model)

dataloader = getattr(dataloaders, args_model.dataloader)
scaler = sbitools.load_scaler(modelpath)
posterior = sbitools.load_posterior(modelpath)
del args


def get_features(datapath, fiducial=0):
    argsf = copy.deepcopy(args_model)
    argsf.datapath = datapath
    print(argsf.datapath)
    argsf.fiducial = fiducial
    #
    features, params = dataloader(argsf)
    nsim = features.shape[1]
    features = features.reshape(-1, features.shape[-1])
    params = params.reshape(-1, params.shape[-1])
    features = sbitools.standardize(features, scaler=scaler, log_transform=argsf.logit)[0]

    return features, params


############
cosmonames = r'$\Omega_m$,$\Omega_b$,$h$,$n_s$,$\sigma_8$'.split(",")
cosmonames = cosmonames + ["Mcut", "sigma", "M0", "M1", "alpha"]
dfrac, nsamples = 0.05, 500

#
fig, ax = plt.subplots(1, 5, figsize=(15, 4))

datapath =  'z10-N0001/zheng07/'
features, params = get_features(datapath)
trues, mus, stds, ranks = sbiplots.get_ranks(features, params, posterior, test_frac=dfrac, nsamples=nsamples)
fig, ax = sbiplots.plot_coverage(ranks, figure=[fig, ax], titles=cosmonames, label='FOF')

datapath =  'z10-N0001/zheng07_ab/'
features, params = get_features(datapath)
trues, mus, stds, ranks = sbiplots.get_ranks(features, params, posterior, test_frac=dfrac, nsamples=nsamples)
fig, ax = sbiplots.plot_coverage(ranks, figure=[fig, ax], titles=cosmonames, label='FOF-ab', plotscatter=False)

datapath =  'z10-N0001/zheng07-rock/'
features, params = get_features(datapath)
trues, mus, stds, ranks = sbiplots.get_ranks(features, params, posterior, test_frac=dfrac, nsamples=nsamples)
fig, ax = sbiplots.plot_coverage(ranks, figure=[fig, ax], titles=cosmonames, label='Rockstar', plotscatter=False)

datapath =  'z10-N0001/zheng07_ab-rock/'
features, params = get_features(datapath)
trues, mus, stds, ranks = sbiplots.get_ranks(features, params, posterior, test_frac=dfrac, nsamples=nsamples)
fig, ax = sbiplots.plot_coverage(ranks, figure=[fig, ax], titles=cosmonames, label='Rockstar-ab', plotscatter=False)

datapath =  'z10-N0001/zheng07-fid/'
features, params = get_features(datapath)
trues, mus, stds, ranks = sbiplots.get_ranks(features, params, posterior, test_frac=dfrac, nsamples=nsamples)
fig, ax = sbiplots.plot_coverage(ranks, figure=[fig, ax], titles=cosmonames, label='FOF (fid)', plotscatter=False)

ax[0].legend()

plt.savefig(modelpath + 'compare_coverage.py')

#     for _ in range(argsf.nposterior):
#         ii = np.random.randint(0, features.shape[0], 1)[0]
#         savename = modelpath + 'posterior-%s%04d.png'%(args.suffix, ii/nsim)
#         fig, ax = sbiplots.plot_posterior(features[ii], params[ii], posterior, titles=cosmonames, savename=savename, ndim=5)
#     sbiplots.test_diagnostics(features, params, posterior, titles=cosmonames, savepath=modelpath, test_frac=0.05, nsamples=500, suffix='-%s'%args.suffix)

# diagnostics(posterior, scaler)

