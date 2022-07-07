import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../src/')
import sbitools, sbiplots
import argparse
import pickle
import dataloaders
import torch

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--datapath', type=str, help='datapath in contrastive data folder')
parser.add_argument('--modelpath', type=str, help='modelpath in contrastive analysis folder')
parser.add_argument('--kmax', type=float, default=0.5, help='kmax, default=0.5')
parser.add_argument('--kmin', type=float, default=0.005, help='kmin, default=0.005')
parser.add_argument('--nlayers', type=int, default=5, help='number of layers, default=5')
parser.add_argument('--nhidden', type=int, default=32, help='number of hddden params, default=32')
parser.add_argument('--model', type=str, default="maf", help='model, one of maf, nsf or mdn, default=maf')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999, default=0')
parser.add_argument('--batch', type=int, default=128, help='batch size, default=128')
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')
parser.add_argument('--fithod', type=int, default=1, help='fit HOD params, default=1')
parser.add_argument('--nposterior', type=int, default=10, help='how many posteriors to plot, default=10')
parser.add_argument('--ampnorm', type=int, default=0, help='normalize at large scales by amplitude, default=0')
parser.add_argument('--ells', type=str, default="024", help='which ells')
parser.add_argument('--fiducial', type=int, default=0, help='running for fiducial simulations')
args = parser.parse_args()

np.random.seed(args.seed)
#

modelpath = "/mnt/ceph/users/cmodi/contrastive/analysis/" + args.modelpath
datapath = "/mnt/ceph/users/cmodi/contrastive/data/" + args.datapath
path = datapath.replace("data", "analysis")
folder = "%s-kmax%02d-nl%02d-nh%02d-s%03d-ells/"%(args.model, args.kmax*100, args.nlayers, args.nhidden, args.seed)
if args.suffix != "": 
    folder = folder[:-1] + "-%s/"%args.suffix 
savepath = path + folder[:-1] + "-%s/"%args.modelpath.split("/")[1]
modelpath = modelpath + folder
os.makedirs(savepath, exist_ok=True)
print("\nWorking directory : ", savepath)
#####

features, params = dataloaders.hod_ells(args)
features = features.reshape(-1, features.shape[-1])
params = params.reshape(-1, params.shape[-1])
scaler = sbitools.load_scaler(modelpath)
features = sbitools.standardize(features, scaler=scaler, log_transform=True)[0]
print(features.shape)
posterior = sbitools.load_posterior(modelpath)
print(posterior)


cosmonames = r'$\Omega_m$,$\Omega_b$,$h$,$n_s$,$\sigma_8$'.split(",")
cosmonames = cosmonames + ["Mcut", "sigma", "M0", "M1", "alpha"]
for _ in range(args.nposterior):
    ii = np.random.randint(0, features.shape[0], 1)[0]
    savename = savepath + 'posterior%04d.png'%(ii//10)
    fig, ax = sbiplots.plot_posterior(features[ii], params[ii], posterior, titles=cosmonames, savename=savename, ndim=10)

if args.fiducial:
    sbiplots.test_fiducial(features, params, posterior, titles=cosmonames, savepath=savepath, \
                              test_frac=0.5, nsamples=500)
else:
    sbiplots.test_diagnostics(features, params, posterior, titles=cosmonames, savepath=savepath, \
                              test_frac=0.05, nsamples=500)
