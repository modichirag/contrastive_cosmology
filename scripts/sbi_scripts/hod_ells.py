import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../src/')
import sbitools, sbiplots
import argparse
import pickle

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--datapath', type=str, help='datapath in contrastive data folder')
parser.add_argument('--kmax', type=float, default=0.5, help='kmax, default=0.5')
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
args = parser.parse_args()

np.random.seed(args.seed)
#

datapath = "/mnt/ceph/users/cmodi/contrastive/data/" + args.datapath
path = datapath.replace("data", "analysis")
folder = "%s-kmax%02d-nl%02d-nh%02d-s%03d-ells/"%(args.model, args.kmax*100, args.nlayers, args.nhidden, args.seed)
if args.suffix != "": 
    folder = folder[:-1] + "-%s/"% args.suffix 
savepath = path + folder
os.makedirs(savepath, exist_ok=True)
#####

#####
### Dataloader
def dataloader():
    pk = np.load(datapath + '/power_ell.npy')
    pk += 1e8 #add offset to make it positive
    print("Shape of raw power spectra : ", pk.shape)
    k = np.load('/mnt/ceph/users/cmodi/contrastive/data/z05-chang/zheng07//0000/power_0.npy')[:, 0]
    ngal = np.load(datapath + '/gals.npy')[..., 0]
    nhod = pk.shape[1]
    print("Number of HODs per cosmology : ", nhod)
    #
    ik05 = np.where(k>args.kmax)[0][0]
    pk = pk[..., 1:ik05, :]
    
    #pk = np.swapaxes(pk, 2, 3)
    #pk = pk.reshape([pk.shape[0], pk.shape[1], -1])
    #pk = pk[..., 0]
    #if args.ampnorm: pk /= pk[..., 1:2] # Normalize at large sacles

    if args.ampnorm: pk /= pk[..., 1:2, :] # Normalize at large sacles
    if len(args.ells) > 1:
        if args.ells == "02": pk = pk[..., [0, 1]]
        if args.ells == "04": pk = pk[..., [0, 2]]
        if args.ells == "24": pk = pk[..., [1, 2]]
        if args.ells == "024": pass
        pk = np.swapaxes(pk, 2, 3)
        pk = pk.reshape([pk.shape[0], pk.shape[1], -1])
    else:
        if args.ells == "0": pk = pk[..., 0]
        if args.ells == "2": pk = pk[..., 1]
        if args.ells == "4": pk = pk[..., 2]


    
    features = np.concatenate([pk, np.expand_dims(ngal, -1)], axis=-1)
    print("Features shape : ", features.shape)
    #
    cosmo_params = sbitools.quijote_params()[0]
    ncosmop = cosmo_params.shape[-1]
    cosmo_params = np.repeat(cosmo_params, nhod, axis=0).reshape(-1, nhod, ncosmop)
    if args.fithod == 1: 
        hod_params = np.load(datapath + 'hodp.npy')
        nhodp = hod_params.shape[-1]
        hod_params = sbitools.minmax(hod_params.reshape(-1, nhodp), log_transform=False)[0]
        hod_params = hod_params.reshape(-1, nhod, nhodp)
        params = np.concatenate([cosmo_params, hod_params], axis=-1)
    else: 
        params = cosmo_params
    print("Parameters shape : ", params.shape)

    return features, params

#
### test_tain_split
features, params = dataloader()
train, test, tidx = sbitools.test_train_split(features, params, train_size_frac=0.8)
trainx, trainy = train
testx, testy = test

### Standaradize
trainx, testx, scaler = sbitools.standardize(trainx, secondary=testx, log_transform=True)
with open(savepath + "scaler.pkl", "wb") as handle:
    pickle.dump(scaler, handle)

print(scaler.mean_)
print(scaler.scale_)

#############
### SBI
prior = sbitools.sbi_prior(params.reshape(-1, params.shape[-1]), offset=0.1)
try:
    print("Load an existing posterior model")
    #raise Exception
    with open(savepath + "posterior.pkl", "rb") as handle:
        posterior = pickle.load(handle)
except Exception as e:
    print("##Exception##\n", e)
    print("Training a new NF")
    inference, density_estimator, posterior = sbitools.sbi(trainx, trainy, prior, \
                                            model=args.model, nlayers=args.nlayers, \
                                                           nhidden=args.nhidden, batch_size=args.batch)
    with open(savepath + "posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)

    with open(savepath + "inference.pkl", "wb") as handle:
        pickle.dump(inference, handle)

### Diagnostics
cosmonames = r'$\Omega_m$,$\Omega_b$,$h$,$n_s$,$\sigma_8$'.split(",")
cosmonames = cosmonames + ["Mcut", "sigma", "M0", "M1", "alpha"]
for _ in range(args.nposterior):
    ii = np.random.randint(0, testx.shape[0], 1)[0]
    fig, ax = sbiplots.plot_posterior(testx[ii], testy[ii], posterior, titles=cosmonames)
    plt.savefig(savepath + 'posterior%04d.png'%(tidx[1][ii//params.shape[1]]))
sbiplots.test_diagnostics(testx, testy, posterior, titles=cosmonames, savepath=savepath, test_frac=0.2, nsamples=500)
