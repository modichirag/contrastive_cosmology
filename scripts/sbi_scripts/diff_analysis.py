import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../src/')
import sbitools, sbiplots
import argparse
import pickle, json
import dataloaders
import torch
#from dataclasses import dataclass
import copy

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--modelpath', type=str, help='modelpath in contrastive analysis folder')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999, default=0')
parser.add_argument('--embed', type=int, default=0, help='embed features')
args = parser.parse_args()

np.random.seed(args.seed)
#

modelpath = "/mnt/ceph/users/cmodi/contrastive/analysis/" + args.modelpath + '/'
z, nbar = args.modelpath.split('/')[0].split('-')
z = float(z[1:])/10.
nbar = float(nbar[1:])*1e-4
print(z, nbar)


with open(modelpath + 'args.json') as f:
    args_model = json.load(f)
print(args_model)
args_model = sbitools.Objectify(**args_model)
print(args_model)

dataloader = getattr(dataloaders, args_model.dataloader)
scaler = sbitools.load_scaler(modelpath)
posterior = sbitools.load_posterior(modelpath)


def get_features(datapath, fiducial=0, embed=False):
    argsf = copy.deepcopy(args_model)
    argsf.datapath = datapath
    print("\n"+argsf.datapath)
    argsf.fiducial = fiducial
    #
    features, params = dataloader(argsf, testing=True)
    nsim = features.shape[1]
    features = features.reshape(-1, features.shape[-1])
    params = params.reshape(-1, params.shape[-1])
    if argsf.standardize: 
        features = sbitools.standardize(features, scaler=scaler, log_transform=argsf.logit)[0]

    if embed:
        try:
            model = torch.load(modelpath + 'model.pb')
            print("Embedding model loaded")
            device = 'cpu'
            features = sbitools.embed_data(features, model.encoder, device=device)
            print("Features compressed")

        except Exception as e:
            print('### Exception occured ###\n', e)
            print('Features are not compressed')
            
    return features, params



#suffixes = ['', '_ab', '_velab', '-rock', '_ab-rock', '_velab-rock']
suffixes = ['-rock', '_ab-rock', '_velab-rock']
os.makedirs(modelpath + 'diagnostics/', exist_ok=True)
cosmonames = sbiplots.cosmonames

for i, suffix in enumerate(suffixes):
    
    datapath = args.modelpath.split('/')[0] + '/zheng07' + suffix + '/'
    features, params = get_features(datapath, embed=bool(args.embed))
    if not 'rock' in suffix:  suffix = suffix + '-fof'
    suffix = suffix[1:]

    for _ in range(5):
        ii = np.random.randint(0, features.shape[0], 1)[0]
        savename = modelpath + 'diagnostics/posterior-%s%04d.png'%(suffix, ii/10)
        fig, ax = sbiplots.plot_posterior(features[ii], params[ii], posterior,
                                          titles=cosmonames, savename=savename, ndim=5)


    try:
        trues, mus, stds, ranks = sbiplots.get_ranks(features, params, posterior, test_frac=0.05, nsamples=100, ndim=5)
    except Exception as e: 
        print('Exception occured : ', e)
        continue
    tosave = np.stack([trues, mus, stds, ranks], axis=-1)
    np.save(modelpath + 'diagnostics/ranks-%s'%suffix, tosave)

    fig, ax = sbiplots.plot_coverage(ranks, titles=cosmonames)
    plt.savefig(modelpath + 'diagnostics/coverage-%s'%suffix)
    plt.close()

    fig, ax = sbiplots.plot_ranks_histogram(ranks, titles=cosmonames)
    plt.savefig(modelpath + 'diagnostics/rankplot-%s'%suffix)
    plt.close()

    fig, ax = sbiplots.plot_predictions(trues, mus, stds,  titles=cosmonames)
    plt.savefig(modelpath + 'diagnostics/prediction-%s'%suffix)
    plt.close()
