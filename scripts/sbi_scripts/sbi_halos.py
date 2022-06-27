import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../src/')
import sbitools, sbiplots
import argparse
import pickle

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--kmax', type=float, default=0.5, help='kmax')
parser.add_argument('--nlayers', type=int, default=5, help='kmax')
parser.add_argument('--nhidden', type=int, default=32, help='kmax')
parser.add_argument('--model', type=str, default="maf", help='model, one of maf, nsf or mdn')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999')
args = parser.parse_args()

np.random.seed(args.seed)
path = "/mnt/ceph/users/cmodi/contrastive/analysis/halos/"
folder = "%s-kmax%02d-nl%02d-nh%02d-s%03d/"%(args.model, args.kmax*100, args.nlayers, args.nhidden, args.seed)
savepath = path + folder
os.makedirs(savepath, exist_ok=True)
###
params, params_fid, cosmonames = sbitools.quijote_params()
ndim = len(params_fid)
prior = sbitools.sbi_prior(params, offset=0.25)

#####
### Dataloader
pk = np.load('/mnt/ceph/users/cmodi/contrastive/data/z01-N0001/zheng07/power_h1e-4.npy')
k = np.load('/mnt/ceph/users/cmodi/contrastive/data/z01-N0001/zheng07/0000/power_h1e-4.npy')[:, 0]
#pk = np.load('../../sbi-scattering/data/quijote/latin_hypercube/pkmatter.npy')
#pk_fid = np.load('../../sbi-scattering/data/quijote/fiducial/pkmatter.npy')[..., 1]
#
ik05 = np.where(k>args.kmax)[0][0]
k = k[1:ik05]
pk = pk[:, 1:ik05]
#
    

###test_tain_split
train, test, tidx = sbitools.test_train_split(pk, params, train_size_frac=0.8)
trainx, trainy = train
testx, testy = test
trainx, scaler = sbitools.standardize(trainx, log_transform=True)
testx, _ = sbitools.standardize(testx, log_transform=True, scaler=scaler)

try:
    print("Load an existing posterior model")
    with open(savepath + "posterior.pkl", "rb") as handle:
        posterior = pickle.load(handle)
except Exception as e:
    print("##Exception##\n", e)
    print("Training a new NF")
    inference, density_estimator, posterior = sbitools.sbi(trainx, trainy, prior, \
                                            model=args.model, nlayers=args.nlayers, \
                                            nhidden=args.nhidden)
    with open(savepath + "posterior.pkl", "wb") as handle:
        pickle.dump(posterior, handle)

    with open(savepath + "inference.pkl", "wb") as handle:
        pickle.dump(inference, handle)


###Diagnostics
ii = np.random.randint(0, testx.shape[0], 1)[0]
fig, ax = sbiplots.plot_posterior(testx[ii], testy[ii], posterior, titles=cosmonames)
plt.savefig(savepath + 'posterior%04d.png'%(tidx[1][ii]))
sbiplots.test_diagnostics(testx, testy, posterior, titles=cosmonames, savepath=savepath)

