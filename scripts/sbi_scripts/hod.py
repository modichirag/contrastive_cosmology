import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../src/')
import sbitools, sbiplots
import argparse
import pickle

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--kmax', type=float, default=0.5, help='kmax')
parser.add_argument('--nlayers', type=int, default=5, help='kmax')
parser.add_argument('--nhidden', type=int, default=32, help='kmax')
parser.add_argument('--model', type=str, default="maf", help='model, one of maf, nsf or mdn')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999')
parser.add_argument('--batch', type=int, default=128, help='batch size')
parser.add_argument('--suffix', type=str, default="", help='batch size')
parser.add_argument('--fithod', type=int, default=1, help='fit HOD params')
parser.add_argument('--nposterior', type=int, default=10, help='how many posteriors to plot')
parser.add_argument('--datapath', type=str, default="", help='datapath in contrastive data folder')
args = parser.parse_args()

nhod = 10
np.random.seed(args.seed)
#
if args.datapath == "": datapath = "/mnt/ceph/users/cmodi/contrastive/data/z05-chang/zheng07/"
else: datapath = "/mnt/ceph/users/cmodi/contrastive/data/" + args.datapath
path = datapath.replace("data", "analysis")
folder = "%s-kmax%02d-nl%02d-nh%02d-s%03d/"%(args.model, args.kmax*100, args.nlayers, args.nhidden, args.seed)
if args.suffix != "": folder = folder[:-1] + "-%s/"% args.suffix 
savepath = path + folder
os.makedirs(savepath, exist_ok=True)
###
params, params_fid, cosmonames = sbitools.quijote_params()
cosmonames = cosmonames + ["Mcut", "sigma", "M0", "M1", "alpha"]
ndim = len(params_fid)
hod_params = np.load(datapath + 'hodp.npy')
nhodp = hod_params.shape[-1]
hod_params = sbitools.minmax(hod_params.reshape(-1, nhodp), log_transform=False)[0]
if args.fithod == 1: all_params = np.hstack([np.repeat(params, 10, axis=0), hod_params])
else: all_params = np.repeat(params, 10, axis=0)
prior = sbitools.sbi_prior(all_params, offset=0.1)
hod_params = hod_params.reshape(-1, nhod, nhodp)

#####
### Dataloader
pk = np.load(datapath + '/power.npy')
k = np.load(datapath + '/0000/power_0.npy')[:, 0]
ngal = np.load(datapath + '/gals.npy')[..., 0]
#pk_fid = np.load('../../sbi-scattering/data/quijote/fiducial/pkmatter.npy')[..., 1]
#
ik05 = np.where(k>args.kmax)[0][0]
k = k[1:ik05]
pk = pk[..., 1:ik05]
#
print(params.shape, hod_params.shape, pk.shape)

###test_tain_split
train, test, tidx = sbitools.test_train_split(pk, params, train_size_frac=0.8)
trainx, trainy = train
testx, testy = test
trainx = trainx.reshape(-1, trainx.shape[-1])
testx = testx.reshape(-1, trainx.shape[-1])
trainy = np.repeat(trainy, 10, axis=0)
testy = np.repeat(testy, 10, axis=0)
train_hod, test_hod = hod_params[tidx[0]].reshape(-1, hod_params.shape[-1]), \
                      hod_params[tidx[1]].reshape(-1, hod_params.shape[-1])
print(trainy.shape, train_hod.shape)
print(testy.shape, test_hod.shape)
if args.fithod:
    trainy = np.hstack([trainy, train_hod])
    testy = np.hstack([testy, test_hod])
print(trainx.shape, testx.shape)
print(trainy.shape, testy.shape)
trainx, scaler = sbitools.standardize(trainx, log_transform=True)
testx, _ = sbitools.standardize(testx, log_transform=True, scaler=scaler)
traingalx, scalerngal = sbitools.standardize(ngal[tidx[0]].reshape(-1, 1), log_transform=True)
testgalx, _ = sbitools.standardize(ngal[tidx[1]].reshape(-1, 1), log_transform=True, scaler=scalerngal)
trainx = np.hstack([trainx, traingalx])
testx = np.hstack([testx, testgalx])
print(trainx.shape, testx.shape)
######
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

###Diagnostics
for _ in range(args.nposterior):
    ii = np.random.randint(0, testx.shape[0], 1)[0]
    fig, ax = sbiplots.plot_posterior(testx[ii], testy[ii], posterior, titles=cosmonames)
    plt.savefig(savepath + 'posterior%04d.png'%(tidx[1][ii//10]))
sbiplots.test_diagnostics(testx, testy, posterior, titles=cosmonames, savepath=savepath, test_frac=.25, nsamples=500)

