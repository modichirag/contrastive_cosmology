import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../src/')
import sbitools, sbiplots
from embed_nets import Arch, Simple_MLP
import argparse
import pickle, json
import dataloaders
from io import StringIO
from torch import nn
import torch
import torch.optim as optim

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataloader', type=str, help='which dataloader to use')
parser.add_argument('--datapath', type=str, help='datapath in contrastive data folder')
parser.add_argument('--datapath2', type=str, help='datapath in contrastive data folder')
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
parser.add_argument('--logit', type=int, default=1, help='take log transform of pells')
parser.add_argument('--standardize', type=int, default=1, help='whiten the dataset')
parser.add_argument('--retrain', type=int, default=0, help='retrain the network')
args = parser.parse_args()

np.random.seed(args.seed)
#
datapath = "/mnt/ceph/users/cmodi/contrastive/data/" + args.datapath
path = datapath.replace("data", "analysis")
folder = "%s-kmax%02d-nl%02d-nh%02d-s%03d-ells/"%(args.model, args.kmax*100, args.nlayers, args.nhidden, args.seed)
folder = "test/"
if args.suffix != "": 
    folder = folder[:-1] + "-%s/"% args.suffix 
savepath = path + folder
os.makedirs(savepath, exist_ok=True)
print("\nWorking directory : ", savepath)
cosmonames = r'$\Omega_m$,$\Omega_b$,$h$,$n_s$,$\sigma_8$'.split(",")
cosmonames = cosmonames + ["Mcut", "sigma", "M0", "M1", "alpha"]

##
print("Which dataloader to use?")
dataloader = getattr(dataloaders, args.dataloader)
print(dataloader)
with open(savepath + 'args.json', 'w') as fp:
    json.dump(vars(args), fp, indent=4, sort_keys=True)

#####
#####
#############
def analysis_mse(features, params, model):
    data = sbitools.test_train_split(features, params, train_size_frac=0.8)

    ### Standaradize
    if args.standardize: 
        data.trainx, data.testx, scaler = sbitools.standardize(data.trainx, secondary=data.testx, log_transform=args.logit)
    else: scaler = None
    with open(savepath + "scaler.pkl", "wb") as handle:
        pickle.dump(scaler, handle)
    np.save(savepath + 'tidx', data.tidx)

    ### MSEj
    scheduler = lambda opt: optim.lr_scheduler.ExponentialLR(opt, gamma=0.95)
    losses, opt = sbitools.train(data.trainx, data.trainy, model=model, 
                                 criterion= nn.MSELoss(), 
                                 lr = 1e-3, niter=50000, 
                                 batch_size=128, scheduler=scheduler)
    torch.save(model, savepath+'model.pb')
    #
    plt.plot(losses[len(losses)//10:])
    plt.savefig(savepath + "losses.png")
    plt.loglog()
    plt.grid(which='both')
    plt.close()

    return data, model, scaler


#############
def analysis(data, params):

    ### SBI
    prior = sbitools.sbi_prior(params.reshape(-1, params.shape[-1]), offset=0.2)
    print("output in a file now")
    tmp_out = StringIO()
    sys.stdout = tmp_out
    posterior = sbitools.sbi(data.trainx, data.trainy, prior, \
                                  model=args.model, nlayers=args.nlayers, \
                                  nhidden=args.nhidden, batch_size=args.batch, savepath=savepath, retrain=bool(args.retrain))
    sys.stdout = sys.__stdout__
    print(tmp_out.getvalue())
    with open(savepath + 'fit.log', 'w') as f:
        f.write(tmp_out.getvalue())

    return posterior


def diagnostics(data, posterior):
    print("Diagnostics for test dataset")
    for _ in range(args.nposterior):
        ii = np.random.randint(0, data.testx.shape[0], 1)[0]
        savename = savepath + 'posterior%04d.png'%(data.tidx[1][ii//params.shape[1]])
        fig, ax = sbiplots.plot_posterior(data.testx[ii], data.testy[ii], posterior, titles=cosmonames, savename=savename)
    sbiplots.test_diagnostics(data.testx, data.testy, posterior, titles=cosmonames, savepath=savepath, test_frac=0.2, nsamples=500)


#############
features, params = dataloader(args)
print("features and params shapes : ", features.shape, params.shape)

arch = Arch(features.shape[-1], params.shape[-1], nlayers=3, nhidden=[128, 64])
print("architecture : ", arch.__dict__)
with open(savepath + 'arch.json', 'w') as fp:
    json.dump(arch.__dict__, fp, indent=4, sort_keys=True)

model = Simple_MLP(arch)
print(model)


for i in model.parameters():
    print(i.shape)
print()

##train mse model
data, model, scaler = analysis_mse(features, params, model)
torch.save(model, savepath+'model.pb')

model = torch.load(savepath+'model.pb')
print(model)
sbiplots.mlp_diagnostics(data.testx, data.testy, model, titles=cosmonames, savepath=savepath, test_frac=0.2, suffix='mse')
    

##embed data and train nf model
x = sbitools.embed_data(data.trainx, model)
print(x.shape, data.trainx.shape)
data.trainx = x
x = sbitools.embed_data(data.testx, model)
print(x.shape, data.testx.shape)
data.testx = x

posterior = analysis(data, params)

#
if params.shape[-1] > len(cosmonames): 
    cosmonames = cosmonames + ["a_%d"%i for i in range(params.shape[-1] - len(cosmonames))]
print(cosmonames)

diagnostics(data, posterior)
diagnostics_train(data, posterior)
#try: diagnostics_fiducial(data, posterior, scaler)
#except: pass
#diagnostics_rockstar(data, posterior, scaler)


