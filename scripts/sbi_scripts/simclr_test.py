import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../src/')
import sbitools, sbiplots
from embed_nets import Arch, Simple_MLP
import simclr
from simclr import SimCLR, Info_nce_loss, PS_loader
import argparse
import pickle, json
import dataloaders
from io import StringIO
from torch import nn
import torch
import torch.optim as optim


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--datapath', type=str, help='datapath in contrastive data folder')
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

folder = "test-simclr-wdecay/"
if args.suffix != "": 
    folder = folder[:-1] + "-%s/"% args.suffix 
savepath = path + folder
print("\nWorking directory : ", savepath)

##
with open(savepath + 'args.json') as f:
    args_model = json.load(f)
args_model = sbitools.Objectify(**args_model)

print(args_model)
    
print("Which dataloader to use?")
dataloader = getattr(dataloaders, args_model.dataloader)
print(dataloader)


#####
#############
def data_and_model(features, params):

    print("features and params shapes : ", features.shape, params.shape)
    data = sbitools.test_train_split(features, params, train_size_frac=0.8)

    ### Standaradize
    if args.standardize: 
        data.trainx, data.testx, scaler = sbitools.standardize(data.trainx, secondary=data.testx, log_transform=args.logit)
    else: scaler = None

    model = torch.load(savepath+'model.pb')
    return data, model, scaler



#############
features, params = dataloader(args_model)
data, model, scaler = data_and_model(features, params)
print(model)
print(model.__dict__)
device = 'cpu'


##embed data and train nf model
x = sbitools.embed_data(data.trainx, model.encoder, device=device)
data.trainx = x
x = sbitools.embed_data(data.testx, model.encoder, device=device)
data.testx = x

for j in range(x.shape[-1]):
    print(j, np.corrcoef(data.trainy[:, 4], data.trainx[:, j])[0, 1])
print()
for j in range(x.shape[-1]):
    print(j, np.corrcoef(data.trainy[:, 0], data.trainx[:, j])[0, 1])
	
plt.figure()
plt.plot(data.trainy[::, 4], data.trainx[::, 11], '.', ms=1)
#plt.semilogy()
plt.savefig('test.png')

# for j in model.parameters():
#     print(j.shape, j.min(), j.max(), j.mean(), j.std())
