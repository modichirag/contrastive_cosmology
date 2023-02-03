import numpy as np
#import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../../src/')
sys.path.append('../')
import sbitools, sbiplots
import argparse
import pickle, json
import dataloaders
from io import StringIO
import wandb
import yaml
from sbi.inference import SNPE
from sbi.utils.get_nn_models import posterior_nn
import torch

## Parse arguments

sweep_id = sys.argv[1]
nmodels = sys.argv[2]
config_data = sys.argv[3]
configd = yaml.load(open(f'{config_data}'), Loader=yaml.Loader)
configd = sbitools.Objectify(**configd)

#
folder = '//z%d-N%04d/%s/'%(configd.z*10, configd.nbar*1e4, '%s')
datapath = folder%configd.hodmodel
if configd.finder == 'rockstar': datapath = datapath[:-1] + '-rock/'
configd.datapath = datapath
datapath = "/mnt/ceph/users/cmodi/contrastive/data/" + datapath
path = datapath.replace("data", "analysis")

folder = "kmax%02d-ells/"%(configd.kmax*100)
savepath = path + folder + '/%s/'%sweep_id
os.makedirs(savepath, exist_ok=True)
print("\nWorking directory : ", savepath)
##


#############
dataloader = getattr(dataloaders, configd.dataloader)
features, params = dataloader(configd)
prior = sbitools.sbi_prior(params.reshape(-1, params.shape[-1]), offset=0.2)
print("features and params shapes : ", features.shape, params.shape)
data = sbitools.test_train_split(features, params, train_size_frac=0.8, random_state=0)
if configd.standardize: 
    data.trainx, data.testx, scaler = sbitools.standardize(data.trainx, secondary=data.testx, log_transform=configd.logit)
else: scaler = None
with open(savepath + "scaler.pkl", "wb") as handle:
    pickle.dump(scaler, handle)
np.save(savepath + 'tidx', data.tidx)
cosmonames = r'$\Omega_m$,$\Omega_b$,$h$,$n_s$,$\sigma_8$'.split(",")
cosmonames = cosmonames + ["Mcut", "sigma", "M0", "M1", "alpha"]
print(data.trainx)
print(np.isnan(data.trainx).sum())
print(np.isinf(data.trainx).sum())
print(data.trainy)
print(np.isnan(data.trainy).sum())
print(np.isinf(data.trainy).sum())



#####
#############
def train_sweep(config=None):

    
    with wandb.init(config=config) as run:

        # Copy your config 
        # run.name = 'model%d'%np.random.randint(1000)
        config = wandb.config
        print("running for model name : ", run.name)
        modelpath = savepath + run.name + '/'
        os.makedirs(modelpath, exist_ok=True)
        
        # SBI
        density_estimator_build_fun = posterior_nn(model='maf',
                                                   hidden_features=config['num_hidden'], 
                                                   num_transforms=config['num_transforms'])
        inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun)
        inference.append_simulations(
            torch.from_numpy(data.trainy.astype('float32')), 
            torch.from_numpy(data.trainx.astype('float32')))

        density_estimator = inference.train(show_train_summary=True, 
                                            training_batch_size=config['batch_size'],
                                            learning_rate=config['learning_rate'],
                                            validation_fraction=0.2)
        posterior = inference.build_posterior(density_estimator)

        # Make the loss and optimizer
        for i in range(len(inference.summary['train_log_probs'])):
            metrics = {"train_loss": inference.summary['train_log_probs'][i],
                   "validation_loss": inference.summary['validation_log_probs'][i]}
            wandb.log(metrics)
        wandb.run.summary["best_validation_log_prob"] = inference.summary['best_validation_log_probs'][0]
        wandb.log({'output_directory': modelpath})
        
        sbitools.save_posterior(posterior, modelpath)
        sbitools.save_inference(inference, modelpath)

        


if __name__ == '__main__':

    print(f"run for {nmodels} models")
    wandb.agent(sweep_id=sweep_id, function=train_sweep, count=nmodels, project='sbijobs')

