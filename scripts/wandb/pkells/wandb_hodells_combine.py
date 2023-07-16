import numpy as np
#import matplotlib.pyplot as plt
import sys, os
sys.path.append('../../../src/')
sys.path.append('../')
import sbitools, sbiplots
import loader_hod_ells as loader
import wandb
import yaml
from folder_path import hodells_path

## Parse arguments
config_data = sys.argv[1]
nmodels = int(sys.argv[2])
cfgd_dict = yaml.load(open(f'{config_data}'), Loader=yaml.Loader)
sweep_id = cfgd_dict['sweep']['id']
print(sweep_id)

cuts = cfgd_dict['datacuts']
args = {}
for i in cfgd_dict.keys():
    args.update(**cfgd_dict[i])
cfgd = sbitools.Objectify(**args)

#
datapath1 = f'/mnt/ceph/users/cmodi/contrastive/data/{cfgd.simulation}/FoF/z{int(cfgd.z*10):02d}-N{int(cfgd.nbar/1e-4):04d}/{cfgd.hodmodel}/'
datapath2 = f'/mnt/ceph/users/cmodi/contrastive/data/{cfgd.simulation}/Rockstar/z{int(cfgd.z*10):02d}-N{int(cfgd.nbar/1e-4):04d}/{cfgd.hodmodel}/'
print(f"data path :\n{datapath1}\n{datapath2}")
cfgd.analysis_path = hodells_path(cfgd_dict)
cfgd.model_path = cfgd.analysis_path + '/%s/'%sweep_id
os.makedirs(cfgd.model_path, exist_ok=True)
print("\nWorking directory : ", cfgd.model_path)
#os.system('cp {config_data} {cfgd.model_path}')
##


#############
features1, params1 = loader.hod_ells_lh(datapath1, cfgd)
features2, params2 = loader.hod_ells_lh(datapath2, cfgd)
features = np.concatenate([features1, features2], axis=1)
params = np.concatenate([params1, params2], axis=1)

#####
#############
def train_sweep(config=None):

    
    with wandb.init(config=config) as run:

        # Copy your config 
        # run.name = 'model%d'%np.random.randint(1000)
        cfgm = wandb.config
        cfgm = sbitools.Objectify(**cfgm)
        cfgm.retrain = True

        print("running for model name : ", run.name)
        cfgm.model_path = f"{cfgd.model_path}/{run.name}/"
        os.makedirs(cfgm.model_path, exist_ok=True)

        data, posterior, inference, summary = sbitools.analysis(cfgd, cfgm, features, params, verbose=False)

        # Make the loss and optimizer
        for i in range(len(summary['train_log_probs'])):
            metrics = {"train_log_probs": summary['train_log_probs'][i],
                   "validation_log_probs": summary['validation_log_probs'][i]}
            wandb.log(metrics)
        wandb.run.summary["best_validation_log_prob"] = summary['best_validation_log_prob']
        print(wandb.run.summary["best_validation_log_prob"])
        wandb.log({'output_directory': cfgm.model_path})
               


if __name__ == '__main__':

    print(f"run for {nmodels} models")
    wandb.agent(sweep_id=sweep_id, function=train_sweep, count=nmodels, project='quijote-hodells')

