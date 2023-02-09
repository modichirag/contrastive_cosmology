import numpy as np
import sys, os
import wandb
from ruamel import yaml
import folder_path
wandb.login()

config_data = sys.argv[1]
sweep_id = wandb.sweep(sweep=yaml.load(open(f'./configs/config_wandb_v1.yaml'), Loader=yaml.Loader), project='quijote-hodells', entity='modichirag92')
print(sweep_id)
nmodels = 1

#save config file with sweep id
cfgd = yaml.load(open(f'{config_data}'), Loader=yaml.RoundTripLoader)
cfgd['sweep'] = {'id' : sweep_id}
fname = config_data.split('/')[-1]
analysis_path = folder_path.hodells_path(cfgd)
model_path = f'{analysis_path}/{sweep_id}/'
config_path = f'{model_path}/sweep_{fname}'
os.makedirs(model_path, exist_ok=True)
with open(config_path, 'w') as outfile:
    yaml.dump(cfgd, outfile, Dumper=yaml.RoundTripDumper)

print(f"config path saved at:\n{config_path}\n")
#run once to initiate the sweep    
command = f"time python -u wandb_hodells.py {config_path} {nmodels}"
print(command)
os.system(command)

