import numpy as np
import sys, os
import wandb
from ruamel import yaml
import folder_path
wandb.login()

config_data = sys.argv[1]
print("config file : ", config_data)
cfgd = yaml.load(open(f'{config_data}'), Loader=yaml.RoundTripLoader)
fname = config_data.split('/')[-1]

#initialize sweep
sweep_id = wandb.sweep(sweep=yaml.load(open(f'./configs/config_wandb_v1.yaml'), Loader=yaml.Loader), project='quijote-hodells', entity='modichirag92')
print("Schedule sweep with id : ", sweep_id)
cfgd['sweep'] = {'id' : sweep_id}
nmodels = 1

#save config file in sweep folder
if 'ells' in config_data:
    print("Scheduling for Pk ells")
    analysis_path = folder_path.hodells_path(cfgd)
    model_path = f'{analysis_path}/{sweep_id}/'
    config_path = f'{model_path}/sweep_{fname}'
    if 'combine' in config_data: command = f"time python -u ./pkells_combine/wandb_hodells.py {config_path} {nmodels}"
    else: command = f"time python -u ./pkells/wandb_hodells.py {config_path} {nmodels}"

elif ('bspec' in config_data) or ('qspec' in config_data):
    print("Scheduling for Bispectrum")
    analysis_path = folder_path.bispec_path(cfgd)
    model_path = f'{analysis_path}/{sweep_id}/'
    config_path = f'{model_path}/sweep_{fname}'
    command = f"time python -u ./bispec/wandb_bispec.py {config_path} {nmodels}"
    
elif ('pknb' in config_data):
    print("Scheduling for pk and bispectrum")
    analysis_path = folder_path.pknb_path(cfgd)
    model_path = f'{analysis_path}/{sweep_id}/'
    config_path = f'{model_path}/sweep_{fname}'
    command = f"time python -u ./pknb/wandb_pknb.py {config_path} {nmodels}"

elif ('scat' in config_data):
    print("Scheduling for pk and bispectrum")
    analysis_path = folder_path.scat_path(cfgd)
    model_path = f'{analysis_path}/{sweep_id}/'
    config_path = f'{model_path}/sweep_{fname}'
    command = f"time python -u ./scat/wandb_scat.py {config_path} {nmodels}"
    
os.makedirs(model_path, exist_ok=True)
with open(config_path, 'w') as outfile:
    yaml.dump(cfgd, outfile, Dumper=yaml.RoundTripDumper)

print(f"config path saved at:\n{config_path}\n")
    
#run once to initiate the sweep    
print(command)
os.system(command)

