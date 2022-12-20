import numpy as np
import sys, os
import wandb
import yaml
wandb.login()


#sweep_id = "lfjxq81k"
#sweep_id = "7u86e3ba"
sweep_id = "1p2zcshl"
#sweep_id = wandb.sweep(sweep=yaml.load(open(f'config_wandb.yaml'), Loader=yaml.Loader), project='sbijobs', entity='modichirag92')
print(sweep_id)
nmodels = 5
config_data = "config_data.yaml"

command = f"time python -u wandb_analysis.py {sweep_id} {nmodels} {config_data}"
print(command)
os.system(command)
