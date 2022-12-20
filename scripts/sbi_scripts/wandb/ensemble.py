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


api = wandb.Api()
sweep = api.sweep('modichirag92/sbi-intro3/3f6ue5dh')
sweep_runs = sweep.runs
for i in sweep_runs:
    print(i.name, i.summary['best_validation_log_prob'])
