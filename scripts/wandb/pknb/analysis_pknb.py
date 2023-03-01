import numpy as np
import sys, os
sys.path.append('../../../src/')
sys.path.append('../')
import sbitools, sbiplots
import argparse
import pickle, json
import loader_hod_pknb as loader
import yaml
import folder_path


cfg_data = sys.argv[1]
cfg_model = sys.argv[2]
cfgd_dict = yaml.load(open(f'{cfg_data}'), Loader=yaml.Loader)
cfgm_dict = yaml.load(open(f'{cfg_model}'), Loader=yaml.Loader)['flow']
cuts = cfgd_dict['datacuts']
args = {}
for i in cfgd_dict.keys():
    args.update(**cfgd_dict[i])
cfgm = sbitools.Objectify(**cfgm_dict)
cfgd = sbitools.Objectify(**args)
np.random.seed(cfgd.seed)


#
datapath = f'/mnt/ceph/users/cmodi/contrastive/data/{cfgd.simulation}/{cfgd.finder}/z{int(cfgd.z*10):02d}-N{int(cfgd.nbar/1e-4):04d}/{cfgd.hodmodel}/'
model_path = f'/{cfgm.model}-nt{cfgm.ntransforms}-{cfgm.nhidden}-b{cfgm.batch}-b{cfgm.batch}-lr{cfgm.lr}/'
print(f"data path : {datapath}\nmodel folder name : {model_path}")

cfgd.analysis_path = folder_path.pknb_path(cfgd_dict)
print(cfgd.analysis_path)
#cfgd.analysis_path = './tmp/'
cfgm.model_path = cfgd.analysis_path + model_path
os.makedirs(cfgm.model_path, exist_ok=True)
print("\nWorking directory : ", cfgm.model_path)

#############
features, params = loader.hod_pknb_lh(datapath, cfgd)
print("features and params shapes : ", features.shape, params.shape)

data, posterior, inference, summary = sbitools.analysis(cfgd, cfgm, features, params)

print("Diagnostics for test dataset")
for i in range(cfgd.ndiagnostics):
    fig, ax = sbiplots.plot_posterior(data.testx[i], data.testy[i], posterior, savename=f'{cfgm.model_path}/corner{i}.png')


print("Diagnostics for training dataset")
for i in range(cfgd.ndiagnostics):
    fig, ax = sbiplots.plot_posterior(data.testx[i], data.testy[i], posterior, savename=f'{cfgm.model_path}/corner-train{i}.png')


