import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.append('../../../src/')
sys.path.append('../')
print('current path : ', sys.path[0])

import sbitools, sbiplots
import argparse
import pickle, json
from io import StringIO
import wandb
import yaml
import torch
import loader_hod_bispec as loader_bk
import folder_path
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

api = wandb.Api()


simulation = sys.argv[1]
finder = sys.argv[2]
hodmodel_data = sys.argv[3]
hodmodel = sys.argv[4]
nposterior = int(sys.argv[5])
z = 0.5
nbar = 0.0004
test_frac = 0.4
nsamples = 200

basepath = '/mnt/ceph/users/cmodi/contrastive/analysis/'
numd = f'z{int(z*10):02d}-N{int(nbar/1e-4):04d}'
print(f"numd suffix - {numd}") #'z05-N0004'


print()
datapath = f'/mnt/ceph/users/cmodi/contrastive/data/{simulation}/{finder}/z{int(z*10):02d}-N{int(nbar/1e-4):04d}/{hodmodel_data}/'
print(f"data path : {datapath}")
savepath = f'/mnt/ceph/users/cmodi/contrastive/diagnostics/z{int(z*10):02d}-N{int(nbar/1e-4):04d}/ens{nposterior}/{simulation}-{finder}-{hodmodel_data}/rankstats/'
print(f"plots will be saved at : {savepath}")
os.makedirs(savepath, exist_ok=True)
os.makedirs(f"{savepath}/data/", exist_ok=True)

#Which simulation to test
train_idx, test_idx = sbitools.test_train_split(np.arange(2000), None, train_size_frac=0.85, retindex=True)



def insert_sweep_name(path):
    if '%s' in path: 
        dirname = '/'.join(cfg_path.split('/')[:-2])
    else: 
        dirname = path
    print(f"In directory {dirname}")
    for root, dirs, files in os.walk(dirname):
        if len(dirs) > 1 : 
            print('More than 1 sweeps, abort!')
            raise
        break
    print(f"Sweep found : {dirs[0]}")
    if '%s' in path: 
        return path%dirs[0]
    else:
        return path + f'/{dirs[0]}/'
    

def setup_cfg(cfg_path, verbose=False):
    
    cfg_path = insert_sweep_name(cfg_path)
    cfg_dict = yaml.load(open(f'{cfg_path}'), Loader=yaml.Loader)
    sweep_id = cfg_dict['sweep']['id']
    sweep = api.sweep(f'modichirag92/quijote-hodells/{sweep_id}')
    #sort in the order of validation log prob                                                                                                                     
    names, log_prob = [], []
    for run in sweep.runs:
        if run.state == 'finished':
            # print(run.name, run.summary['best_validation_log_prob'])
            try:
                model_path = run.summary['output_directory']
                names.append(run.name)
                log_prob.append(run.summary['best_validation_log_prob'])
            except Exception as e:
                print('Exception in checking state of run : ', e)
    idx = np.argsort(log_prob)[::-1]
    names = np.array(names)[idx]

    args = {}
    for i in cfg_dict.keys():
        args.update(**cfg_dict[i])
    cfg = sbitools.Objectify(**args)
    cfg.analysis_path = folder_path.bispec_path(cfg_dict, verbose=verbose)
    print('analysis path : ', cfg.analysis_path)
    scaler = sbitools.load_scaler(cfg.analysis_path)

    toret = {'sweepid':sweep_id, 'cfg':cfg, 'idx':idx, 'scaler':scaler, 'names':names,  'cfg_dict':cfg_dict}
    return toret



def parse_name(cfg_path, suffix=''):
    
    if cfg_path.find('quijote') + 1 :
        msim = 'quijote'
        if cfg_path.find('Rockstar') + 1 :
            msim = 'quijote-rockstar'
        if cfg_path.find('combine') + 1 :
            msim = 'quijote-combine'
    else: msim = 'fastpm' 
    summ = 'bk'
    kmax = cfg_path.split("kmax")[1][:3]
    name = f'{msim}-{hodmodel}-{summ}-kmax{kmax}{suffix}'
    print(f"saving at {name}")
    return name


def get_ranks(sweepdict, savename, nposterior=nposterior, nsamples=nsamples, test_frac=test_frac, cosmoonly=True, verbose=False):    

    #load the files from the data folder
    cfgm = sweepdict['cfg']
    cfgm_dict = sweepdict['cfg_dict']
    features, params = loader_bk.hod_bispec_lh(datapath, cfgm) # datacuts are used from model config
    data = sbitools.test_train_split(features, params, train_size_frac=0.85)
    scaler = sweepdict['scaler']
    data.trainx = sbitools.standardize(data.trainx, scaler=scaler, log_transform=cfgm.logit)[0]
    data.testx = sbitools.standardize(data.testx, scaler=scaler, log_transform=cfgm.logit)[0]
    print('test data shape : ', data.testx.shape)
    
    sweepid = sweepdict['sweepid']
    posteriors = []
    for j in range(nposterior):
        name = sweepdict['names'][j]
        if verbose: print(name)
        model_path = f"{sweepdict['cfg'].analysis_path}/{sweepid}/{name}/"
        posteriors.append(sbitools.load_posterior(model_path))
    posterior = NeuralPosteriorEnsemble(posteriors=posteriors)

    trues, mus, stds, ranks = sbiplots.get_ranks(data.testx, data.testy, posterior, test_frac=test_frac, nsamples=nsamples, ndim=5)
    print('ranks shape :' , ranks.shape)
    
    tosave = np.stack([trues, mus, stds, ranks], axis=-1)
    np.save(f'{savepath}/data/ranks-{savename}', tosave)
    
    fig, ax = sbiplots.plot_coverage(ranks, titles=sbiplots.cosmonames)
    plt.savefig(f'{savepath}/coverage-{savename}.png')
    plt.close()
    
    fig, ax = sbiplots.plot_ranks_histogram(ranks, titles=sbiplots.cosmonames)
    plt.savefig(f'{savepath}/rankplot-{savename}.png')
    plt.close()
    
    fig, ax = sbiplots.plot_predictions(trues, mus, stds,  titles=sbiplots.cosmonames)
    plt.savefig(f'{savepath}/predictions-{savename}.png')
    plt.close()



def run(cfg_path, suffix=''):
    print(cfg_path)
    sweepdict = setup_cfg(cfg_path)
    savename = parse_name(cfg_path, suffix=suffix)
    get_ranks(sweepdict, savename=savename)




# ## Quijote + combine
# for kmax in [0.2, 0.3, 0.5]:
#     print()
#     try:
#         cfg_path = f'{basepath}/quijote/combine/{numd}/{hodmodel}/bk-kmax{kmax:0.1f}-kmin0.005-ngals-standardize/%s/sweep_config_bspec_quijote_combine.yaml'
#         run(cfg_path)
#     except Exception as e:
#         print(f"Exception occured when working for path\n{cfg_path}\n{e}")

        
## Quijote + FOF

suffix = '-train94'
suffixname = '-train94-nogals'

for kmax in [0.5]:
    print()
    try:
        #cfg_path = f'{basepath}/quijote/FoF/{numd}/{hodmodel}/bk-kmax{kmax:0.1f}-kmin0.005-ngals-standardize/%s/sweep_config_bspec_quijote.yaml'
        cfg_path = f'{basepath}/quijote/FoF/{numd}/{hodmodel}/bk-kmax{kmax:0.1f}-kmin0.005-standardize{suffix}/%s/sweep_config_bspec_quijote.yaml'
        run(cfg_path, suffixname)
    except Exception as e:
        print(f"Exception occured when working for path\n{cfg_path}\n{e}")

        
## Quijote + Rockstar
for kmax in [0.5]:
    print()
    try:
        cfg_path = f'{basepath}/quijote/Rockstar/{numd}/{hodmodel}/bk-kmax{kmax:0.1f}-kmin0.005-standardize{suffix}/%s/sweep_config_bspec_quijote.yaml'
        run(cfg_path, suffixname)
    except Exception as e:
        print(f"Exception occured when working for path\n{cfg_path}\n{e}")

        
## Fastpm + FOF
for kmax in [0.5]:
    print()
    try:
        cfg_path = f'{basepath}/fastpm/FoF/{numd}/{hodmodel}/bk-kmax{kmax:0.1f}-kmin0.005-standardize{suffix}/%s/sweep_config_bspec_fastpm.yaml'
        run(cfg_path, suffixname)
    except Exception as e:
        print(f"Exception occured when working for path\n{cfg_path}\n{e}")





##### Old & deprecated
    
# ## Quijote + FOF
# print()
# cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//quijote/FoF/z05-N0004/zheng07/bk-kmax0.2-kmin0.005-ngals-standardize/84zhmbwr/sweep_config_bspec_quijote.yaml'
# run(cfg_path)

# # cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//quijote/FoF/z05-N0004/zheng07_velab/bk-kmax0.3-kmin0.005-ngals-standardize/hhglght1/sweep_config_bspec_quijote.yaml'
# # run(cfg_path)

# # cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//quijote/FoF/z05-N0004/zheng07_velab/bk-kmax0.5-kmin0.005-ngals-standardize/d8075uar/sweep_config_bspec_quijote.yaml'
# # run(cfg_path)

# ## FastPM + FoF
# print()
# cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//fastpm/FoF/z05-N0004/zheng07/bk-kmax0.2-kmin0.005-ngals-standardize/odgoz7w9/sweep_config_bspec_fastpm.yaml'
# run(cfg_path)

# # cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//fastpm/FoF/z05-N0004/zheng07_velab/bk-kmax0.3-kmin0.005-ngals-standardize/9tlu63w4/sweep_config_bspec_fastpm.yaml'
# # run(cfg_path)

# # cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//fastpm/FoF/z05-N0004/zheng07_velab/bk-kmax0.5-kmin0.005-ngals-standardize/3ba457rn/sweep_config_bspec_fastpm.yaml'
# # run(cfg_path)


# ## Quijote rockstar
# print()
# cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//quijote/Rockstar/z05-N0004/zheng07/bk-kmax0.2-kmin0.005-ngals-standardize/in5wqqge/sweep_config_bspec_quijote.yaml'
# run(cfg_path)

# # cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//quijote/Rockstar/z05-N0004/zheng07_velab/bk-kmax0.3-kmin0.005-ngals-standardize/up2ek2kz/sweep_config_bspec_quijote.yaml'
# # run(cfg_path)

# # cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//quijote/Rockstar/z05-N0004/zheng07_velab/bk-kmax0.5-kmin0.005-ngals-standardize/cz7al32p/sweep_config_bspec_quijote.yaml'
# # run(cfg_path)
