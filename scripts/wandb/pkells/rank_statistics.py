import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.append('../../../src/')
sys.path.append('../')
import sbitools, sbiplots
import argparse
import pickle, json
from io import StringIO
import wandb
import yaml
import torch
import loader_hod_ells as loader
import folder_path
from sbi.utils.posterior_ensemble import NeuralPosteriorEnsemble

api = wandb.Api()

simulation = sys.argv[1]
finder = sys.argv[2]
hodmodel_data = sys.argv[3]     # hod of test dataset
hodmodel = sys.argv[4]          # hod of training dataset
nposterior = int(sys.argv[5])
z = 0.5
nbar = 0.0004
test_frac = 0.2
nsamples = 200

#keys for analysis
basepath = '/mnt/ceph/users/cmodi/contrastive/analysis/'
numd = f'z{int(z*10):02d}-N{int(nbar/1e-4):04d}'
print(f"numd suffix - {numd}") #'z05-N0004'


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
    if ('scat' in cfg_path):
        cfg.analysis_path = folder_path.scat_path(cfg_dict, verbose=verbose)
    elif ('pknb' in cfg_path):
        cfg.analysis_path = folder_path.pknb_path(cfg_dict, verbose=verbose)
    elif 'ells' in cfg_path:
        cfg.analysis_path = folder_path.hodells_path(cfg_dict, verbose=verbose)
    elif ('bspec' in cfg_path) or ('qspec' in cfg_path):
        cfg.analysis_path = folder_path.bispec_path(cfg_dict, verbose=verbose)
    scaler = sbitools.load_scaler(cfg.analysis_path)

    toret = {'sweepid':sweep_id, 'cfg':cfg, 'idx':idx, 'scaler':scaler, 'names':names, 'cfg_dict':cfg_dict}
    return toret


def get_ranks(sweepdict, savename, nposterior=nposterior, nsamples=nsamples, test_frac=test_frac, cosmoonly=True, verbose=False):    

    #load the files from the data folder
    cfgm = sweepdict['cfg']
    cfgm_dict = sweepdict['cfg_dict']
    features, params = loader.hod_ells_lh(datapath, cfgm) # datacuts are used from model config
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



def parse_name(cfg_path, suffix=''):   

    if cfg_path.find('quijote') + 1 :
        msim = 'quijote'
        if cfg_path.find('Rockstar') + 1 :
            msim = 'quijote-rockstar'
        if cfg_path.find('combine') + 1 :
            msim = 'quijote-combine'
    else: msim = 'fastpm' 
    if cfg_path.find('scat') + 1 :
        name = f'{msim}-scat-ells012-exps0123-orders012'
        print(f"saving at {name}")
        return name
    if cfg_path.find('pknb') + 1 :
        summ = 'pknb'
        kmax = cfg_path.split("kmax")[1][:6] + cfg_path.split("kmax")[2][:6]
    elif cfg_path.find('ells') + 1 :
        summ = 'ells'
        kmax = cfg_path.split("kmax")[1][:3]
    else:
        summ = 'bk'
        kmax = cfg_path.split("kmax")[1][:3]
    name = f'{msim}-{hodmodel}-{summ}-kmax{kmax}{suffix}'
    print(f"saving at {name}")
    return name


def run(cfg_path, suffix=''):
    print("\n")
    print(cfg_path)
    sweepdict = setup_cfg(cfg_path)
    savename = parse_name(cfg_path, suffix)
    get_ranks(sweepdict, savename=savename)
    



suffix = '-train94'
suffixname = '-train94-nogals'

# ## Quijote + combine
# for kmax in [0.3, 0.5]:
#     print()
#     try:
#         cfg_path = f'{basepath}/quijote/combine/{numd}/{hodmodel}/ells024-kmax{kmax:0.1f}-kmin0.005-ngals-offset_amp10000.0-standardize/%s/sweep_config_hodells_quijote_combine.yaml'
#         run(cfg_path)
#     except Exception as e:
#         print(f"Exception occured when working for path\n{cfg_path}\n{e}")


## Quijote + FOF
for kmax in [0.3, 0.5]:
    print()
    try:
        #cfg_path = f'{basepath}/quijote/FoF/{numd}/{hodmodel}/ells024-kmax{kmax:0.1f}-kmin0.005-ngals-offset_amp10000.0-standardize{suffix}/%s/sweep_config_hodells_quijote.yaml'
        cfg_path = f'{basepath}/quijote/FoF/{numd}/{hodmodel}/ells024-kmax{kmax:0.1f}-kmin0.005-offset_amp10000.0-standardize{suffix}/%s/sweep_config_hodells_quijote.yaml'
        run(cfg_path, suffixname)
    except Exception as e:
        print(f"Exception occured when working for path\n{cfg_path}\n{e}")



## Quijote + Rockstar
for kmax in [0.3, 0.5]:
    print()
    try:
        #cfg_path = f'{basepath}/quijote/Rockstar/{numd}/{hodmodel}/ells024-kmax{kmax:0.1f}-kmin0.005-ngals-offset_amp10000.0-standardize{suffix}/%s/sweep_config_hodells_quijote.yaml'
        cfg_path = f'{basepath}/quijote/Rockstar/{numd}/{hodmodel}/ells024-kmax{kmax:0.1f}-kmin0.005-offset_amp10000.0-standardize{suffix}/%s/sweep_config_hodells_quijote.yaml'
        run(cfg_path, suffixname)
    except Exception as e:
        print(f"Exception occured when working for path\n{cfg_path}\n{e}")

                


## FastPM + FOF
for kmax in [0.3, 0.5]:
    print()
    try:
        #cfg_path = f'{basepath}/fastpm/FoF/{numd}/{hodmodel}/ells024-kmax{kmax:0.1f}-kmin0.005-ngals-offset_amp10000.0-standardize{suffix}/%s/sweep_config_hodells_fastpm.yaml'
        cfg_path = f'{basepath}/fastpm/FoF/{numd}/{hodmodel}/ells024-kmax{kmax:0.1f}-kmin0.005-offset_amp10000.0-standardize{suffix}/%s/sweep_config_hodells_fastpm.yaml'
        run(cfg_path, suffixname)
    except Exception as e:
        print(f"Exception occured when working for path\n{cfg_path}\n{e}")
        



# ## Quijote + FOF
# # cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//quijote/FoF/z05-N0004/zheng07_velab/ells024-kmax0.5-kmin0.005-ngals-offset_amp10000.0-standardize/28a1ot12/sweep_config_hodells_quijote.yaml'
# # run(cfg_path)

# # cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//quijote/FoF/z05-N0004/zheng07_velab/ells024-kmax0.3-kmin0.005-ngals-offset_amp10000.0-standardize/2yxe7efo/sweep_config_hodells_quijote.yaml'
# # run(cfg_path)

# # ## Quijote + Rockstar
# # cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//quijote/Rockstar/z05-N0004/zheng07_velab/ells024-kmax0.5-kmin0.005-ngals-offset_amp10000.0-standardize/pu51arma/sweep_config_hodells_quijote.yaml'
# # run(cfg_path)

# # cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//quijote/Rockstar/z05-N0004/zheng07_velab//ells024-kmax0.3-kmin0.005-ngals-offset_amp10000.0-standardize/mr67wv5p/sweep_config_hodells_quijote.yaml'
# # run(cfg_path)


# ## FastPM + FOF
# cfg_path = '/mnt/ceph/users/cmodi/contrastive/analysis//fastpm/FoF/z05-N0004/zheng07_velab/ells024-kmax0.5-kmin0.005-ngals-offset_amp10000.0-standardize/y1nzq7gl/sweep_config_hodells_fastpm.yaml'
# run(cfg_path)

# cfg_path = f'/mnt/ceph/users/cmodi/contrastive/analysis//fastpm/FoF/z05-N0004/zheng07_velab/ells024-kmax0.3-kmin0.005-ngals-offset_amp10000.0-standardize/hdpegkie/sweep_config_hodells_fastpm.yaml'
# run(cfg_path)
        
