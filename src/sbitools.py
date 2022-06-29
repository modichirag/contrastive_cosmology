import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils


###
def quijote_params():
    params = np.load('/mnt/ceph/users/cmodi/Quijote/params_lh.npy')
    params_fid = np.load('/mnt/ceph/users/cmodi/Quijote/params_fid.npy')
    ndim = len(params_fid)
    cosmonames = r'$\Omega_m$,$\Omega_b$,$h$,$n_s$,$\sigma_8$'.split(",")
    return params, params_fid, cosmonames

###
def sbi_prior(params, offset=0.25):
    '''
    Generate priors for parameters of the simulation set with offset from min and max value
    '''
    lower_bound, upper_bound = .1 * np.round(10 * params.min(0)) * (1-offset),\
                                      .1 * np.round(10 * params.max(0)) * (1+offset)
    lower_bound, upper_bound = (torch.from_numpy(lower_bound.astype('float32')), 
                                torch.from_numpy(upper_bound.astype('float32')))
    prior = utils.BoxUniform(lower_bound, upper_bound)
    print("prior : ", lower_bound, upper_bound)
    return prior

###
def test_train_split(x, y, train_size_frac=0.8, random_state=0, reshape=True):
    '''
    Split the data into test and training dataset
    '''
    train, test = train_test_split(np.arange(x.shape[0])[:, np.newaxis], 
                                   train_size=train_size_frac, random_state=random_state)
    train_id = train.ravel()
    test_id = test.ravel()
    train_x = x[train_id]
    test_x = x[test_id]
    train_y = y[train_id]
    test_y = y[test_id]    
    if reshape:
        if len(train_x.shape) > 2:
            nsim = train_x.shape[1] # assumes that features are on last axis
            train_x = train_x.reshape(-1, train_x.shape[-1])
            test_x = test_x.reshape(-1, train_x.shape[-1])
            train_y = train_y.reshape(-1, train_y.shape[-1])
            test_y = test_y.reshape(-1, train_y.shape[-1])

    return [train_x, train_y], [test_x, test_y], [train_id, test_id]


###
def standardize(data, secondary=None, log_transform=True, scaler=None):
    '''
    Given a dataset, standardize by removing mean and scaling by standard deviation
    '''
    if log_transform:
        data = np.log10(data)
        if secondary is not None:
            secondary = np.log10(secondary)
    if scaler is None: 
        scaler = StandardScaler()
        data_s = scaler.fit_transform(data)
    else: 
        data_s = scaler.transform(data)
    if secondary is not None:
        secondary_s = scaler.transform(secondary)
        return data_s, secondary_s, scaler
    return data_s, scaler


###
def minmax(data, log_transform=True, scaler=None):
    '''
    Given a dataset, standardize by removing mean and scaling by standard deviation
    '''
    if log_transform:
        data = np.log10(data)
    if scaler is None: 
        scaler = MinMaxScaler()
        data_s = scaler.fit_transform(data)
    else: 
        data_s = scaler.transform(data)
    return data_s, scaler


###
def sbi(trainx, trainy, prior, nhidden=32, nlayers=5, model='maf', batch_size=128):

    density_estimator_build_fun = posterior_nn(model=model, \
                                               hidden_features=nhidden, \
                                               num_transforms=nlayers)
    inference = SNPE(prior=prior, density_estimator=density_estimator_build_fun)
    inference.append_simulations(
        torch.from_numpy(trainy.astype('float32')), 
        torch.from_numpy(trainx.astype('float32')))
    density_estimator = inference.train(training_batch_size=batch_size, show_train_summary=True)
    posterior = inference.build_posterior(density_estimator)
    return inference, density_estimator, posterior
