'''A simple test for Gaussian posterior to make expected rank distribution
for simulations generated from sample parameter point
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

import sys
sys.path.append('../../../hmc/src/')
from pyhmc import PyHMC

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--id0', type=int, default=0, help='first quijote seed')
parser.add_argument('--id1', type=int, default=2000, help='last quijote seed')
args = parser.parse_args()

###
def simulation(p, seed=0):
    m, c = p[0], p[1]
    np.random.seed(seed)
    x = np.random.uniform(0, 10, 100)
    y = m*x + c + np.random.normal(0, 0.1, x.size)
    return x, y

 
@tf.function
def unnormalized_log_prob(p, x, y):
    #Chisq                                                                                                                                                    
    m, c = p[0], p[1]
    residual = (y - m*x - c)/0.1
    chisq = tf.multiply(residual, residual)
    chisq = 0.5 * tf.reduce_sum(chisq)
    logprob = -chisq
    #Prior                                                                                                                                                  
    logprior = tf.reduce_sum(tfd.Normal(0, 1).log_prob(m))
    logprior = logprior + tf.reduce_sum(tfd.Normal(0, 1).log_prob(c))
    toret = logprob + logprior
    return  toret


@tf.function
def grad_log_prob(p, x, y):
    with tf.GradientTape() as tape:
        tape.watch(p)
        logposterior = tf.reduce_sum(unnormalized_log_prob(p, x, y))
    grad = tape.gradient(logposterior, p)
    return grad


def check_rank_hist():
  nposterior = 500
  samples = []
  for i in range(nposterior):
    samples.append(np.load('./tmp/figs/samples%d.npy'%i))
    ptrue = np.load('./tmp/ptrue.npy')
    print(ptrue)

    fig, ax = plt.figure(1, 2, figsize=(10, 4))
    nbins = 10 
    for j in range(2):
      ranks = []
      for i in range(nposterior):
        ranks.append((samples[i][:, j] < ptrue[j]).sum())

      ncounts = len(ranks)/nbins
      ax[j].hist(ranks, bins=nbins)
      ax[j].axhline(ncounts, color='k')
      ax[j].axhline(ncounts + ncounts**0.5, color='k', ls="--")
      ax[j].axhline(ncounts - ncounts**0.5, color='k', ls="--")
    plt.savefig('./tmp/rankhist')



#################
ptrue = [0.32, 0.4]
np.save('./tmp/ptrue', ptrue)
for seed in range(args.id0, args.id1):
  
    x, y = simulation(ptrue, seed=seed)
    print("For seed : ", seed)
    plt.figure()
    plt.plot(x, y, '.')
    plt.plot(x, ptrue[0]*x + ptrue[1])
    plt.savefig('./tmp/figs/data%d'%seed)

    ### DO HMC
    x, y = tf.constant(x, dtype=tf.float32), tf.constant(y, dtype=tf.float32)
    py_log_prob = lambda p: unnormalized_log_prob(tf.constant(p, dtype=tf.float32), x, y).numpy().astype(np.float32)
    py_grad_log_prob = lambda p: grad_log_prob(tf.constant(p, dtype=tf.float32), x, y).numpy().astype(np.float32)
    hmckernel = PyHMC(py_log_prob, py_grad_log_prob)

    q = np.array([0.1, 2.]).astype(np.float32)
    stepsize = 0.001
    samples, accs = [], []
    burnin = 100
    for i in range(1100):
        if i%500 == 0 : print(i)
        lpsteps = np.random.randint(10, 20, 1)[0]
        q, _, acc, energy, _ = hmckernel.hmc_step(q.copy(), lpsteps, stepsize)
        if i > burnin:
          samples.append(q)
          accs.append(acc)

    samples, accs = np.array(samples), np.array(accs)
    np.save('./tmp/figs/samples%d'%seed, samples)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(samples[:, 0])
    ax[1].plot(samples[:, 1])
    plt.savefig('./tmp/figs/samples%d'%seed)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(samples[:, 0])
    ax[0].axvline(ptrue[0], color='k')
    ax[1].hist(samples[:, 1])
    ax[1].axvline(ptrue[1], color='k')
    plt.savefig('./tmp/figs/hist%d'%seed)
    plt.close()

