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

import sys, os
sys.path.append('../../../hmc/src/')
from pyhmc import PyHMC

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--id0', type=int, default=0, help='first quijote seed')
parser.add_argument('--nsim', type=int, default=1, help='last quijote seed')
parser.add_argument('--mode', type=str, default='sample', help='last quijote seed')
parser.add_argument('--l0', type=float, default=0.0, help='lower val for uniform prior')
parser.add_argument('--l1', type=float, default=1.0, help='upper val for uniform prior')
parser.add_argument('--mu', type=float, default=0.0, help='mean of normal')
parser.add_argument('--std', type=float, default=1.0, help='std of normal')
parser.add_argument('--prior', type=str, default='normal', help='upper val for uniform prior')

args = parser.parse_args()


folder = '/mnt/ceph/users/cmodi/contrastive/analysis/rank_fiducial/'
mu, std = args.mu, args.std
l0, l1 = args.l0, args.l1
noise = 0.5
ptrue = [0.32, 0.4]
#
if args.prior == 'normal': suffix = 'normal-mu%dp%ds%dp%d_noise0p5/'%(mu%10, (mu*10)%10, std%10, (std*10)%10)
elif args.prior == 'uniform': suffix = 'uniform-%dp%d-%dp%d_noise0p5/'%(l0%10, (l0*10)%10, l1%10, (l1*10)%10)
print(suffix)

savefolder = folder + suffix
os.makedirs(savefolder, exist_ok=True)
print(savefolder)
np.save(savefolder + 'ptrue', ptrue + [noise])


###
def simulation(p, noise, seed=0):
    m, c = p[0], p[1]
    np.random.seed(seed)
    x = np.random.uniform(0, 10, 100)
    y = m*x + c + np.random.normal(0, noise, x.size)
    return x, y


@tf.function
def normal_prior(m, c, mu=mu, std=std):
  dist = tfd.Normal(mu, std)
  logprior = tf.reduce_sum(dist.log_prob(m))
  logprior = logprior + tf.reduce_sum(dist.log_prob(c))
  return logprior

@tf.function
def uniform_prior(m, c, l0=l0, l1=l1):
  dist = tfd.Uniform(l0, l1)
  logprior = tf.reduce_sum(dist.log_prob(m))
  logprior = logprior + tf.reduce_sum(dist.log_prob(c))
  return logprior


@tf.function
def unnormalized_log_prob(p, x, y, noise):

    m, c = p[0], p[1]
    residual = (y - m*x - c)/noise
    chisq = tf.multiply(residual, residual)
    chisq = 0.5 * tf.reduce_sum(chisq)
    logprob = -chisq
    #Prior
    if "normal" in suffix:
      toret = logprob + normal_prior(m, c)
    elif "uniform" in suffix:
      toret = logprob + uniform_prior(m, c)
    return  toret


@tf.function
def grad_log_prob(p, x, y, noise):
    with tf.GradientTape() as tape:
        tape.watch(p)
        logposterior = tf.reduce_sum(unnormalized_log_prob(p, x, y, noise))
    grad = tape.gradient(logposterior, p)
    return grad


def check_rank_hist():
	ptrue = np.load(savefolder + '/ptrue.npy')
	print(ptrue)
	nposterior = 200
	samples = []
	for i in range(nposterior):
		samples.append(np.load(savefolder + '%d/samples.npy'%i))
	
	fig, ax = plt.subplots(1, 2, figsize=(10, 4))
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
	plt.suptitle(suffix)
	plt.savefig(savefolder + '/rankhist')
	plt.close()



def run_experiment(seed, noise=noise):
	x, y = simulation(ptrue, noise, seed=seed)
	print("For seed : ", seed)
	savepath = savefolder + '%d/'%seed
	os.makedirs(savepath, exist_ok=True)
	
	plt.figure()
	plt.plot(x, y, '.')
	plt.plot(x, ptrue[0]*x + ptrue[1])
	plt.savefig(savepath + '/data')
	plt.close()
	
	
    ### DO HMC
	x, y = tf.constant(x, dtype=tf.float32), tf.constant(y, dtype=tf.float32)
	noise = tf.constant(noise)
	py_log_prob = lambda p: unnormalized_log_prob(tf.constant(p, dtype=tf.float32), x, y, noise).numpy().astype(np.float32)
	py_grad_log_prob = lambda p: grad_log_prob(tf.constant(p, dtype=tf.float32), x, y, noise).numpy().astype(np.float32)
	hmckernel = PyHMC(py_log_prob, py_grad_log_prob)
	
	q = np.array([0.5, 0.3]).astype(np.float32)
	stepsize = 0.01
	samples, accs = [], []
	burnin = 200
	nsamples = 1000
	for i in range(nsamples + burnin):
		if i%500 == 0 : print(i)
		lpsteps = np.random.randint(10, 20, 1)[0]
		q, _, acc, energy, _ = hmckernel.hmc_step(q.copy(), lpsteps, stepsize)
		if i > burnin:
			samples.append(q)
			accs.append(acc)
			
	samples, accs = np.array(samples), np.array(accs)
	np.save(savepath + '/samples', samples)

	fig, ax = plt.subplots(1, 2, figsize=(10, 4))
	ax[0].plot(samples[:, 0])
	ax[1].plot(samples[:, 1])
	plt.savefig(savepath + 'samples')
	plt.close()

	fig, ax = plt.subplots(1, 2, figsize=(10, 4))
	ax[0].hist(samples[:, 0])
	ax[0].axvline(ptrue[0], color='k')
	ax[1].hist(samples[:, 1])
	ax[1].axvline(ptrue[1], color='k')
	plt.savefig(savepath + 'hist')
	plt.close()
	return 0
	
	

if args.mode == 'plot':
	check_rank_hist()
else:
	ids = np.arange(args.id0, args.id0+args.nsim)
	for seed in ids:
		run_experiment(seed)

	check_rank_hist()

sys.exit()
