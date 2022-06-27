import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


###
def plot_posterior(x, y, posterior, nsamples=1000, titles=None):
    
    posterior_samples = posterior.sample((nsamples,), x=torch.from_numpy(x.astype('float32'))).detach().numpy()
    mu, std = posterior_samples.mean(axis=0), posterior_samples.std(axis=0)
    p = y
    ndim = len(y)
    #
    fig, axar = plt.subplots(ndim, ndim, figsize=(3*ndim, 3*ndim), sharex='col')
    nbins = 'auto'
    for i in range(ndim):
        axar[i, i].hist(np.array(posterior_samples[:, i]), density=True, bins=nbins);
        axar[i, i].axvline(mu[i], color='r');
        axar[i, i].axvline(mu[i] + std[i], color='r', ls="--");
        axar[i, i].axvline(mu[i] - std[i], color='r', ls="--");
        axar[i, i].axvline(p[i], color='k')
        for j in range(0, i):
            axar[i, j].plot(posterior_samples[:, j], posterior_samples[:, i], '.')
        for j in range(i+1, ndim):
            axar[i, j].set_axis_off()
    if titles is not None:
        for i in range(ndim): 
            axar[i, i].set_title(titles[i])

    return fig, axar


###
def test_diagnostics(x, y, posterior, nsamples=500, rankplot=True, titles=None, savepath="", test_frac=1.0):

    ndim = y.shape[1]
    if titles is None: titles = []*ndim
    ranks = []
    mus, stds = [], []
    trues = []
    for ii in range(x.shape[0]):
        if ii%100 == 0: print("Test iteration : ",ii)
        if np.random.uniform() > test_frac: continue
        posterior_samples = posterior.sample((nsamples,),
                                             x=torch.from_numpy(x[ii].astype('float32')), 
                                             show_progress_bars=False).detach().numpy()
        mu, std = posterior_samples.mean(axis=0), posterior_samples.std(axis=0)
        rank = [(posterior_samples[:, i] < y[ii, i]).sum() for i in range(ndim)]
        mus.append(mu)
        stds.append(std)
        ranks.append(rank)
        trues.append(y[ii])
    mus, stds, ranks = np.array(mus), np.array(stds), np.array(ranks)
    trues = np.array(trues)

    #plot ranks
    plt.figure(figsize=(15, 4))
    nbins = 10
    ncounts = x.shape[0]/nbins
    for i in range(5):
    #     plt.hist(np.array(ranks)[:, i], bins=10, histtype='step', lw=2)
        plt.subplot(151 + i)
        plt.hist(np.array(ranks)[:, i], bins=nbins)
        plt.title(titles[i])
        plt.xlabel('rank')
        plt.ylabel('counts')
        plt.grid()
        plt.axhline(ncounts, color='k')
        plt.axhline(ncounts - ncounts**0.5, color='k', ls="--")
        plt.axhline(ncounts + ncounts**0.5, color='k', ls="--")
    suptitle = savepath.split('/')[-2]
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(savepath + 'rankplot.png')

    #plot predictions
    if ndim > 5: fig, ax = plt.subplots(ndim//5, 5, figsize=(15, 4*ndim//5))
    else: fig, ax = plt.subplots(1, 5, figsize=(15, 4))
    for j in range(ndim):
        ax.flatten()[j].errorbar(trues[:, j], mus[:, j], stds[:, j], fmt="none", elinewidth=0.5, alpha=0.5)
        #if j == 0 : ax.flatten()[0].set_ylabel(lbls[iss], fontsize=12)
        ax.flatten()[j].plot(y[:, j], y[:, j], 'k.', ms=0.2, lw=0.5)
        ax.flatten()[j].grid(which='both', lw=0.5)
        ax.flatten()[j].set_title(titles[j], fontsize=12)
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(savepath + 'predictions.png')
    

