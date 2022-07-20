import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


###
def plot_ranks_histogram(ranks, nbins=10, npars=5, titles=None, savepath=None, figure=None, suffix=""):

    ncounts = ranks.shape[0]/nbins
    if titles is None: titles = [""]*npars

    if figure is None: 
        fig, ax = plt.subplots(1, npars, figsize=(npars*3, 4))
    else:
        fig, ax = figure
    
    for i in range(npars):
        ax[i].hist(np.array(ranks)[:, i], bins=nbins)
        ax[i].set_title(titles[i])
        ax[0].set_ylabel('counts')

    for axis in ax:
        axis.set_xlabel('rank')
        axis.grid(visible=True)
        axis.axhline(ncounts, color='k')
        axis.axhline(ncounts - ncounts**0.5, color='k', ls="--")
        axis.axhline(ncounts + ncounts**0.5, color='k', ls="--")
    plt.tight_layout()
    if savepath is not None:
        suptitle = savepath.split('/')[-2]
        plt.suptitle(suptitle)
        plt.tight_layout()
        plt.savefig(savepath + 'rankplot%s.png'%suffix)
    return fig, ax


###
def plot_coverage(ranks, npars=5, titles=None, savepath=None, figure=None, suffix="", label="", plotscatter=True):

    ncounts = ranks.shape[0]
    if titles is None: titles = [""]*npars
    unicov = [np.sort(np.random.uniform(0, 1, ncounts)) for j in range(20)]

    if figure is None: 
        fig, ax = plt.subplots(1, npars, figsize=(npars*3, 4))
    else:
        fig, ax = figure
    
    for i in range(npars):
        xr = np.sort(ranks[:, i])
        xr = xr/xr[-1]
        cdf = np.arange(xr.size)/xr.size
        if plotscatter: 
            for j in range(len(unicov)): ax[i].plot(unicov[j], cdf, lw=1, color='gray', alpha=0.2)
        ax[i].plot(xr, cdf, lw=2, label=label)
        ax[i].set_title(titles[i])
        ax[0].set_ylabel('CDF')

    for axis in ax:
        #axis.set_xlabel('rank')
        axis.grid(visible=True)

    plt.tight_layout()
    if savepath is not None:
        suptitle = savepath.split('/')[-2]
        plt.suptitle(suptitle)
        plt.tight_layout()
        plt.savefig(savepath + 'coverage%s.png'%suffix)
    return fig, ax


###
def plot_posterior(x, y, posterior, nsamples=1000, titles=None, savename="", ndim=None):
    
    posterior_samples = posterior.sample((nsamples,), x=torch.from_numpy(x.astype('float32'))).detach().numpy()
    mu, std = posterior_samples.mean(axis=0), posterior_samples.std(axis=0)
    p = y
    if ndim is None: ndim = len(y)
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

    if savename != "": plt.savefig(savename)
    return fig, axar


###
def get_ranks(x, y, posterior, test_frac=1.0, nsamples=500):
    ndim = y.shape[1]
    ndim = min(ndim, posterior.sample((1,),  x=torch.from_numpy(x[0].astype('float32')), 
                                             show_progress_bars=False).detach().numpy().shape[1])

    ranks = []
    mus, stds = [], []
    trues = []
    for ii in range(x.shape[0]):
        if ii%1000 == 0: print("Test iteration : ",ii)
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
    return trues, mus, stds, ranks


###
def test_diagnostics(x, y, posterior, nsamples=500, titles=None, savepath="", test_frac=1.0, suffix=""):

    ndim = y.shape[1]
    ndim = min(ndim, posterior.sample((1,),  x=torch.from_numpy(x[0].astype('float32')), 
                                             show_progress_bars=False).detach().numpy().shape[1])
    if titles is None: titles = []*ndim

    trues, mus, stds, ranks = get_ranks(x, y, posterior, test_frac, nsamples=nsamples)

    #plot ranks and coverage
    _ = plot_ranks_histogram(ranks, titles=titles, savepath=savepath, suffix=suffix)
    _ = plot_coverage(ranks, titles=titles, savepath=savepath, suffix=suffix)

    #plot predictions
    if ndim > 5: fig, ax = plt.subplots(ndim//5, 5, figsize=(15, 4*ndim//5))
    else: fig, ax = plt.subplots(1, 5, figsize=(15, 4))
    for j in range(min(ndim, len(ax.flatten()))):
        ax.flatten()[j].errorbar(trues[:, j], mus[:, j], stds[:, j], fmt="none", elinewidth=0.5, alpha=0.5)
        #if j == 0 : ax.flatten()[0].set_ylabel(lbls[iss], fontsize=12)
        ax.flatten()[j].plot(y[:, j], y[:, j], 'k.', ms=0.2, lw=0.5)
        ax.flatten()[j].grid(which='both', lw=0.5)
        ax.flatten()[j].set_title(titles[j], fontsize=12)
    suptitle = savepath.split('/')[-2]
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(savepath + 'predictions%s.png'%suffix)
    



###
def test_fiducial(x, y, posterior, nsamples=500, rankplot=True, titles=None, savepath="", test_frac=1.0, suffix=""):

    ndim = y.shape[1]
    ndim = min(ndim, posterior.sample((1,),  x=torch.from_numpy(x[0].astype('float32')), 
                                             show_progress_bars=False).detach().numpy().shape[1])
    if titles is None: titles = []*ndim

    trues, mus, stds, ranks = get_ranks(x, y, posterior, test_frac, nsamples=nsamples)

    # ranks = []
    # mus, stds = [], []
    # trues = []
    # for ii in range(x.shape[0]):
    #     if ii%100 == 0: print("Test iteration : ",ii)
    #     if np.random.uniform() > test_frac: continue
    #     posterior_samples = posterior.sample((nsamples,),
    #                                          x=torch.from_numpy(x[ii].astype('float32')), 
    #                                          show_progress_bars=False).detach().numpy()
    #     mu, std = posterior_samples.mean(axis=0), posterior_samples.std(axis=0)
    #     rank = [(posterior_samples[:, i] < y[ii, i]).sum() for i in range(ndim)]
    #     mus.append(mu)
    #     stds.append(std)
    #     ranks.append(rank)
    #     trues.append(y[ii])
    # mus, stds, ranks = np.array(mus), np.array(stds), np.array(ranks)
    # trues = np.array(trues)

    #plot ranks
    _ = plot_ranks_histogram(ranks, titles=titles, savepath=savepath, suffix=suffix)
    _ = plot_coverage(ranks, titles=titles, savepath=savepath, suffix=suffix)

    # plt.figure(figsize=(15, 4))
    # nbins = 10
    # ncounts = ranks.shape[0]/nbins
    # for i in range(5):
    # #     plt.hist(np.array(ranks)[:, i], bins=10, histtype='step', lw=2)
    #     plt.subplot(151 + i)
    #     plt.hist(np.array(ranks)[:, i], bins=nbins)
    #     plt.title(titles[i])
    #     plt.xlabel('rank')
    #     plt.ylabel('counts')
    #     plt.grid()
    #     plt.axhline(ncounts, color='k')
    #     plt.axhline(ncounts - ncounts**0.5, color='k', ls="--")
    #     plt.axhline(ncounts + ncounts**0.5, color='k', ls="--")
    # suptitle = savepath.split('/')[-2]
    # plt.suptitle(suptitle)
    # plt.tight_layout()
    # plt.savefig(savepath + 'rankplot%s.png'%suffix)

    # #plot coverage
    # plt.figure(figsize=(15, 4))
    # ncounts = ranks.shape[0]/nbins
    # unicov = [np.sort(np.random.uniform(0, 1, ranks.shape[0])) for j in range(20)]
    # for i in range(5):
    #     plt.subplot(151 + i)
    #     xr = np.sort(ranks[:, i])
    #     xr = xr/xr[-1]
    #     cdf = np.arange(xr.size)/xr.size
    #     for j in range(len(unicov)):
    #         plt.plot(unicov[j], cdf, lw=1, color='gray', alpha=0.2)
    #     plt.plot(xr, cdf, lw=2)
    #     plt.title(titles[i])
    #     plt.xlabel('rank')
    #     plt.ylabel('CDF')
    #     plt.grid()
    # suptitle = savepath.split('/')[-2]
    # plt.suptitle(suptitle)
    # plt.tight_layout()
    # plt.savefig(savepath + 'coverage%s.png'%suffix)

    #plot predictions
    if ndim > 5: fig, ax = plt.subplots(ndim//5, 5, figsize=(15, 4*ndim//5))
    else: fig, ax = plt.subplots(1, 5, figsize=(15, 4))
    for j in range(min(ndim, len(ax.flatten()))):
        axis = ax.flatten()[j]
        axis.errorbar(np.arange(mus.shape[0]), mus[:, j], stds[:, j], fmt="none", elinewidth=0.5, alpha=0.5)
        axis.axhline(trues[0, j], color='k')
        #axis.plot(y[:, j], y[:, j], 'k.', ms=0.2, lw=0.5)
        axis.grid(which='both', lw=0.5)
        axis.set_title(titles[j], fontsize=12)
    suptitle = savepath.split('/')[-2]
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.savefig(savepath + 'predictions%s.png'%suffix)
    

