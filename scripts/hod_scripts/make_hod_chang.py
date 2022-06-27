'''
Generate galaxy catalog
'''
import os, sys 
import numpy as np 
import argparse, json
from pmesh.pm import ParticleMesh, RealField
from nbodykit.lab import FFTPower, ProjectedFFTPower, ArrayMesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#
sys.path.append('../../src/')
import halos as Halos
import galaxies as Galaxies
import tools, hodtools

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, help='HOD model name')
parser.add_argument('--z', type=float, help='redshift')
parser.add_argument('--id0', type=int, default=0, help='first quijote seed')
parser.add_argument('--id1', type=int, default=2000, help='last quijote seed')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999')
parser.add_argument('--nhod', type=int, default=10, help='number of HOD sims')
parser.add_argument('--alpha', type=float, default=0.7, help='alpha, slope of satellites')
parser.add_argument('--satfrac', type=float, default=0.2, help='satellite fraction')
parser.add_argument('--nbar', type=float, default=0.0001, help='numbder density of galaxies')
parser.add_argument('--suffix', type=str, default="", help='suffix for parent folder with z and nnbar')
parser.add_argument('--suffix2', type=str, default="", help='suffix for subfolder with model name')
args = parser.parse_args()


m_hod = args.model
id0, id1, nhod = args.id0, args.id1, args.nhod
zred, alpha_fid = args.z, args.alpha
nbar, satfrac = args.nbar, args.satfrac
#
bs, nc = 1000, 256
pm = ParticleMesh(Nmesh=[nc, nc, nc], BoxSize=bs, dtype='f8')
if nbar != 0: data_dir = '/mnt/ceph/users/cmodi/contrastive/data/z%02d-N%04d/'%(zred*10, nbar/1e-4)
else: data_dir = '/mnt/ceph/users/cmodi/contrastive/data/z%02d/'%(zred*10)
if args.suffix != "": data_dir = data_dir[:-1] + '-%s/'%args.suffix
os.makedirs(data_dir, exist_ok=True)
if args.suffix2 == "": data_dir = data_dir + '%s/'%args.model
else: data_dir = data_dir + '%s/'%(args.model + "-%s"%args.suffix2)
os.makedirs(data_dir, exist_ok=True)
#


def sample_HOD(): 
    ''' sample HOD value based on priors set by Parejko+(2013)
    'logMmin': 13.03, 
    'sigma_logM':0.38,
    'logM0': 13.27, 
    'logM1': 14.08, 
    'alpha': 0.76
    '''
    hod_fid = np.array([13.03, 0.38, 13.27, 14.08, 0.76])
    #hod_fid = np.array([_hod_fid['logMmin'], _hod_fid['sigma_logM'], _hod_fid['logM0'], _hod_fid['logM1'], _hod_fid['alpha']])
    dhod = 2*np.array([0.029, 0.06, 0.13, 0.06, 0.18])

    _hod = hod_fid + dhod * np.random.uniform(-0.5, 0.5, size=(5))

    return {'logMmin': _hod[0], 'sigma_logM': _hod[1], 'logM0': _hod[2], 'logM1': _hod[3], 'alpha': _hod[4]}


def save_power(f, savepath, num=None, suff=""):
    if num is None: gal = pm.paint(f['Position'].compute())
    else: gal = pm.paint(f['Position'][:num].compute())
    #np.save(savepath + "field_zheng07%s.%d"%(suff, j), gal)
    #
    mesh = gal / gal.cmean() - 1
    ps = FFTPower(mesh, mode='1d').power.data
    k, p = ps['k'], ps['power'].real
    np.save(savepath%"power"+suff, np.stack([k, p]).T)
    #
    gal_comp = tools.cic_compensation(gal, order=2)
    mesh = gal_comp / gal_comp.cmean() - 1
    ps = FFTPower(mesh, mode='1d').power.data
    k, pc = ps['k'], ps['power'].real
    np.save(savepath%"compensated_power"+suff, np.stack([k, pc]).T)
    return k, p, pc


for i_lhc in range(id0, id1):
    print('LHC %i' % i_lhc)
    # read in halo catalog
    halos = Halos.Quijote_LHC_HR(i_lhc, z=zred)

    ps = []
    ngals = []
    for i_hod in range(nhod): 
        print('  HOD %i' % i_hod)
        
        save_dir = data_dir + '%04d/'%i_lhc
        os.makedirs(save_dir, exist_ok=True)        
        if i_hod == 0:
            save_power(halos, save_dir+"%s_h")
            save_power(halos, save_dir+"%s_h1e-4", num=int(1e-4*bs**3))
                        
                        
        # sample HOD
        iseed = args.seed+i_hod
        np.random.seed(iseed+i_lhc*9999)
        theta_hod = sample_HOD()
        print(theta_hod)
        np.save(save_dir+'hodp_%d'%iseed, theta_hod)
        hod = Galaxies.hodGalaxies(halos, theta_hod, seed=0, hod_model=m_hod)
        #hod.save(save_dir+'cat_%i.bf' % iseed) 

        #calculate some summary numbers on galaxy types
        galsum = hodtools.galaxy_summary(hod, bs=bs, filename=save_dir+'gals_%i.json'%iseed)
        print(galsum)
        ngals.append(galsum['number density'])
        ps.append(save_power(hod, save_dir+"/%s"+"_%d"%iseed))
        
    #Make PS figure
    print("mean number denisty : ", np.mean(ngals))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for i in range(len(ps)):
        ax[0].plot(ps[i][0], ps[i][1])
        ax[1].plot(ps[i][0], ps[i][2])
    for axis in ax:
        axis.loglog()
        axis.grid(which='both')
    plt.tight_layout()
    plt.savefig(save_dir + 'pk.png')
