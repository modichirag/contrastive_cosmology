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
parser.add_argument('--nhod', type=int, default=20, help='number of HOD sims')
parser.add_argument('--suffix', type=str, default="broad", help='suffix for parent folder with z and nnbar')
parser.add_argument('--suffix2', type=str, default="", help='suffix for subfolder with model name')
parser.add_argument('--rewrite', type=int, default=0, help='rewrite files which already exist')
args = parser.parse_args()



m_hod = args.model
id0, id1, nhod = args.id0, args.id1, args.nhod
zred = args.z
#
bs, nc = 1000, 256
pm = ParticleMesh(Nmesh=[nc, nc, nc], BoxSize=bs, dtype='f8')
data_dir = '/mnt/ceph/users/cmodi/contrastive/data/z%02d/'%(zred*10)
if args.suffix != "": data_dir = data_dir[:-1] + '-%s/'%args.suffix
os.makedirs(data_dir, exist_ok=True)
if args.suffix2 == "": data_dir = data_dir + '%s/'%args.model
else: data_dir = data_dir + '%s/'%(args.model + "-" + args.suffix2)
os.makedirs(data_dir, exist_ok=True)
#


for i_lhc in range(id0, id1):
    print('LHC %i' % i_lhc)
    # read in halo catalog
    halos = Halos.Quijote_LHC_HR(i_lhc, z=zred)   
    save_dir = data_dir + '%04d/'%i_lhc
    os.makedirs(save_dir, exist_ok=True)        

    #
    if args.rewrite: do_hod = True
    else: do_hod = False
    for i_hod in range(nhod):
        if not os.path.isfile(save_dir + 'power_%d.npy'%i_hod) : do_hod = True
    if not os.path.isfile(save_dir + 'power_ell.npy') : do_hod = True
    if not do_hod: continue
    #
    hods = hodtools.sample_HOD_broad(args.model, nhod, i_lhc)
    print(hods.shape)
    #
    ps, ngals, gals, pmus, pells = [], [], [], [], []
    for i_hod in range(nhod): 
        print('  HOD %i' % i_hod)
        # sample HOD
        iseed = args.seed+i_hod
        _hod = hods[i_hod]
        theta_hod = {'logMmin': _hod[0], 'sigma_logM': _hod[1], 'logM0': _hod[2], 'logM1': _hod[3], 'alpha': _hod[4]}
        np.save(save_dir+'hodp_%d'%iseed, theta_hod)
        hod = Galaxies.hodGalaxies(halos, theta_hod, seed=0, hod_model=m_hod)
        #hod.save(save_dir+'cat_%i.bf' % iseed) 
        galsum = hodtools.galaxy_summary(hod, bs=bs, filename=save_dir+'gals_%i.json'%iseed)
        print(theta_hod)
        print(galsum)
        #
        ngals.append(galsum['number density'])
        k, p = hodtools.get_power(hod, pm)
        np.save(save_dir+"power_%d"%iseed, p)
        k, pmu, pell = hodtools.get_power_rsd(hod, pm, Nmu=6)
        np.save(save_dir+"power_rsd_%d"%iseed, pmu)
        np.save(save_dir+"power_ell_%d"%iseed, pell)
        ps.append(p)
        pmus.append(pmu)
        pells.append(pell)
        gals.append([galsum['total'], galsum['centrals'], galsum['satellites']])
        #hods.append([theta_hod[key] for key in theta_hod.keys()])

    np.save(save_dir + "power", np.array(ps))
    np.save(save_dir + "power_rsd", np.array(pmus))
    np.save(save_dir + "power_ell", np.array(pells))
    np.save(save_dir + "gals", np.array(gals))
    np.save(save_dir + "hodp", np.array(hods)) 
    #Make PS figure
    print("mean number denisty : ", np.mean(ngals))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for i in range(len(ps)):
        ax[0].plot(ps[i][0], ps[i][1])
        #ax[1].plot(ps[i][0], ps[i][2])
    for axis in ax:
        axis.loglog()
        axis.grid(which='both')
    plt.tight_layout()
    plt.savefig(save_dir + 'pk.png')
