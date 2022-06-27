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
sys.path.append('../src/')
import halos as Halos
import galaxies as Galaxies
import tools, hodtools
import readfof

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, help='HOD model name')
parser.add_argument('--z', type=float, help='redshift')
parser.add_argument('--i0', type=int, default=0, help='first quijote seed')
parser.add_argument('--i1', type=int, default=2000, help='last quijote seed')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999')
parser.add_argument('--nhod', type=int, default=10, help='number of HOD sims')
parser.add_argument('--alpha', type=float, default=0.7, help='alpha, slope of satellites')
parser.add_argument('--satfrac', type=float, default=0.2, help='satellite fraction')
parser.add_argument('--nbar', type=float, default=0.0001, help='numbder density of galaxies')
args = parser.parse_args()


m_hod = args.model
i0, i1, nhod = args.i0, args.i1, args.nhod
zred, alpha_fid = args.z, args.alpha
nbar, satfrac = args.nbar, args.satfrac
#
bs, nc = 1000, 256
pm = ParticleMesh(Nmesh=[nc, nc, nc], BoxSize=bs, dtype='f8')
data_dir = '/mnt/ceph/users/cmodi/contrastive/data/z%02d-N%04d/'%(zred, nbar/1e-4)
os.makedirs(data_dir, exist_ok=True)
data_dir = data_dir + '%s/'%args.model
#

def setup_hod(halos, nbar=nbar, satfrac=satfrac, bs=bs, alpha_fid=alpha_fid):
    hmass = halos['Mass'].compute()
    numdhalos = hmass.size/bs**3
    numhalos_nbarf = int(nbar * bs**3 * (1-satfrac))
    print("Halo number density and halo fraction used : ",numdhalos/1e-4, numhalos_nbarf/hmass.size)
    #raise Exception if (numhalos_nbarf/hmass.size) # diagnostics if number of halos < number density
    #
    mcut = hmass[:numhalos_nbarf][-1]
    nsat = satfrac * nbar * bs**3
    mdiff = (hmass - mcut + mcut*1e-3)[:numhalos_nbarf] ** alpha_fid
    msum = mdiff.sum()/nsat
    m1 = msum**(1/alpha_fid)
    #mcut = 10**(np.log10(mcut) + 0.2)  ##offset by log_sigma/2 to account for scatter
    print("M1, mcut : ", np.log10(m1), np.log10(mcut))
    return mcut, m1


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


for i_lhc in range(i0, i1):
    print('LHC %i' % i_lhc)
    # read in halo catalog
    save_dir = data_dir + '%04d/'%i_lhc
    if not os.path.isfile(save_dir + "power_h.npy") or not os.path.isfile(save_dir + "power_h1e-4.npy"): 
        halos = Halos.Quijote_LHC_HR(i_lhc, z=zred)
        hpath = '/mnt/ceph/users/fvillaescusa/Quijote/Halos/FoF/latin_hypercube/HR_%d//' #folder hosting t
        catalog = hpath%i_lhc
        if zred == 1.0: snapnum = 2 
        if zred == 0.0: snapnum = 4
        FoF = readfof.FoF_catalog(catalog, snapnum, long_ids=False,
                                  swap=False, SFR=False, read_IDs=False)
        pos = FoF.GroupPos/1e3            #Halo positions in Mpc/h
        mass  = FoF.GroupMass*1e10          #Halo masses in Msun/h
        Npart = FoF.GroupLen    

        print(np.allclose(halos['Position'][:5].compute(), pos[:5]))

        ps = []

        save_power(halos, save_dir+"%s_h")
        save_power(halos, save_dir+"%s_h1e-4", num=int(1e-4*bs**3))
    else:
        print("halo power exists for %d"%i_lhc)
    #     #continue
    #     #if not os.path.isfile(fhod) or not os.path.isdir(fgal): 
    #     # sample HOD
    #     iseed = args.seed+i_hod
    #     theta_hod = hodtools.sample_conditional_HOD(m_hod, np.log10(mcut), m1=np.log10(m1), seed=iseed)
    #     np.save(save_dir+'hodp_%d'%iseed, theta_hod)
    #     hod = Galaxies.hodGalaxies(halos, theta_hod, seed=0, hod_model=m_hod)
    #     hod.save(save_dir+'cat_%i.bf' % iseed) 

    #     #calculate some summary numbers on galaxy types
    #     galsum = hodtools.galaxy_summary(hod, bs=bs, filename=save_dir+'gals_%i.json'%iseed)
    #     print(galsum)
    #     ps.append(save_power(hod, save_dir+"/%s"+"_%d"%iseed))
        
    # continue
    # #Make PS figure
    # fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    # for i in range(len(ps)):
    #     ax[0].plot(ps[i][0], ps[i][1])
    #     ax[1].plot(ps[i][0], ps[i][2])
    # for axis in ax:
    #     axis.loglog()
    #     axis.grid(which='both')
    # plt.tight_layout()
    # plt.savefig(save_dir + 'pk.png')
