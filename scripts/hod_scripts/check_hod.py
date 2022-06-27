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
parser.add_argument('--rewrite', type=int, default=0, help='rewrite files which already exist')
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
else: data_dir = data_dir + '%s/'%(args.model + "-" + args.suffix2)
os.makedirs(data_dir, exist_ok=True)
#

def setup_hod(halos, nbar=nbar, satfrac=satfrac, bs=bs, alpha_fid=alpha_fid):
    if nbar != 0.:
        hmass = halos['Mass'].compute()
        numdhalos = hmass.size/bs**3
        numhalos_nbarf = int(nbar * bs**3 * (1-satfrac))
        print("Halo number density and halo fraction used : ",numdhalos/1e-4, numhalos_nbarf/hmass.size)
        #raise Exception if (numhalos_nbarf/hmass.size) # diagnostics if number of halos < number density
        #
        mcut = hmass[:numhalos_nbarf][-1]
        mcut = 10**(np.log10(mcut) + 0.1)  ##offset by log_sigma/2 to account for scatter
        masses = [hmass[-1], mcut]
        toret = [numdhalos/1e-4,
                 numhalos_nbarf/hmass.size, 
                 (hmass>mcut).sum()/hmass.size,
                 (hmass>10**(np.log10(mcut) + 0.8)).sum()/hmass.size,
                 (hmass>10**(np.log10(mcut) - 0.8)).sum()/hmass.size
             ]
        print(toret)
        return masses, toret

hfrac = []
for i_lhc in range(id0, id1):
    print('LHC %i' % i_lhc)
    # read in halo catalog
    halos = Halos.Quijote_LHC_HR(i_lhc, z=zred)

    masses, toret = setup_hod(halos)
    hfrac.append([i_lhc] + toret)

hfrac = np.array(hfrac)
np.save('./tmp/hfrac_z%02d_n%04d'%(zred*10, args.nbar*1e4), hfrac)
