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
import tools, hodtools

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--z', type=float, help='redshift')
parser.add_argument('--id0', type=int, default=0, help='first quijote seed')
parser.add_argument('--id1', type=int, default=1, help='last quijote seed')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999')
parser.add_argument('--finder', type=str, default="fof", help='which halo finder')
parser.add_argument('--suffix', type=str, default="", help='suffix for parent folder with z and nnbar')
parser.add_argument('--fiducial', type=int, default=0, help='for fiducial simulations')
parser.add_argument('--simulation', type=str, default='quijote', help='for fiducial simulations')
args = parser.parse_args()


id0, id1 = args.id0, args.id1
zred = args.z
if zred is None:
    print("\nERROR: Need to specify redshift\n")
    sys.exit()
else:
    print(f"\nRun with args: {args}\n")

#
bs, nc = 1000, 256
pm = ParticleMesh(Nmesh=[nc, nc, nc], BoxSize=bs, dtype='f8')
#data_dir = '/mnt/ceph/users/cmodi/contrastive/data/z%02d/halos/'%(zred*10)

if args.fiducial:
    data_dir = '/mnt/ceph/users/cmodi/Quijote/fiducial/power_spectrum/'
else:
    data_dir = '/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/power_spectrum/'

subfolder = ""
if args.finder == "fof": subfolder =  'FoF'
else: subfolder =  'Rockstar'
if args.simulation == "fastpm" : subfolder = subfolder + "_fastpm"
savepath = data_dir + f"/{subfolder}/"
print(f"Save data in {savepath}")
os.makedirs(savepath, exist_ok=True)
print()
#

def get_power_and_save(i_lhc, halos, pm, numd=None):

    save_dir = savepath + '%04d/'%i_lhc
    os.makedirs(save_dir, exist_ok=True)
    if numd is None:
        num = None
        suffix = ""
    else:
        num = int(numd * bs**3)
        suffix = f"_n{numd:.1e}"
    
    #
    k, ph = hodtools.get_power(halos, pm, num=num) 
    np.save(save_dir + f"k", k)
    np.save(save_dir + f"power{suffix}", np.array([k, ph]).T)
    
    k, pmu, pell = hodtools.get_power_rsd(halos, pm, num=num, Nmu=12)
    np.save(save_dir + f"power_rsd{suffix}", pmu.real)
    np.save(save_dir + f"power_ell{suffix}", pell.real)
    


for i_lhc in range(id0, id1):

    print('LHC %i' % i_lhc)
    
    # read in halo catalog
    if args.simulation == 'fastpm':
        if args.fiducial: 
            print("No fiducial runs for FastPM")
            raise NotImplementedError
        else: 
            halos = Halos.FastPM_LHC_HR(i_lhc, z=zred, finder=args.finder)
    else:
        if args.fiducial: 
            halos = Halos.Quijote_fiducial_HR(i_lhc, z=zred, finder=args.finder)
        else:
            halos = Halos.Quijote_LHC_HR(i_lhc, z=zred, finder=args.finder)

    halos = halos.sort('Mass', reverse=True)
    print("Halo read")

    for numd in [None, 1e-4, 3e-4, 5e-4, 1e-3]:
        get_power_and_save(i_lhc, halos, pm, numd=numd)
    
