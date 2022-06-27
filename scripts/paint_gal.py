import numpy as np
import nbodykit
from nbodykit.lab import BigFileCatalog
from pmesh.pm import ParticleMesh, RealField
from nbodykit.lab import FFTPower, ProjectedFFTPower, ArrayMesh
import tools
import sys, os

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--id0', type=int, default=0, help='sim number to start painting from')
parser.add_argument('--id1', type=int, default=2000, help='sim number to paint upto')
args = parser.parse_args()


nc = 256
bs = 1000
pm = ParticleMesh(Nmesh=[nc, nc, nc], BoxSize=bs, dtype='f8')
    

savefolder = "/mnt/ceph/users/cmodi/simbig/mesh//N%04d/"%nc
path = '/mnt/ceph/users/cmodi/simbig/catalogs/hod.quijote_LH%d.z0p0.zheng07%s.%d.bf'
#path = '/mnt/ceph/users/fvillaescusa/Quijote/Halos/FoF/latin_hypercube_nwLH/%d//' #folder hosting the catalogue
snapnum = 4
redshift = 0.0

for isim in range(args.id0, args.id1):
    savepath = savefolder + '%04d/'%isim
    os.makedirs(savepath, exist_ok=True)
    if isim%10 == 0: print(isim)
    numds, galtypes = [], []
    for j in range(3):
        for suff in ["", "_ab"]:
            print(isim, j, suff)
            #
            f = BigFileCatalog(path%(isim, suff, j))
            numd = f['Position'].shape[0]/bs**3 
            print('Number density = %0.2e'%numd)
            numds.append(["%d"%j + suff, numd])
            galtype = np.unique(f['gal_type'].compute(), return_counts=True)
            galtypes.append(["%d"%j + suff, galtype])
            #
            gal = pm.paint(f['Position'].compute())
            np.save(savepath + "field_zheng07%s.%d"%(suff, j), gal)
            #
            mesh = gal / gal.cmean() - 1
            ps = FFTPower(mesh, mode='1d').power.data
            k, p = ps['k'], ps['power'].real
            np.save(savepath + "power_zheng07%s.%d"%(suff, j), np.stack([k, p]).T)
            #
            gal_comp = tools.cic_compensation(gal)
            mesh = gal_comp / gal_comp.cmean() - 1
            ps = FFTPower(mesh, mode='1d').power.data
            k, p = ps['k'], ps['power'].real
            np.save(savepath + "compensated_power_zheng07%s.%d"%(suff, j), np.stack([k, p]).T)
            #
            del f, gal, gal_comp, mesh

    print(numds)
    print(galtypes)
    np.save(savepath + 'number_density', numds)
    np.save(savepath + 'gal_type', galtypes)
    
