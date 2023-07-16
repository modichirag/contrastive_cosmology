import numpy as np
import h5py as h5
import os, sys, time
from nbodykit.lab import FFTPower, BigFileCatalog
from pmesh.pm import ParticleMesh, RealField
import bskit_main as bskit
#
sys.path.append('../../src/')
import halos as Halos
import galaxies as Galaxies
import tools, hodtools

from nbodykit.lab import FFTPower, BigFileCatalog
from pmesh.pm import ParticleMesh, RealField
from pyspectrum.pyspectrum import Bk_periodic



nc = 512
bs = 2000
pm = ParticleMesh(Nmesh=[nc, nc, nc], BoxSize=bs, dtype='f8')
kf = 2*np.pi/bs
kfid = np.load('/mnt/ceph/users/cmodi/contrastive/data/k-256.npy')

allpells = []
allngals = []
savepath = '/mnt/home/cmodi/Research/Projects/contrasative_cosmology/data/stats/'

for i in range(1, 11):

    print()
    print("For data-cube : ", i)
    
    infile = h5.File(f"/mnt/home/cmodi/Research/Projects/contrasative_cosmology/data/mock_lcdm_redshift-space_{i}.h5", 'r')
    galaxies = infile['galaxies']
    x     = galaxies['x'] #each in units of comoving-Mpc/h
    y     = galaxies['y']
    z_los = galaxies['z_los']
    infile.close()
    
    pos = np.stack([x, y, z_los], axis=1)
    ngal = x.size
    ngal_rescaled = ngal/2**3

    #
    print("Measure power spectrum")
    mesh = pm.paint(pos)
    pkrsd = FFTPower(mesh/mesh.cmean(), mode='2d', Nmu=12, poles=[0, 2, 4], dk=2*np.pi/1000)
    pells = np.array([pkrsd.poles.data['power_%d'%i].real for i in [0, 2, 4]]).T
    np.save(f'{savepath}/power_ell_{i}.npy', pells)
    np.save(f'{savepath}/ngal_{i}.npy', ngal_rescaled)

    #
    print("Measure bispectrum")
    start = time.time()
    bispec = Bk_periodic((pos).T, Lbox=bs, Ngrid=720, step=6, Ncut=6, Nmax=26, silent=False, nthreads=32)
    print("Total time taken: ", time.time() - start)
    
    save_data = np.array([bispec['i_k1']*kf, bispec['i_k2']*kf, bispec['i_k3']*kf, bispec['b123'], bispec['q123'], bispec['b123_sn']])
    np.save(f'{savepath}/bispec_{i}.npy', save_data)

    
