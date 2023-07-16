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
from pyspectrum.pyspectrum import Bk_periodic


bs = 2000
nc = 720
kf = 2*np.pi/bs

 
for i in range(1, 11):
    infile = h5.File("../../data/mock_lcdm_redshift-space_%d.h5"%i, 'r')
    galaxies = infile['galaxies']
    x     = galaxies['x'] #each in units of comoving-Mpc/h
    y     = galaxies['y']
    z_los = galaxies['z_los']
    pos = np.stack([x, y, z_los], axis=1)
    infile.close()
    
    # ##################################################
    
    start = time.time()
    bspace = Bk_periodic(pos.T, Lbox=bs, Ngrid=nc, step=3*2, Ncut=3*2, Nmax=26, silent=False, nthreads=32)
    print("Total time taken: ", time.time() - start)
    
    # print(bspace)
    save_data = np.array([bspace['i_k1']*kf, bspace['i_k2']*kf, bspace['i_k3']*kf, bspace['b123'], bspace['q123']])
    np.save('../../data/bspec_redshift-space_%d'%i, save_data)
    
