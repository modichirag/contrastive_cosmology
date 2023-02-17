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


# infile = h5.File("../../data/mock_lcdm_redshift-space_I.h5", 'r')

# galaxies = infile['galaxies']
# x     = galaxies['x'] #each in units of comoving-Mpc/h
# y     = galaxies['y']
# z_los = galaxies['z_los']
# pos = np.stack([x, y, z_los], axis=1)
# infile.close()


# nc3 = 128
# bs = 2000
# pm3 = ParticleMesh(Nmesh=[nc3, nc3, nc3], BoxSize=bs, dtype='f8')
# mesh3 = pm3.paint(pos)



i_lhc = 0
finder = 'FoF'
m_hod = 'zheng07'
zred = 0.5
nbar = 4.e-4
bs = 1000
nc = 360
pm = ParticleMesh(Nmesh=[nc, nc, nc], BoxSize=bs, dtype='f8')
kf = 2*np.pi/bs

alpha = 0.7
satfrac = 0.2
num = int(nbar*bs**3)

#model suffix

halos = Halos.Quijote_fiducial_HR(i_lhc, z=zred, finder=finder)
halos = halos.sort('Mass', reverse=True)
hmass = halos['Mass'].compute()
print("Halo read. Mass: ", hmass)
hsize = hmass.shape[0]

#Setup HOD
mcut, m1 = hodtools.setup_hod(halos, nbar=nbar, satfrac=satfrac, bs=bs, alpha_fid=alpha)


iseed = 0
theta_hod = hodtools.sample_conditional_HOD(m_hod, np.log10(mcut), m1=np.log10(m1), seed=i_lhc*9999+iseed, ab_scatter=0.2)
        
#hod = Galaxies.hodGalaxies(halos, theta_hod, seed=0, hod_model=m_hod)
hodmodel = Galaxies.hodGalaxies_cache(halos, hod_model=m_hod)
hod = halos.populate(hodmodel, seed=0, **theta_hod)
hod.repopulate(**theta_hod)
        

los=[0, 0, 1]
pos = hod['Position'] + hod['VelocityOffset']*los    
pos = pos[:num].compute()
mesh = pm.paint(pos)
print(pos.shape)

# ###################################################
# # Try Shivam's code

# start = time.time()
# fftb = bskit.FFTBispectrum(mesh, Nmesh=nc, BoxSize=np.ones(3)*bs,
#                             dk=2*kf,
#                             kmin=1.5*kf,
#                             kmax=50*kf,
#                             # second=None,third=None,
#                             # num_lowk_bins=self.num_low_k_bins,dk_high=self.dk_high,
#                             # triangle_type=self.triangle_type,squeezed_bin_index=self.squeezed_bin_index,
#                             # isos_mult=self.isos_mult,isos_tol=self.isos_tol,
#                             # for_grid_info_only=self.for_grid_info_only
#                           )
# print("Time to set up: ", time.time() - start)
# print(fftb.attrs)
# print("\nEdge list")
# print(bskit.generate_bin_edge_list(fftb.attrs['kmin'],fftb.attrs['kmax'],
#                                             fftb.attrs['dk'],fftb.attrs['num_lowk_bins'],
#                                             fftb.attrs['dk_high']))
# print("\nTriangle list")
# print(bskit.generate_triangle_bin_list(fftb.attrs['kmin'],fftb.attrs['kmax'],
#                                             fftb.attrs['dk']))
# print()
# num_k_bins = len(bskit.generate_bin_edge_list(fftb.attrs['kmin'],fftb.attrs['kmax'],
#                                             fftb.attrs['dk'],fftb.attrs['num_lowk_bins'],
#                                             fftb.attrs['dk_high']))
# print("number of bins: ", num_k_bins)
# start_i = 0
# #end_i = num_k_bins
# end_i = 40
# fftb.measure_bispectrum(imin=start_i,imax=end_i,verbose=0)
# Bk_obj_fftb = fftb.b
# save_data = np.array([Bk_obj_fftb['k_mean'][:,0],Bk_obj_fftb['k_mean'][:,1],Bk_obj_fftb['k_mean'][:,2],
#                 Bk_obj_fftb['B']]).T
# print("Total time taken for nbkit: ", time.time() - start)
# np.save('bspec_nbkit', save_data)


# ##################################################
print("Try Chang's code next")

from pyspectrum.pyspectrum import Bk_periodic
start = time.time()
bspace = Bk_periodic(pos.T, Lbox=bs, Ngrid=nc, step=3, Ncut=3, Nmax=26, silent=False, nthreads=32)
print("Total time taken: ", time.time() - start)

# print(bspace)

save_data = np.array([bspace['i_k1']*kf, bspace['i_k2']*kf, bspace['i_k3']*kf, bspace['b123'], bspace['q123']])
np.save('bspec_chang', save_data)

# # start = time.time()
# # bspace = Bk_periodic(pos.T, Lbox=2000, Ngrid=360, Nmax=25, silent=False)
# # print("Total time taken: ", time.time() - start)

# # start = time.time()
# # bspace = Bk_periodic(pos.T, Lbox=2000, Ngrid=360, Nmax=40, silent=False)
# # print("Total time taken: ", time.time() - start)
