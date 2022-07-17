'''

module to inferace with different N-body catalogs incl. Quijote 


'''
import os
import numpy as np 
import quijote as Quijote 



def Quijote_LHC_HR(i, z=1.0, finder='FoF'): 
    ''' Read halo catalog from the high resolution Quijote LHC. 


    Parameters
    ---------- 
    i : int 
        ith realization of the Quijote LHC simulations 

    z : float
        redshift of the halo catalog. Quijote halo catalogs are available at
        z = 0, 0.5, 1., 2., and 3.
    
    finder : str
             Halo finder, FoF or Rockstar
    

    Return
    ------
    cat : nbodykit.lab.HaloCatalog 
        Quijote HR LHC halo catalog  
    '''
    # directory that contains the Quijote LHC HR
    #halo_folder = "/mnt/ceph/users/fvillaescusa/Quijote/Halos/FoF/latin_hypercube_nwLH/%d/"%i
    if finder == 'FoF': halo_folder = "/mnt/ceph/users/fvillaescusa/Quijote/Halos/FoF/latin_hypercube/HR_%d/"%i
    elif finder == 'Rockstar': halo_folder = "/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/Rockstar/%d/"%i
    
    # look up cosmology of the LHC realization
    Om, Ob, h, ns, s8 = Quijote_LHC_cosmo(i)
    
    # read halo catalog 
    halos = Quijote.Halos(halo_folder, z, Om=Om, Ob=Ob, h=h, ns=ns, s8=s8, Mnu=0., finder=finder)
    return halos


def Quijote_fiducial_HR(i, z=1.0, finder='FoF'): 
    ''' Read halo catalog from the high resolution Quijote at fiducial cosmology. 


    Parameters
    ---------- 
    i : int 
        ith realization of the Quijote LHC simulations 

    z : float
        redshift of the halo catalog. Quijote halo catalogs are available at
        z = 0, 0.5, 1., 2., and 3.

    finder : str
             Halo finder, FoF or Rockstar
    
    Return
    ------
    cat : nbodykit.lab.HaloCatalog 
        Quijote HR fiducial halo catalog  
    '''
    # directory that contains the Quijote LHC HR
    halo_folder = "/mnt/ceph/users/cmodi/Quijote/fiducial/halos/FoF/%d/"%i
    
    # fiducial cosmology (Villaesuca-Navarro+2020) 
    Om, Ob, h, ns, s8 = Quijote_fiducial_cosmo() 
    
    # read halo catalog 
    halos = Quijote.Halos(halo_folder, z, Om=Om, Ob=Ob, h=h, ns=ns, s8=s8, Mnu=0., finder=finder)
    return halos


def Quijote_LHC_cosmo(i): 
    ''' cosmology look up for LHC realization i at redshift z 

    outputs: Omega_m, Omega_l, h, ns, s8

    '''
    fcosmo = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
            'quijote_lhc_cosmo.txt')
    #fcosmo = 'quijote_lhc_cosmo.txt'
    # Omega_m, Omega_l, h, ns, s8
    cosmo = np.loadtxt(fcosmo, unpack=True, usecols=range(5)) 

    Om = cosmo[0][i]
    Ob = cosmo[1][i]
    h  = cosmo[2][i]
    ns = cosmo[3][i]
    s8 = cosmo[4][i]

    return Om, Ob, h, ns, s8


def Quijote_fiducial_cosmo(): 
    ''' fiducial cosmology 

    '''
    # Omega_m, Omega_l, h, ns, s8
    Om = 0.3175
    Ob = 0.049
    h  = 0.6711
    ns = 0.9624
    s8 = 0.834

    return Om, Ob, h, ns, s8
