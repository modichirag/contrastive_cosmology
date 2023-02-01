'''

module to inferace with different N-body catalogs incl. Quijote 


'''
import os
import numpy as np 
import quijote as Quijote 
import nbodykit.lab as NBlab


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



def FastPM_LHC_HR(i, z=1.0, finder='FoF'): 
    ''' Read halo catalog from the high resolution Quijote LHC. 


    Parameters
    ---------- 
    i : int 
        ith realization of the Quijote LHC simulations 

    z : float
        redshift of the halo catalog. Quijote halo catalogs are available at
        z = 0, 0.5, 1.
    
    finder : str
             Halo finder, FoF only
    Return
    ------
    cat : nbodykit.lab.HaloCatalog 
        Quijote HR LHC halo catalog  
    '''
    # directory that contains the FastPM LHC HR
    if finder == 'Rockstar':
        print("Rockstar catalogs for FastPM simulations are not available")
        raise NotImplementedError
    
    # convert z_to_a
    if z in [1.0, 0.5, 0.0]:
        a = "%0.4f"%(1/(1+z)) 
    else:
        print("FastPM catalogs only have snapshots at z=0, 0.5 and 1")
        raise NotImplementedError

    #Proceed if checks are satisfied
    # look up cosmology of the LHC realization which is the same as Quijote
    Om, Ob, h, ns, s8 = Quijote_LHC_cosmo(i)
    
    # define cosmology; caution: we don't match sigma8 here
    bs = 1000.
    Mnu = 0.
    cosmo = NBlab.cosmology.Planck15.clone(
            h=h, 
            Omega0_b=Ob, 
            Omega0_cdm=Om - Ob,
            m_ncdm=[None, Mnu][Mnu > 0.], 
            n_s=ns) 
    Ol = 1. - Om 
    Hz = 100.0 * np.sqrt(Om * (1. + z)**3 + Ol) # km/s/(Mpc/h)

    if finder == 'FoF':
        cat = NBlab.BigFileCatalog(f'/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/FoF_fastpm/{i}/fof_{a}/LL-0.200/')
        cat['Mass'] = cat["Length"] *float(cat.attrs['M0'] * 1e10)
        # calculate velocity offset
        rsd_factor = (1. + z) / Hz
        cat['VelocityOffset'] = cat['Velocity'].compute()*rsd_factor

    cat = NBlab.HaloCatalog(cat, cosmo=cosmo, redshift=z, mdef='vir') 
    cat['Position'] = (cat['Position'].compute()%bs )
    cat.attrs['BoxSize'] = np.array([bs, bs, bs])
    cat.attrs['rsd_factor'] = rsd_factor
    cat.attrs['Om'] = Om
    cat.attrs['Ob'] = Ob
    cat.attrs['Ol'] = Ol
    cat.attrs['h'] = h 
    cat.attrs['ns'] = ns
    cat.attrs['s8'] = s8
    cat.attrs['Hz'] = Hz # km/s/(Mpc/h)
    return cat


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
