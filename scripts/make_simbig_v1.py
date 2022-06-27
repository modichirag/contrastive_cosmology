'''

script for constructing SIMBIG HOD catalogs for Quijote LHC simulations 
with a forward model of the BOSS LOWZ SGC sample 

v1 includes assembly bias and velocity bias parameters in the HOD 


'''
import os, sys 
import numpy as np 
from simbig import halos as Halos
from simbig import galaxies as Galaxies
from simbig import forwardmodel as FM

from simbig import util as UT 
from simbig import obs as CosmoObs

np.random.seed(1)

dat_dir = '/tigress/chhahn/simbig/cmass/'


def sample_HOD(typ='train'): 
    ''' sample HOD value based on priors set by Reid+(2014)
    '''
    if typ == 'train': frange = 1.
    elif typ == 'test': frange = 0.1

    _hod_fid = Galaxies.thetahod_literature('reid2014_cmass')
    hod_fid = np.array([
        _hod_fid['logMmin'], 
        _hod_fid['sigma_logM'], 
        _hod_fid['logM0'],
        _hod_fid['logM1'], 
        _hod_fid['alpha']])

    dhod = 2*np.array([0.029, 0.06, 0.13, 0.06, 0.18])

    _hod = hod_fid + dhod * np.random.uniform(-0.5 * frange, 0.5 * frange, size=(5))

    return {'logMmin': _hod[0], 'sigma_logM': _hod[1], 'logM0': _hod[2], 'logM1': _hod[3], 'alpha': _hod[4]}


def sample_theta_bias(typ='train'): 
    ''' sample assembly and velocity bias parameters
    '''
    if typ == 'train': 
        abias = np.clip(0.2 * np.random.normal(), -1., 1.) 
        conc = np.random.uniform(0.2, 2.0, size=1)
        eta_c = np.random.uniform(0., 0.7, size=1)
        eta_s = np.random.uniform(0.2, 2.0, size=1) 
    elif typ == 'test': 
        abias = np.clip(0.02 * np.random.normal(), -1., 1.) 
        conc = np.random.uniform(0.9, 1.1, size=1)
        eta_c = np.random.uniform(0., 0.1, size=1)
        eta_s = np.random.uniform(0.9, 1.1, size=1) 

    return np.array([abias, conc, eta_c, eta_s]) 


def train_mocks(i0, i1, overwrite=False): 
    ''' construct training data by randomly sampling HOD parameters and 
    constructing a mock galaxy survey 
    '''
    for i_lhc in range(i0, i1+1): 
        print('LHC %i' % i_lhc)
    
        # read in halo catalog
        halos = Halos.Quijote_LHC_HR_Rockstar(i_lhc, z=0.5)

        for i_hod in range(5):
            fgal = os.path.join(dat_dir, 'train', 
                    'hod.quijote_LH%i.z0p5.cmass_sgc.v1.%i.bf' % (i_lhc, i_hod))
            fhod = fgal.replace('.bf', '.npy')
            #fplk = fgal.replace('hod.', 'plk', 'plk.hod.').replace('.bf', '.dat') 
            fplk = fgal.replace('hod.', 'plk.hod.').replace('.bf', '.dat') 

            if os.path.isdir(fgal) and os.path.isfile(fhod) and not overwrite: 
                continue
            
            # HOD parameters            
            theta = sample_HOD(typ='train')
            abias, conc, eta_c, eta_s = sample_theta_bias()

            theta['mean_occupation_centrals_assembias_param1'] = abias
            theta['mean_occupation_satellites_assembias_param1'] = abias

            theta['conc_gal_bias.satellites'] = conc
            theta['eta_vb.centrals'] = eta_c
            theta['eta_vb.satellites'] = eta_s

            np.save(fhod, theta)

            # populate with HOD
            if i_hod == 0: 
                _Z07AB = Galaxies.VelAssembiasZheng07Model()
                Z07AB = _Z07AB.to_halotools(halos.cosmo, 
                        halos.attrs['redshift'], 
                        halos.attrs['mdef'], 
                        sec_haloprop_key='halo_nfw_conc')
                hod = halos.populate(Z07AB, **theta)
            else: 
                hod.repopulate(**theta)

            gals = FM.BOSS(
                    hod, 
                    sample='cmass-south', 
                    veto=True, 
                    fiber_collision=True, 
                    silent=True)
            gals.save(fgal)

            # construct corresponding randoms
            rand = FM.BOSS_randoms(gals, veto=True, sample='cmass-south') 

            # measure power spectrum 
            k, p0k, p2k, p4k = CosmoObs.Plk_survey(gals, rand,
                    Ngrid=360, 
                    dk=0.005, 
                    P0=1e4, 
                    silent=True)

            # save power spectrum to file 
            hdr = 'k, p0k, p2k, p4k' 
            np.savetxt(fplk, np.array([k, p0k, p2k, p4k]).T, 
                    fmt='%.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 


def test_mocks(i0, i1, overwrite=False): 
    ''' construct test mocks with assembly bias and velocity bias
    '''
    for i_fid in range(i0, i1+1): 
        print('Fiducial %i' % i_fid)
        halos = Halos.Quijote_fiducial_HR_Rockstar(i_fid, z=0.5)
        
        for i_hod in range(5):
            fgal = os.path.join(dat_dir, 'tests',
                    'hod.quijote_fid%i.z0p5.cmass_sgc.v1.%i.bf' % (i_fid, i_hod))
            fhod = fgal.replace('.bf', '.npy')
            fplk = fgal.replace('hod.', 'plk.hod.').replace('.bf', '.dat') 
            
            if os.path.isdir(fgal) and os.path.isfile(fhod) and not overwrite: 
                continue

            # HOD parameters 
            theta = sample_HOD(typ='test')
            abias, conc, eta_c, eta_s = sample_theta_bias(typ='test')

            theta['mean_occupation_centrals_assembias_param1'] = abias
            theta['mean_occupation_satellites_assembias_param1'] = abias

            theta['conc_gal_bias.satellites'] = conc
            theta['eta_vb.centrals'] = eta_c
            theta['eta_vb.satellites'] = eta_s
            
            np.save(fhod, theta)

            if i_hod == 0: 
                _Z07AB = Galaxies.VelAssembiasZheng07Model()
                Z07AB = _Z07AB.to_halotools(halos.cosmo, 
                        halos.attrs['redshift'], 
                        halos.attrs['mdef'], 
                        sec_haloprop_key='halo_nfw_conc')

                # populate with HOD
                hod = halos.populate(Z07AB, **theta)
            else: 
                # populate with HOD
                hod.repopulate(**theta)

            gals = FM.BOSS(
                    hod, 
                    sample='cmass-south', 
                    veto=True, 
                    fiber_collision=True, 
                    silent=True)
            gals.save(fgal)

            # construct corresponding randoms
            rand = FM.BOSS_randoms(gals, veto=True, sample='cmass-south') 

            # measure power spectrum 
            k, p0k, p2k, p4k = CosmoObs.Plk_survey(gals, rand,
                    Ngrid=360, 
                    dk=0.005, 
                    P0=1e4, 
                    silent=True)

            # save power spectrum to file 
            hdr = 'k, p0k, p2k, p4k' 
            np.savetxt(fplk, np.array([k, p0k, p2k, p4k]).T, 
                    fmt='%.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 


def test0_mocks(i0, i1, overwrite=False): 
    ''' construct test mocks without assembly bias and velocity bias
    '''
    for i_fid in range(i0, i1+1): 
        print('Fiducial %i' % i_fid)
        halos = Halos.Quijote_fiducial_HR_Rockstar(i_fid, z=0.5)
        
        for i_hod in range(5):
            fgal = os.path.join(dat_dir, 'tests0',
                    'hod.quijote_fid%i.z0p5.cmass_sgc.v1.%i.bf' % (i_fid, i_hod))
            fhod = fgal.replace('.bf', '.npy')
            fplk = fgal.replace('hod.', 'plk.hod.').replace('.bf', '.dat') 
            
            if os.path.isdir(fgal) and os.path.isfile(fhod) and not overwrite: 
                continue

            # HOD parameters            
            theta = sample_HOD(typ='test')
            np.save(fhod, theta)
                
            # populate with HOD
            hod = Galaxies.hodGalaxies(halos, theta_hod)

            gals = FM.BOSS(
                    hod, 
                    sample='cmass-south', 
                    veto=True, 
                    fiber_collision=True, 
                    silent=True)
            gals.save(fgal)

            # construct corresponding randoms
            rand = FM.BOSS_randoms(gals, veto=True, sample='cmass-south') 

            # measure power spectrum 
            k, p0k, p2k, p4k = CosmoObs.Plk_survey(gals, rand,
                    Ngrid=360, 
                    dk=0.005, 
                    P0=1e4, 
                    silent=True)

            # save power spectrum to file 
            hdr = 'k, p0k, p2k, p4k' 
            np.savetxt(fplk, np.array([k, p0k, p2k, p4k]).T, 
                    fmt='%.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 


def test2_mocks(i0, i1, overwrite=False): 
    ''' construct test mocks with assembly bias and velocity bias
    '''
    for i_fid in range(i0, i1+1): 
        print('Fiducial %i' % i_fid)
        halos = Halos.Quijote_fiducial_HR_Rockstar(i_fid, z=0.5)
        
        for i_hod in range(5):
            fgal = os.path.join(dat_dir, 'tests2',
                    'hod.quijote_fid%i.z0p5.cmass_sgc.v1.%i.bf' % (i_fid, i_hod))
            fhod = fgal.replace('.bf', '.npy')
            fplk = fgal.replace('hod.', 'plk.hod.').replace('.bf', '.dat') 
            
            if os.path.isdir(fgal) and os.path.isfile(fhod) and not overwrite: 
                continue

            # HOD parameters            
            theta = sample_HOD(typ='train')
            abias, conc, eta_c, eta_s = sample_theta_bias()

            theta['mean_occupation_centrals_assembias_param1'] = abias
            theta['mean_occupation_satellites_assembias_param1'] = abias

            theta['conc_gal_bias.satellites'] = conc
            theta['eta_vb.centrals'] = eta_c
            theta['eta_vb.satellites'] = eta_s

            np.save(fhod, theta)

            # populate with HOD
            if i_hod == 0: 
                _Z07AB = Galaxies.VelAssembiasZheng07Model()
                Z07AB = _Z07AB.to_halotools(halos.cosmo, 
                        halos.attrs['redshift'], 
                        halos.attrs['mdef'], 
                        sec_haloprop_key='halo_nfw_conc')
                hod = halos.populate(Z07AB, **theta)
            else: 
                hod.repopulate(**theta)

            gals = FM.BOSS(
                    hod, 
                    sample='cmass-south', 
                    veto=True, 
                    fiber_collision=True, 
                    silent=True)
            gals.save(fgal)

            # construct corresponding randoms
            rand = FM.BOSS_randoms(gals, veto=True, sample='cmass-south') 

            # measure power spectrum 
            k, p0k, p2k, p4k = CosmoObs.Plk_survey(gals, rand,
                    Ngrid=360, 
                    dk=0.005, 
                    P0=1e4, 
                    silent=True)

            # save power spectrum to file 
            hdr = 'k, p0k, p2k, p4k' 
            np.savetxt(fplk, np.array([k, p0k, p2k, p4k]).T, 
                    fmt='%.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 


def train_halo(i0, i1): 
    ''' construct training data by randomly sampling HOD parameters and 
    constructing a mock galaxy survey 
    '''
    for i_lhc in range(i0, i1+1): 
        print('LHC %i' % i_lhc)
    
        # read in halo catalog
        halos = Halos.Quijote_LHC_HR_Rockstar(i_lhc, z=0.5)

        k, p0k, p2k, p4k = CosmoObs.Plk_box(halos)

        fplk = os.path.join(dat_dir, 'train', 'plk', 'plk.halo.quijote_LH%i.z0p5.dat' % i_lhc)

        # measure power spectrum 
        k, p0k, p2k, p4k = CosmoObs.Plk_box(halos, Lbox=1000., Ngrid=360, dk=0.005)

        # save power spectrum to file 
        hdr = 'k, p0k, p2k, p4k' 
        np.savetxt(fplk, np.array([k, p0k, p2k, p4k]).T, 
                fmt='%.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 


def test_halo(i0, i1): 
    ''' construct training data by randomly sampling HOD parameters and 
    constructing a mock galaxy survey 
    '''
    for i_fid in range(i0, i1+1): 
        print('Fid %i' % i_fid)
    
        # read in halo catalog
        halos = Halos.Quijote_fiducial_HR_Rockstar(i_fid, z=0.5)

        k, p0k, p2k, p4k = CosmoObs.Plk_box(halos)

        fplk = os.path.join(dat_dir, 'tests', 'plk', 'plk.halo.quijote_fid%i.z0p5.dat' % i_fid)

        # measure power spectrum 
        k, p0k, p2k, p4k = CosmoObs.Plk_box(halos, Lbox=1000., Ngrid=360, dk=0.005)

        # save power spectrum to file 
        hdr = 'k, p0k, p2k, p4k' 
        np.savetxt(fplk, np.array([k, p0k, p2k, p4k]).T, 
                fmt='%.5e %.5e %.5e %.5e', delimiter='\t', header=hdr)
    return None 


def fix_bug(): 
    ''' script for fixing any bugs. This script is here for posterity

    * 2022/05/05: something went wrong with the hod realizations 5-10. They will be deleted.  
    '''
    # 2022/05/05 fix
    for i_lhc in range(2000): 
        print('LHC %i' % i_lhc)
        for i_hod in range(5, 10):
            fgal = os.path.join(dat_dir, 'train',
                    'hod.quijote_LH%i.z0p5.cmass_sgc.v1.%i.bf' % (i_lhc, i_hod))
            fplk = os.path.join(dat_dir, 'train', 'plk', 
                    'plk.hod.quijote_LH%i.z0p5.cmass_sgc.v1.%i.dat' % (i_lhc, i_hod))

            os.system('rm -rf %s' % fgal) 
            os.system('rm %s' % fplk)

    return None 


typ = sys.argv[1]
if typ == 'fix_bug': fix_bug()
i0  = int(sys.argv[2]) 
i1  = int(sys.argv[3])

if typ == 'train': 
    overwrite = (sys.argv[4] == 'True') 
    train_mocks(i0, i1, overwrite=overwrite) 
elif typ == 'test': 
    overwrite = (sys.argv[4] == 'True') 
    test_mocks(i0, i1, overwrite=overwrite) 
elif typ == 'test2': 
    overwrite = (sys.argv[4] == 'True') 
    test2_mocks(i0, i1, overwrite=overwrite) 
elif typ == 'train_halo': 
    train_halo(i0, i1)
elif typ == 'test_halo': 
    test_halo(i0, i1)
