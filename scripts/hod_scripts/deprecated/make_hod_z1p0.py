'''

script for constructing HOD catalogs for the Quijote LHC simulations
with a forward model of the BOSS CMASS galaxy sample
Use this file for z=0.5 catalog

'''
import os, sys 
import numpy as np 
sys.path.append('../../src/')
import halos as Halos
import galaxies as Galaxies
import json

np.random.seed(1)

m_hod = sys.argv[1] # hod model 
i0 = int(sys.argv[2]) 
i1 = int(sys.argv[3])
zred = 1.0
nhod = 5
#numd = 4e-4

def sample_HOD(m_hod): 
    ''' sample HOD value based on priors set by Jae-woo Kim et al (2015), Table 1, last row
    (https://iopscience.iop.org/article/10.1088/0004-637X/806/2/189/pdf))
    They set M1=Mcut (which is Mmin) (see below eq. 4) and there M1 and M0 are SWITCHED.
    So we set M0=Mcut and take their value of M0 for M1
    The scatter uncertainties are increased to match z0,low z numbers that Chang used
    '''
    if m_hod == 'zheng07':
        hod_min = np.array([13., 0.6, 13., 13.9, 0.9])
        dhod = np.array([0.15, 0.1, 0.4, 0.3, 0.4])
        _hod = hod_min + dhod * np.random.uniform(size=(5))
        return {'logMmin': _hod[0], 'sigma_logM': _hod[1], 'logM0': _hod[2], 'logM1': _hod[3], 'alpha': _hod[4]}
    elif m_hod == 'zheng07_ab': 
        hod_min = np.array([13., 0.6, 13., 13.9, 0.9, 0., 0.])
        dhod = np.array([0.15, 0.1, 0.4, 0.3, 0.4, 0.5, 0.5])
        _hod = hod_min + dhod * np.random.uniform(size=(7))
        return {'logMmin': _hod[0], 'sigma_logM': _hod[1], 'logM0': _hod[2], 'logM1': _hod[3], 'alpha': _hod[4], 
                'mean_occupation_centrals_assembias_param1': _hod[5], 
                'mean_occupation_satellites_assembias_param1': _hod[6]}
    else: 
        raise NotImplementedError 


dat_dir = '/mnt/ceph/users/cmodi/simbig/z1p0/catalogs/'
for i_lhc in range(i0, i1):
    print('LHC %i' % i_lhc)
    # read in halo catalog
    halos = Halos.Quijote_LHC_HR(i_lhc, z=zred)
    for i_hod in range(nhod): 
        print('  HOD %i' % i_hod)
        
        save_dir = dat_dir + '%04d/'%i_lhc
        os.makedirs(save_dir, exist_ok=True)
        
        fgal = os.path.join(save_dir, '%s.%i.bf' % 
                (m_hod, i_hod))
        fhod = fgal.replace('.bf', '.npy')
        fsum = fgal.replace('.bf', '.json')

        if not os.path.isfile(fhod) or not os.path.isdir(fgal): 
            # sample HOD parameters
            theta_hod = sample_HOD(m_hod)
            np.save(fhod, theta_hod)
            # populate box with HOD
            hod = Galaxies.hodGalaxies(halos, theta_hod, seed=0, hod_model=m_hod)
            # save box 
            hod.save(fgal) 

            #calculate some summary numbers on galaxy types
            gtype = hod['gal_type'].compute()
            galsum = {}
            galsum['total'], galsum['number density'] = gtype.size, gtype.size/1000**3
            galsum['centrals'], galsum['satellites'] = np.unique(gtype, return_counts=True)[1]
            print(galsum)
            #save
            def convert(o):
                if isinstance(o, np.generic): return int(o)  
                raise TypeError
            with open(fsum, 'w') as fp:
                json.dump(galsum, fp, default=convert, indent=4, sort_keys=True)
