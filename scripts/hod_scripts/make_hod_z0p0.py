'''

script for constructing HOD catalogs for the Quijote LHC simulations
with a forward model of the BOSS LOWZ SGC sample
Use this file for z=0 catalog

'''
import os, sys 
import numpy as np 
sys.path.append('../src/')
import halos as Halos
import galaxies as Galaxies
import json

np.random.seed(1)

m_hod = sys.argv[1] # hod model 
i0 = int(sys.argv[2]) 
i1 = int(sys.argv[3])
zred = 0.0
nhod = 5
nbar = 3e-4
bs = 1000
fshift = 0.2                      # hopeful fraction of satellites, used to shift M0 from total number density
numhalos_nbarf = int(nbar * bs**3 * (1-fshift))

def sample_HOD(m_hod): 
    ''' sample HOD value based on priors set by Parejko+(2013)
    '''
    if m_hod == 'zheng07':
        hod_min = np.array([13.2, 0.4, 13.1, 14., 0.7])
        dhod = np.array([0.15, 0.1, 0.4, 0.3, 0.4])
        _hod = hod_min + dhod * np.random.uniform(size=(5))
        return {'logMmin': _hod[0], 'sigma_logM': _hod[1], 'logM0': _hod[2], 'logM1': _hod[3], 'alpha': _hod[4]}
    elif m_hod == 'zheng07_ab': 
        hod_min = np.array([13.2, 0.4, 13.1, 14., 0.7, 0., 0.])
        dhod = np.array([0.2, 0.1, 0.5, 0.4, 0.4, 0.5, 0.5])
        _hod = hod_min + dhod * np.random.uniform(size=(7))
        return {'logMmin': _hod[0], 'sigma_logM': _hod[1], 'logM0': _hod[2], 'logM1': _hod[3], 'alpha': _hod[4], 
                'mean_occupation_centrals_assembias_param1': _hod[5], 
                'mean_occupation_satellites_assembias_param1': _hod[6]}
    else: 
        raise NotImplementedError 


def sample_conditional_HOD(m_hod, mcut, m1=None): 
    ''' sample HOD value based on priors set by Parejko+(2013)
    '''
    m0 = mcut
    if m1 is None: m1 = mcut + 0.5
    if m_hod == 'zheng07':
        hod_min = np.array([mcut, 0.4, m0, m1, 0.7])
        #dhod = np.array([0.15, 0.1, 0.4, 0.3, 0.4])
        dhod = np.array([0.029, 0.06, 0.13, 0.06, 0.18])
        dhod /= 100.
        _hod = hod_min + dhod * np.random.uniform(-1, 1, size=(5))
        return {'logMmin': _hod[0], 'sigma_logM': _hod[1], 'logM0': _hod[2], 'logM1': _hod[3], 'alpha': _hod[4]}
    elif m_hod == 'zheng07_ab': 
        hod_min = np.array([mcut, 0.4, m0, m1, 0.7, 0., 0.])
        dhod = np.array([0.2, 0.1, 0.5, 0.4, 0.4, 0.5, 0.5])
        _hod = hod_min + dhod * np.random.uniform(size=(7))
        return {'logMmin': _hod[0], 'sigma_logM': _hod[1], 'logM0': _hod[2], 'logM1': _hod[3], 'alpha': _hod[4], 
                'mean_occupation_centrals_assembias_param1': _hod[5], 
                'mean_occupation_satellites_assembias_param1': _hod[6]}
    else: 
        raise NotImplementedError 


dat_dir = '/mnt/ceph/users/cmodi/simbig/catalogs/'
nhods = []
for i_lhc in range(i0, i1):
    print('LHC %i' % i_lhc)
    # read in halo catalog
    halos = Halos.Quijote_LHC_HR(i_lhc, z=zred)
    hmass = halos['Mass'].compute()
    #print(hmass.size, )     
    numdhalos = hmass.size/bs**3
    #print(nhod/1e-4, nhod/numd)

    mcut = hmass[:numhalos_nbarf][-1]
    print(mcut/1e10, numhalos_nbarf/hmass.size)
    nhods.append(numhalos_nbarf/hmass.size)
    alpha = 0.7
    nsat = fshift * nbar * bs**3
    mdiff = (hmass - mcut + mcut*1e-3)[:numhalos_nbarf] * alpha
    #print(mdiff)
    #print("Negative mdiff : ", (mdiff < -0.01).sum())
    #print(mdiff.size, np.where(mdiff < 0)[0], mdiff[ np.where(mdiff < 0)[0]]/mcut)
    msum = mdiff.sum()/nsat
    m1 = msum**(1/alpha)
    m1offset = np.log10(m1) - np.log10(mcut)
    print(m1/1e10, mcut/1e10, m1offset)
    

    for i_hod in range(nhod): 
        print('  HOD %i' % i_hod)
        
        save_dir = dat_dir + '%04d/'%i_lhc
        os.makedirs(save_dir, exist_ok=True)
        
        fgal = os.path.join(save_dir, 'z%s.%s.%i.bf' % 
                (str(zred).replace('.', 'p'), m_hod, i_hod))
        fhod = fgal.replace('.bf', '.npy')
        fsum = fgal.replace('.bf', '.json')

        if not os.path.isfile(fhod) or not os.path.isdir(fgal): 
            # sample HOD parameters
            theta_hod = sample_conditional_HOD(m_hod, np.log10(mcut), m1=np.log10(m1))
            #np.save(fhod, theta_hod)
            # populate box with HOD
            hod = Galaxies.hodGalaxies(halos, theta_hod, seed=0, hod_model=m_hod)
            # save box 
            #hod.save(fgal) 

            #calculate some summary numbers on galaxy types
            gtype = hod['gal_type'].compute()
            galsum = {}
            galsum['total'], galsum['number density'] = gtype.size, gtype.size/bs**3
            galsum['centrals'], galsum['satellites'] = np.unique(gtype, return_counts=True)[1]
            galsum['fsat'] = galsum['satellites']/galsum['total']
            print(galsum)
            #save
            def convert(o):
                if isinstance(o, np.generic): return int(o)  
                raise TypeError
            #with open(fsum, 'w') as fp:
            #    json.dump(galsum, fp, default=convert, indent=4, sort_keys=True)

nhods = np.array(nhods)
print((nhods > 1).sum())
print((nhods > 1.1).sum())

