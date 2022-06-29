import numpy as np
import json
import tools
from nbodykit.lab import FFTPower

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


def sample_conditional_HOD(m_hod, mcut, m1=None, seed=0): 
    ''' sample HOD value based on priors set by Parejko+(2013)
    centrals: 0.5*[1+erf((\log M_h - \log M_cut)/\sigma)]
    satellites: ((M_h - M_0)/M_1)**\alpha
    '''
    np.random.seed(seed)
    m0 = mcut
    if m1 is None: m1 = mcut + 0.5
    if m_hod == 'zheng07':
        hod_min = np.array([mcut, 0.4, m0, m1, 0.7])
        dhod = np.array([0.15, 0.1, 0.2, 0.3, 0.3])
        #dhod = np.array([0.029, 0.06, 0.13, 0.06, 0.18])
        _hod = hod_min + dhod * np.random.uniform(-1, 1, size=(5))
        return {'logMmin': _hod[0], 'sigma_logM': _hod[1], 'logM0': _hod[2], 'logM1': _hod[3], 'alpha': _hod[4]}
    elif m_hod == 'zheng07_ab': 
        hod_min = np.array([mcut, 0.4, m0, m1, 0.7, 0., 0.])
        dhod = np.array([0.15, 0.1, 0.2, 0.3, 0.3, 0.5, 0.5])
        #dhod = np.array([0.2, 0.1, 0.5, 0.4, 0.4, 0.5, 0.5])
        _hod = hod_min + dhod * np.random.uniform(size=(7))
        return {'logMmin': _hod[0], 'sigma_logM': _hod[1], 'logM0': _hod[2], 'logM1': _hod[3], 'alpha': _hod[4], 
                'mean_occupation_centrals_assembias_param1': _hod[5], 
                'mean_occupation_satellites_assembias_param1': _hod[6]}
    else: 
        raise NotImplementedError 



def galaxy_summary(hod, bs, filename=None):
    gtype = hod['gal_type'].compute()
    galsum = {}
    galsum['total'], galsum['number density'] = gtype.size, gtype.size/bs**3
    galsum['centrals'], galsum['satellites'] = np.unique(gtype, return_counts=True)[1]
    galsum['fsat'] = galsum['satellites']/galsum['total']
    if filename is not None:
        def _convert(o):
            if isinstance(o, np.generic): return int(o)  
            raise TypeError
        with open(filename, 'w') as fp:
            json.dump(galsum, fp, default=_convert, indent=4, sort_keys=True)
    return galsum


def get_power(f, pm, num=None, compensated=False):
    if num is None: gal = pm.paint(f['Position'].compute())
    else: gal = pm.paint(f['Position'][:num].compute())
    if compensated: gal = tools.cic_compensation(gal, order=2)

    mesh = gal / gal.cmean() - 1
    ps = FFTPower(mesh, mode='1d').power.data
    k, p = ps['k'], ps['power'].real
    return k, p


def get_power_rsd(f, pm, num=None, compensated=False, los=[0, 0, 1]):
    pos = f['Position'] + f['VelocityOffset']*los
    if num is None: gal = pm.paint(pos.compute())
    else: gal = pm.paint(pos[:num].compute())
    if compensated: gal = tools.cic_compensation(gal, order=2)

    mesh = gal / gal.cmean() - 1
    ps = FFTPower(mesh, mode='2d', Nmu=10).power.data
    k, p = ps['k'], ps['power'].real[:, 5:]
    return k, p
