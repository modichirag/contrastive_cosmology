'''

module for galaxy catalogs. Module includes functions to construct galaxy
catalogs using halo occupation distribution (HOD) models. HOD models paint
galaxies onto halo catalogs constructed from N-body simulations 



References
----------
* make_survey pipeline from BOSS: https://trac.sdss3.org/browser/repo/mockFactory/trunk/make_survey?order=name


'''
import os 
import numpy as np 
# --- nbodykit --- 
import nbodykit.lab as NBlab
from nbodykit.hod import Zheng07Model, HODModel


def thetahod_literature(paper): 
    ''' best-fit HOD parameters from the literature. 
    
    Currently, HOD values from the following papers are available:
    * 'parejko2013_lowz'
    * 'manera2015_lowz_ngc'
    * 'manera2015_lowz_sgc'
    * 'redi2014_cmass'
    '''
    if paper == 'parejko2013_lowz': 
        # lowz catalog from Parejko+2013 Table 3. Note that the 
        # parameterization is slightly different so the numbers need to 
        # be converted.
        p_hod = {
                'logMmin': 13.25, 
                'sigma_logM': 0.43, #0.7 * sqrt(2) * log10(e)
                'logM0': 13.27, # log10(kappa * Mmin)
                'logM1': 14.18, 
                'alpha': 0.94
                } 
    elif paper == 'manera2015_lowz_ngc':
        # best-fit HOD of the lowz catalog NGC from Table 2 of Manera et al.(2015) 
        p_hod = {
                'logMmin': 13.20, 
                'sigma_logM': 0.62, 
                'logM0': 13.24, 
                'logM1': 14.32, 
                'alpha': 0.9
                }
    elif paper == 'manera2015_lowz_sgc':
        # best-fit HOD of the lowz catalog SGC from Table 2 of Manera et al.(2015) 
        # Manera+(2015) actually uses a redshift dependent HOD. The HOD that's
        # currently implemented is primarily for the 0.2 < z < 0.35 population, 
        # which has nbar~3x10^-4 h^3/Mpc^3
        p_hod = {
                'logMmin': 13.14, 
                'sigma_logM':0.55,
                'logM0': 13.43, 
                'logM1': 14.58, 
                'alpha': 0.93 
                }
    elif paper == 'reid2014_cmass': 
        # best-fit HOD from Reid et al. (2014) Table 4
        p_hod = {
                'logMmin': 13.03, 
                'sigma_logM':0.38,
                'logM0': 13.27, 
                'logM1': 14.08, 
                'alpha': 0.76
                }
    else:
        raise NotImplementedError
    
    return p_hod 


def hodGalaxies(halos, p_hod, seed=None, hod_model='zheng07'): 
    ''' populate given halo catalog (halos) with galaxies based on HOD model
1    with p_hod parameters. Currently only supports the Zheng+(2007) model.

    Parameters
    ----------
    p_hod : dict
        dictionary specifying the HOD parameters 
    '''
    # check HOD parameters
    if 'alpha' not in p_hod.keys(): 
        raise ValueError
    if 'logMmin' not in p_hod.keys(): 
        raise ValueError
    if 'logM1' not in p_hod.keys(): 
        raise ValueError
    if 'logM0' not in p_hod.keys(): 
        raise ValueError
    if 'sigma_logM' not in p_hod.keys(): 
        raise ValueError

    # populate using HOD
    if hod_model == 'zheng07': 
        hod = halos.populate(Zheng07Model, seed=seed, **p_hod) 
    elif hod_model == 'zheng07_ab': 
        hod = halos.populate(AssembiasZheng07Model, seed=seed, **p_hod) 
    else: 
        raise NotImplementedError
    return hod 


def BOSSGalaxies(sample='lowz-south'): 
    ''' Read in BOSS galaxy catalog. Data can be downloaded from 
    https://data.sdss.org/sas/dr12/boss/lss/
    

    Parameters
    ----------
    sample : string
        Specify the exact BOSS sample. For options see
        https://data.sdss.org/sas/dr12/boss/lss/
        (Default: 'cmass-north')

    Returns
    -------
    data : nbodykit.lab.FITSCatalog object
        BOSS galaxy catalog  
    '''
    if sample == 'lowz-south': 
        fgal = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'dat',
                'galaxy_DR12v5_LOWZ_South.fits.gz')
    elif sample == 'cmass-south': 
        fgal = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'dat',
                'galaxy_DR12v5_CMASS_South.fits.gz')
    elif sample == 'cmass-north': 
        fgal = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'dat',
                'galaxy_DR12v5_CMASS_North.fits.gz')
    else: 
        raise NotImplementedError

    data = NBlab.FITSCatalog(fgal)
    return data


class AssembiasZheng07Model(HODModel):

    @staticmethod
    def to_halotools(cosmo, redshift, mdef, concentration_key=None, **kwargs):
        """
            Return the Zheng 07 HOD model decorated with assembly bias.

        Parameters
        ----------
        cosmo :
            the nbodykit or astropy Cosmology object to use in the model
        redshift : float
            the desired redshift of the model
        mdef : str, optional
            string specifying mass definition, used for computing default
            halo radii and concentration; should be 'vir' or 'XXXc' or
            'XXXm' where 'XXX' is an int specifying the overdensity
        concentration_key : str
            the name of the column that will specify concentration; if not
            provided, the analytic formula from
            `Dutton and Maccio 2014 <https://arxiv.org/abs/1402.7073>`_
            is used.
        **kwargs :
            additional keywords passed to the model components; see the
            Halotools documentation for further details
        Returns
        -------
        :class:`~halotools.empirical_models.HodModelFactory`
            the halotools object implementing the HOD model
        """
        from nbodykit.cosmology import Cosmology
        from halotools.empirical_models import AssembiasZheng07Sats, AssembiasZheng07Cens, NFWPhaseSpace, TrivialPhaseSpace
        from halotools.empirical_models import HodModelFactory 

        kwargs.setdefault('modulate_with_cenocc', True)

        # need astropy Cosmology
        if isinstance(cosmo, Cosmology):
            cosmo = cosmo.to_astropy()

        # determine concentration key
        if concentration_key is None:
            conc_mass_model = 'dutton_maccio14'
        else:
            conc_mass_model = 'direct_from_halo_catalog'

        # determine mass column
        mass_key = 'halo_m' + mdef

        # occupation functions
        cenocc = AssembiasZheng07Cens(prim_haloprop_key=mass_key, **kwargs)
        satocc = AssembiasZheng07Sats(prim_haloprop_key=mass_key, cenocc_model=cenocc, **kwargs)
        satocc._suppress_repeated_param_warning = True

        # profile functions
        kwargs.update({'cosmology':cosmo, 'redshift':redshift, 'mdef':mdef})
        censprof = TrivialPhaseSpace(**kwargs)
        satsprof = NFWPhaseSpace(conc_mass_model=conc_mass_model, **kwargs)

        # make the model
        model = {}
        model['centrals_occupation'] = cenocc
        model['centrals_profile'] = censprof
        model['satellites_occupation'] = satocc
        model['satellites_profile'] = satsprof
        return HodModelFactory(**model)
