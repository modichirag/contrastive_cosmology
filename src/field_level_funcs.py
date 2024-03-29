from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from nbodykit.lab import *
import time

# Setup MPI environment
from mpi4py import MPI
comm = MPI.COMM_WORLD
wrank = comm.Get_rank()
wsize = comm.Get_size()


class field_level_biases():
    def __init__(self, linmesh, verb=True, prepare=True, kfilt_d3=0.5):
        """Class to compute biases at the field-level. Main inputs are the linear mesh (linmesh) at the redshift of interest, taken as an Nbodykit mesh.
        
        On initialization, we compute the galaxy-independent quantities from the initial conditions and Zel'dovich displacements."""
        
        # Load attributes
        self.pm = linmesh.pm
        assert self.pm.BoxSize[0]==self.pm.BoxSize[1]==self.pm.BoxSize[2]
        self.kF = np.pi/self.pm.BoxSize[0]
        
        # Compute 3D k vectors
        self.kk = linmesh.r2c().x
        self.k2 = self.kk[0]**2+self.kk[1]**2+self.kk[2]**2                                                                          
        self.k2[0,0,0] =  1                                                                  
        self.modk = np.sqrt(self.k2)

        # Prepare the transfer functions
        if prepare: self._prepare(linmesh, verb, kfilt_d3=kfilt_d3) 
        
    def fit(self, hmesh, nbar, kmax = 0.4, kNL = 0.45, verb=False, fit_error=False, return_model=False, plot_transfer=False):
        """Fit the bias parameters given a galaxy density field (hmesh) with number-density nbar using all k-modes up to kmax.
        
        We optionally compute the shot-noise parameters, and optionally return the PT model itself."""
        
        init = time.time()
        assert kmax < self.kF*self.pm.Nmesh[0]
        
        if verb: print("Computing transfer functions")
        self.betas = {f: self.power(hmesh, self.orth_fields[f])/self.power(self.orth_fields[f]) for f in self.orth_fields.keys()}
        #self.Nmodes = np.asarray([np.sum((self.modk>=self.kF*(kbin+1)+1e-6)*((self.modk<self.kF*(kbin+2)+1e-6))) for kbin in range(len(self.k))])

        ## First fit for beta_2, beta_G2, beta_3
        def to_min(pars, beta=None):
            return np.sum((beta-beta_model(pars))**2*self.Nmodes*(self.k<kmax))
        def beta_model(pars, k=self.k):
            return pars[0]+pars[1]*k**2+pars[2]*k**4

        par_d2 = minimize(to_min, [1,0,0], args=self.betas['d2']).x
        par_G2 = minimize(to_min, [1,0,0], args=self.betas['G2']).x
        par_d3 = minimize(to_min, [1,0,0], args=self.betas['d3']).x

        ## Extract biases
        biases = {'b2': par_d2[0]*2, 'b3': par_d3[0]*6}
        bG2_plus_27_b1 = par_G2[0]

        ## Now fit for beta1
        beta1_model = lambda b1, bGamma3, bgrad, bgradgrad: b1 + bgrad*self.k**2 + bgradgrad*self.k**4 + (bGamma3+b1/6.)*self.transfer['Gamma3'] + biases['b2']/2.*self.transfer['d2'] + bG2_plus_27_b1*self.transfer['G2'] - b1*self.transfer['S3']
        def to_min_d1(pars):
            return np.sum((self.betas['d1']-beta1_model(*pars))**2*self.Nmodes*(self.k<kmax))

        par_d1 = minimize(to_min_d1, [1,0,0,0]).x
        biases['b1'], biases['bGamma3'], biases['bgrad'], biases['bgradgrad'] = par_d1
        biases['bG2'] = bG2_plus_27_b1-2./7.*biases['b1']

        if fit_error:
            
            ## Compute error function
            if verb: print("Generating EFT model")
            eft_model_k = self.pm.create('complex', value=0.)
            orth_fields_k = {f: self.orth_fields[f].r2c() for f in self.orth_fields.keys()}
            for kbin in range(len(self.k)):
                kfilt = (self.modk>=self.kF*(kbin+1)+1e-6)*((self.modk<self.kF*(kbin+2)+1e-6))
                eft_model_k[kfilt] += beta1_model(biases['b1'], biases['bGamma3'], biases['bgrad'], biases['bgradgrad'])[kbin]*orth_fields_k['d1'][kfilt]
                eft_model_k[kfilt] += beta_model(par_d2)[kbin]*orth_fields_k['d2'][kfilt]
                eft_model_k[kfilt] += beta_model(par_G2)[kbin]*orth_fields_k['G2'][kfilt]
                eft_model_k[kfilt] += beta_model(par_d3)[kbin]*orth_fields_k['d3'][kfilt]
            eft_model = eft_model_k.c2r()

            ## Fit error function
            if verb: print("Fitting error function")
            self.Perr = self.power(eft_model-hmesh)
            def Perr_model(pars):
                return 1./nbar*(1+pars[0]+pars[1]*(self.k/kNL)**2)
            def to_min_err(pars):
                return np.sum((self.Perr-Perr_model(pars))**2*self.Nmodes*(self.k<kmax))

            biases['alpha0'], biases['alpha1'] = minimize(to_min_err, [1,0]).x

        if verb: print("Fitting complete in %.2f seconds"%(time.time()-init))
            
        ### Create some cosmetic plots to show the fit
        if plot_transfer:
            c = ['b','r','g','purple']
            plt.errorbar(self.k, self.betas['d1'], label=r'$\delta_1^\perp$',c=c[0],marker='.',alpha=0.3,ms=2)
            plt.errorbar(self.k, self.betas['d2'], label=r'$(\delta_1^2)^\perp$',c=c[1],marker='.',alpha=0.3,ms=2)
            plt.errorbar(self.k, self.betas['G2'], label=r'$\mathcal{G}_2^\perp$',c=c[2],marker='.',alpha=0.3,ms=2)
            plt.errorbar(self.k, self.betas['d3'], label=r'$(\delta_1^3)^\perp$',c=c[3],marker='.',alpha=0.3,ms=2)
            plt.plot(self.k, beta1_model(biases['b1'], biases['bGamma3'], biases['bgrad'], biases['bgradgrad']),c=c[0],ls='--')
            plt.plot(self.k, par_d2[0]+par_d2[1]*self.k**2+par_d2[2]*self.k**4,c=c[1],ls='--')
            plt.plot(self.k, par_G2[0]+par_G2[1]*self.k**2+par_G2[2]*self.k**4,c=c[2],ls='--')
            plt.plot(self.k, par_d3[0]+par_d3[1]*self.k**2+par_d3[2]*self.k**4,c=c[3],ls='--')
            plt.vlines(kmax,-10,10,linestyles='--',alpha=0.5,color='k')
            plt.xlabel(r'$k$',fontsize=15);
            plt.ylabel(r'$\beta(k)$',fontsize=15);
            plt.legend(fontsize=12);
            plt.ylim(-10, 10)
            plt.savefig('transfers.png')
            if fit_error:
                plt.figure()
                plt.errorbar(self.k, self.Perr, marker='.',alpha=0.3,ms=2, label=r'$|\delta_{\rm PT}-\delta_g|^2$')
                plt.plot(self.k, Perr_model([biases['alpha0'], biases['alpha1']]),c=c[0],ls='--')
                plt.plot(self.k, 1./nbar+0.*self.k,c='k',ls=':', label=r'$1/\bar{n}$')
                plt.vlines(kmax,1./nbar*0.3,1./nbar*2,linestyles='--',alpha=0.5,color='k')
                plt.xlabel(r'$k$',fontsize=12);
                plt.ylabel(r'$P_{\rm err}(k)$',fontsize=12);
                plt.loglog()
                plt.savefig('error.png')

        if return_model: 
            return biases, eft_model
        else:
            return biases

    def plotter(self, dens, reduce_x = 1, reduce_y=1, reduce_z = 8, ax=2, vmax=None, vmin=None):
        """Utility function to plot a density field"""
        plt.figure()
        plt.imshow(dens[:self.pm.Nmesh[0]//reduce_x,:self.pm.Nmesh[1]//reduce_y,:self.pm.Nmesh[2]//reduce_z].mean(ax),vmax=vmax, vmin=vmin);
        plt.colorbar();
        
    def power(self, field1, field2=None):
        """Compute the power spectrum between two fields, binning in the fundamental mode"""
        
        if type(field2)==type(None):
            field2 = field1.copy()
        
        power = FFTPower(field1, second = field2, mode='1d', kmin=self.kF+1e-6,dk=self.kF)
        
        if not hasattr(self, 'k'):
            self.k = power.power['k']
            self.Nmodes = power.power['modes']
        
        return power.power['power'].real 

    def _prepare(self, linmesh, verb=False, kfilt_d3=0.5):
        """Prepare the field-level bias code. This computes all quantities needed to fit the bias parameters"""

        init = time.time()
        
        # Compute Zel'dovich displacement field
        if verb: print("# Computing Zel'dovich displacements")
        init_pos = self.pm.generate_uniform_particle_grid(shift=0)
        zel_pos = self._compute_zel_displacement(linmesh.r2c(), init_pos, dgrow=1)
        
        # Compute unshifted fields
        if verb: print("# Computing unshifted fields")
        fields = self._compute_fields(linmesh, kfilt_d3=kfilt_d3)

        # Compute shifted fields
        if verb: print("# Computing shifted fields")
        shift_fields = self._apply_shift(fields, init_pos, zel_pos, compensated=True)
        del fields, init_pos, zel_pos
        
        # Compute transfer ratios
        if verb: print("# Computing data-independent transfer ratios")
        power_d1 = self.power(shift_fields['d1'])
        self.transfer = {f: self.power(shift_fields['d1'], shift_fields[f])/power_d1 for f in ['d2','G2','Gamma3','S3']}
        
        # Compute orthogonalized fields
        if verb: print("# Computing orthogonalized fields")
        self.orth_fields = self._orthogonalize_fields(shift_fields, verb=verb)
        del shift_fields
        
        if verb: print("## Preprocessing complete in %.2f seconds"%(time.time()-init))
        
    def _compute_zel_displacement(self, linear_field_k, init_pos, resampler='cic', dgrow=1):
        """ Run first order LPT on linear density field, returns displacements of particles reading out at q. The result has the same dtype as q."""
        
        # Dimension
        ndim = len(self.pm.Nmesh)
        
        source = np.zeros((len(init_pos), ndim), dtype=init_pos.dtype)
        for d in range(len(self.pm.Nmesh)):
            disp = linear_field_k.apply(self._laplace).apply(self._gradient(d), out=Ellipsis).c2r(out=Ellipsis)
            source[..., d] = disp.readout(init_pos, resampler=resampler)*dgrow

        pos = init_pos + source
        pos[pos < 0] += self.pm.BoxSize[0]
        pos[pos > self.pm.BoxSize[0]] -= self.pm.BoxSize[0]
        return pos

    def _compute_fields(self, linmesh, kfilt_d3=0.5):
        """Compute all unshifted fields of interest from a linear mesh. We filter the cubic field at k < 0.5 before cubing, following 1811.10640."""
        fields = {}
        # Linear
        fields['d1'] = linmesh.copy()
        # Quadratic: delta^2 - < delta^2 >
        fields['d2'] = linmesh**2
        fields['d2'] -= fields['d2'].cmean()
        # Cubic (from filtered field)
        filt_lin = ((linmesh.r2c())*(self.modk<kfilt_d3)).c2r()
        fields['d3'] = filt_lin**3
        del filt_lin
        # Tidal G2
        fields['G2'] = self._G2_func(linmesh)
        # Tidal Gamma3
        fields['Gamma3'] = self._Gamma3_func(linmesh)
        # S3 shift (removing low-k limit via G2-delta ~ delta^3-like term)
        psi2_q = np.asarray([-3./14.*fields['G2'].copy().r2c().apply(self._laplace).apply(self._gradient(d, order=0), out=Ellipsis).c2r(out=Ellipsis) for d in range(3)])
        grad_d1 = np.asarray([linmesh.copy().r2c().apply(self._gradient(d, order=0), out=Ellipsis).c2r(out=Ellipsis) for d in range(3)])
        fields['S3'] = self.pm.create(type='real', value=np.sum(psi2_q*grad_d1,axis=0)-2./21.*linmesh**3)
        return fields

    def _apply_shift(self, fields, init_pos, zel_pos, compensated=True):
        """Shift all fields by the Zel'dovich displacement vector"""

        glay = self.pm.decompose(init_pos)

        catalog = ArrayCatalog({'Position':zel_pos}, BoxSize = self.pm.BoxSize, Nmesh = self.pm.Nmesh)

        shift_fields = {}
        for field in fields.keys():
            catalog['mass'] = fields[field].readout(init_pos, layout = glay, resampler='nearest')
            shift_fields[field] = catalog.to_mesh(value='mass', compensated=compensated).paint()

        return shift_fields

    def _orthogonalize_fields(self, shift_fields, verb=False):
    
        # Define covariance of fields
        if verb: print("Generating covariance")
        ops = [shift_fields['d1'], shift_fields['d2'], shift_fields['G2'], shift_fields['d3']]
        cov = np.zeros((4,4,len(self.k)))
        for i in range(4):
            for j in range(4):
                cov[i,j] = self.power(ops[i], ops[j])

        # Create rotation matrix
        if verb: print("Computing rotation matrix")
        M_mats = []
        for kk in range(len(self.k)):
            cov_ij = cov[:,:,kk]
            corr_ij = cov_ij/np.sqrt(np.outer(np.diag(cov_ij), np.diag(cov_ij)))
            chol_ij = np.linalg.cholesky(corr_ij)
            ichol_ij = np.linalg.inv(chol_ij)
            M_ij = (ichol_ij*np.diag(cov_ij)[None,:]**(-0.5))/(np.diag(ichol_ij)[:,None]*np.diag(cov_ij)[:,None]**(-0.5))
            M_mats.append(M_ij)

        # Orthogonalize operators
        if verb: print("Creating orthogonalized basis")
        ops_k = np.asarray([op.r2c() for op in ops])
        orth_fields_k = np.asarray([self.pm.create(type='complex',value=0.) for _ in range(4)])
        for kbin in range(len(self.k)):
            kfilt = (self.modk>=self.kF*(kbin+1)+1e-6)*((self.modk<self.kF*(kbin+2)+1e-6))
            ops_k_filt = ops_k[:,kfilt]
            orth_fields_k[:,kfilt] += M_mats[kbin]@ops_k_filt

        # Create output dictionary and return
        orth_fields = {'d1':self.pm.create(type='complex',value=orth_fields_k[0]).c2r(),
                      'd2':self.pm.create(type='complex',value=orth_fields_k[1]).c2r(),
                      'G2':self.pm.create(type='complex',value=orth_fields_k[2]).c2r(),
                      'd3':self.pm.create(type='complex',value=orth_fields_k[3]).c2r(),
                      }
        
        return orth_fields

    def _laplace(self, k, v):
        kk = k[0]**2+k[1]**2+k[2]**2
        mask = (kk == 0).nonzero()
        kk[mask] = 1
        b = v / kk
        b[mask] = 0
        return b

    def _gradient(self, dir, order=1):
        if order == 0:
            def kernel(k, v):
                # clear the nyquist to ensure field is real                         
                mask = v.i[dir] != v.Nmesh[dir] // 2
                return v * (1j * k[dir]) * mask
        if order == 1:
            def kernel(k, v):
                cellsize = (v.BoxSize[dir] / v.Nmesh[dir])
                w = k[dir] * cellsize

                a = 1 / (6.0 * cellsize) * (8 * np.sin(w) - np.sin(2 * w))
                # a is already zero at the nyquist to ensure field is real          
                return v * (1j * a)
        return kernel

    def _G2_func(self, base):                                                                                                                                          
        '''Takes in a PMesh object in real space. Returns an array of the G2 operator'''          
        g2 = self.pm.create(type='real', value=0)
        basec = base.r2c()
        for i in range(3):
            for j in range(i, 3):                                                       
                #basec = base.r2c()
                basec_k = basec * (self.kk[i]*self.kk[j] / self.k2)              
                baser = basec_k.c2r()                                                                
                g2[...] += baser**2                                                      
                if i != j:                                                              
                    g2[...] += baser**2                                                                                                         
        return g2-base**2

    def _Gamma3_func(self, base):                                                                                                                                          
        '''Takes in a PMesh object in real space. Returns an array of the Gamma3 operator'''          

        Gam3 = self.pm.create(type='real', value=0)
        basec = base.r2c()
        Gam3_ij = [[(basec*(self.kk[i]*self.kk[j] / self.k2)).c2r().copy() for j in range(i+1)] for i in range(3)]
        G_ij = lambda i, j: Gam3_ij[np.max([i,j])][np.min([i,j])]

        # 3-shear term
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    Gam3[...] -= G_ij(i,j)*G_ij(j,k)*G_ij(k,i)

        # 2-shear term
        for i in range(3):
            for j in range(3):
                Gam3[...] += 3./2.*G_ij(i,j)**2*base

        # 0-shear term
        Gam3[...] -= base**3./2.

        return Gam3    
