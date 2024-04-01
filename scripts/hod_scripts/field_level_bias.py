'''
Generate galaxy catalog conditioned on number density with setup HOD
'''
import os, sys 
import numpy as np 
import argparse, json
from pmesh.pm import ParticleMesh, RealField
from nbodykit.lab import FFTPower, ProjectedFFTPower, ArrayMesh, BigFileMesh, BigFileCatalog, ArrayCatalog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
#
sys.path.append('/mnt/home/cmodi/Research/Projects/contrasative_cosmology/src/')
import halos as Halos
import galaxies as Galaxies
import tools, hodtools
from field_level_funcs import field_level_biases

from nbodykit.cosmology import Planck15

# Setup MPI environment
from mpi4py import MPI
comm = MPI.COMM_WORLD
wrank = comm.Get_rank()
wsize = comm.Get_size()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, help='HOD model name')
parser.add_argument('--z', type=float, default=0.5, help='redshift')
parser.add_argument('--finder', type=str, default="Rockstar", help='which halo finder')
#default parameters
parser.add_argument('--nc', type=int, default=256, help='size of grid for simulation')
parser.add_argument('--id0', type=int, default=0, help='first quijote seed')
parser.add_argument('--id1', type=int, default=1, help='last quijote seed')
parser.add_argument('--nhod', type=int, default=10, help='number of HOD sims')
parser.add_argument('--fiducial', type=int, default=0, help='for fiducial simulations')
parser.add_argument('--simulation', type=str, default='quijote', help='for fiducial simulations')
#HOD parameters
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999')
parser.add_argument('--alpha', type=float, default=0.7, help='alpha, slope of satellites')
parser.add_argument('--satfrac', type=float, default=0.2, help='satellite fraction')
parser.add_argument('--nbar', type=float, default=0.0001, help='number density of galaxies')
parser.add_argument('--abscatter', type=float, default=0.2, help='scatter of zheng_ab')
parser.add_argument('--rewrite', type=int, default=0, help='rewrite files which already exist')
#directory name
parser.add_argument('--suffix', type=str, default="", help='suffix for parent folder with z and nnbar')
parser.add_argument('--suffix2', type=str, default="", help='suffix for subfolder with model name')
args = parser.parse_args()

m_hod = args.model
zred = args.z
nbar = args.nbar
bs = 1000
nc = args.nc
pm = ParticleMesh(Nmesh=[args.nc, args.nc, args.nc], BoxSize=bs, dtype='f8', comm=comm)
args.wsize = wsize

if (args.fiducial == 1) :
    data_dir = f'/mnt/ceph/users/cmodi/contrastive/data-fiducial/{args.simulation}/{args.finder}/z{int(zred*10):02d}-N{int(nbar/1e-4):04d}/'
else:
    data_dir = f'/mnt/ceph/users/cmodi/contrastive/data/{args.simulation}/{args.finder}/z{int(zred*10):02d}-N{int(nbar/1e-4):04d}/'

#model suffix
namescatter = ''
if args.abscatter != 0.2: namescatter = '-abs%d'%(args.abscatter*10)
if args.suffix2 != "": args.suffix2 =  "-" + args.suffix2
data_dir = data_dir + f'{args.model}{namescatter}{args.suffix2}/'
if nc == 256: data_dir = data_dir[:-1] + f'-n256/'
if wrank == 0:
    print(f"\nSave in data directory : {data_dir}\n")
os.makedirs(data_dir, exist_ok=True)


def get_growth_factor(i_lhc):
    
    CAMB_file = '/mnt/ceph/users/fvillaescusa/Quijote/Linear_Pk/latin_hypercube/%d/CAMB.params'%i_lhc
    with open(CAMB_file,'r') as cambfile:
        for line in cambfile:
            if line[:6]=='omegab':
                omegab = float(line.split(' = ')[1].strip('\n'))
            elif line[:6]=='omegac':
                omegac = float(line.split(' = ')[1].strip('\n'))
            elif line[:2]=='H0':
                H0 = float(line.split(' = ')[1].strip('\n'))
            elif line[:3]=='YHe':
                YHe = float(line.split(' = ')[1].strip('\n'))
                
    #Omega_m , Omega_c, Omega_b, h = (omegab+omegac)/(H0/100)**2., (omegac)/(H0/100)**2., omegab/(H0/100)**2, H0/100
    #if wrank == 0 : print("Simulation Parameters: ", (omegac, omegab, H0, YHe))
    params = np.load('/mnt/ceph/users/cmodi/Quijote/params_lh.npy')[i_lhc]
    Omega_m, Omega_b, h, n_s, sigma_8 = params
    Omega_c = Omega_m - Omega_b
    if wrank == 0 : print("Simulation Parameters: {Om = %.2f, Ob = %.2f, h = %.2f}"%(Omega_m, Omega_b, h))
    cosmo = Planck15.clone(Omega_cdm = Omega_c, Omega_b = Omega_b, h=h, YHe=YHe)
    Dz = cosmo.scale_independent_growth_factor(args.z)
    if wrank == 0 : print("growth function : ", Dz)
    return Dz

    
def get_linear_mesh(i_lhc):
    linmesh = BigFileMesh(path=f"/mnt/ceph/users/cmodi/fastpm-shivam/WHITE_NOISE_LH_QUIJOTE/N{nc}_bigfile/{i_lhc}/linear/", dataset="LinearDensityK", comm=comm)
    linmesh = linmesh.compute()
    linmesh -= linmesh.cmean()
    # Add initial condition scaling
    Dz = get_growth_factor(i_lhc)
    linmesh *= Dz

    return linmesh



##Loop!
for i_lhc in range(args.id0, args.id1):

    try:
        if wrank == 0 :print('#################\nLHC %i\n' % i_lhc)
        save_dir = data_dir + '%04d/'%i_lhc
        os.makedirs(save_dir, exist_ok=True)        
        with open(save_dir + 'args.json', 'w') as fp:
            json.dump(vars(args), fp, indent=4, sort_keys=True)

        #Do not rerun the simulations already run
        if args.rewrite: do_hod = True
        else: do_hod = False
        if not os.path.isfile(save_dir + 'power_rsd.npy') : do_hod = True
        if not do_hod: continue

        linmesh = get_linear_mesh(i_lhc)
        flb = field_level_biases(linmesh, kfilt_d3=0.5, verb= not bool(wrank)) # decide what smoothing to apply to delta^3 term, as in Marcel's pape

        # read in halo catalog
        if args.simulation == 'fastpm':
            if args.fiducial: 
                print("No fiducial runs for FastPM")
                raise NotImplementedError
            else: 
                halos = Halos.FastPM_LHC_HR(i_lhc, z=zred, finder=args.finder)
        else:
            if args.fiducial: 
                halos = Halos.Quijote_fiducial_HR(i_lhc, z=zred, finder=args.finder)
            else:
                halos = Halos.Quijote_LHC_HR(i_lhc, z=zred, finder=args.finder)
        halos.comm = comm

        halos = halos.sort('Mass', reverse=True)
        hmass = halos['Mass'].compute()
        if wrank == 0 : print("Halo read. Mass: ", hmass)
        hsize = hmass.shape[0]
        hsize = hmass.size
        
        #Setup HOD
        #mcut, m1 = hodtools.setup_hod(halos, nbar=nbar, satfrac=args.satfrac, bs=bs, alpha_fid=args.alpha)

        #
        ps, ngals, gals, pmus, pells, hods = [], [], [], [], [], []
        biases = []
        hfracs = []
        hodmodel, hod = None, None
        pseft = []
        pcheck = []
        
        comm.Barrier()

        for i_hod in range(args.nhod): 
            if wrank == 0 : print('  HOD %i' % i_hod)

            # sample HOD
            iseed = args.seed+i_hod
            #theta_hod = hodtools.sample_conditional_HOD(m_hod, np.log10(mcut), m1=np.log10(m1), seed=i_lhc*9999+iseed, ab_scatter=args.abscatter)
            theta_hod = hodtools.sample_HOD_broad(m_hod, seed=i_lhc*9999+iseed)
            print(f"in rank {comm.rank}, thetahod : ", theta_hod)

            if hodmodel is None:
                np.random.seed(i_lhc*999 + iseed + 23*wrank)
                hodmodel = Galaxies.hodGalaxies_cache(halos, hod_model=m_hod)
                hod = halos.populate(hodmodel, BoxSize=pm.BoxSize, seed=0, **theta_hod)
            else:
                hod.repopulate(**theta_hod)
            #print(f"{wrank} : ", hod.csize, halos.csize)
            
            gtype = hod['gal_type'].compute()
            gtype = np.unique(gtype, return_counts=True)
            gtype = comm.gather(gtype[1])
            
            if comm.rank == 0:
                gtype_clean = []
                for g in gtype:
                    if len(g) == 2: gtype_clean.append(g)
                    elif len(g) == 1: gtype_clean.append([g[0], 0])
                    elif len(g) == 0: gtype_clean.append([0, 0])
                gtype_clean = np.array(gtype_clean)
                if wrank == 0 :
                    print(f"Cleaned gtypes: ", gtype_clean.sum(axis=0))
                gals.append([gtype_clean.sum()] + list(gtype_clean.sum(axis=0)))

            k, p = hodtools.get_power(hod, pm)
            k, pmu, pell = hodtools.get_power_rsd(hod, pm, Nmu=12)
            ps.append(p)
            pmus.append(pmu)
            pells.append(pell)
            hods.append([theta_hod[key] for key in theta_hod.keys()])
            
            gal = hod.to_mesh(BoxSize=pm.BoxSize, Nmesh=pm.Nmesh, compensated=True).compute()
            gal /= gal.cmean()
            gal -= gal.cmean()
            
            ### Fit bias and error. We also return the PT model and plot the transfer function fits
            kmax = 0.3
            bias = flb.fit(gal, nbar, kmax=kmax, verb=not bool(wrank), fit_error=False, return_model=False)
            # bias, eft_model = flb.fit(gal, nbar, kmax=kmax, verb=True, fit_error=True, return_model=True, plot_transfer=True)
            # p = FFTPower(eft_model, mode='1d').power.data
            # k, p = p['k'], p['power'].real
            # pseft.append(p)
            biases.append([bias[key] for key in bias.keys()])
            if wrank == 0: print(f"Bias fit : ", bias)

            comm.Barrier()


        #
        if comm.rank == 0:
            # np.save(save_dir + f"power_eft{wsize}", np.array(pseft).real)

            np.save(save_dir + "power", np.array(ps).real)
            np.save(save_dir + "power_rsd", np.array(pmus).real)
            np.save(save_dir + "power_ell", np.array(pells).real)
            
            np.save(save_dir + "hodp", np.array(hods)) 
            with open(save_dir + "hodp.npy.meta", "w") as f:
                f.write( "\n".join(theta_hod.keys()))

            np.save(save_dir + "biases", np.array(biases)) 
            with open(save_dir + "biases.npy.meta", "w") as f:
                f.write( "\n".join(bias.keys()))

            np.save(save_dir + "gals", np.array(gals))
            with open(save_dir + "gals.npy.meta", "w") as f:
                f.write( "\n".join(["Total", "Centrals", "Satellites"]))

        comm.Barrier()
        
    except Exception as e:
        print(e)

        
comm.Barrier()
sys.exit()
