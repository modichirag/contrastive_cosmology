'''
Generate galaxy catalog conditioned on number density with setup HOD
'''
import os, sys 
import numpy as np 
import argparse, json
from pmesh.pm import ParticleMesh, RealField
from nbodykit.lab import FFTPower, ProjectedFFTPower, ArrayMesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#
sys.path.append('../../src/')
import halos as Halos
import galaxies as Galaxies
import tools, hodtools

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, help='HOD model name')
parser.add_argument('--z', type=float, help='redshift')
parser.add_argument('--finder', type=str, help='which halo finder')
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

print(args)

m_hod = args.model
zred = args.z
nbar = args.nbar
bs = 1000
pm = ParticleMesh(Nmesh=[args.nc, args.nc, args.nc], BoxSize=bs, dtype='f8')

##
if (args.fiducial == 1) :
    data_dir = f'/mnt/ceph/users/cmodi/contrastive/data-fiducial/{args.simulation}/{args.finder}/z{int(zred*10):02d}-N{int(nbar/1e-4):04d}/'
else:
    data_dir = f'/mnt/ceph/users/cmodi/contrastive/data/{args.simulation}/{args.finder}/z{int(zred*10):02d}-N{int(nbar/1e-4):04d}/'
if args.suffix != "": data_dir = data_dir[:-1] + '-%s/'%args.suffix
os.makedirs(data_dir, exist_ok=True)

#model suffix
namescatter = ''
if args.abscatter != 0.2: namescatter = '-abs%d'%(args.abscatter*10)
if args.suffix2 != "": args.suffix2 =  "-" + args.suffix2
data_dir = data_dir + f'{args.model}{namescatter}{args.suffix2}/'
print(f"\nSave in data directory : {data_dir}\n")
os.makedirs(data_dir, exist_ok=True)


##Loop!
for i_lhc in range(args.id0, args.id1):

    print('LHC %i' % i_lhc)
    save_dir = data_dir + '%04d/'%i_lhc
    os.makedirs(save_dir, exist_ok=True)        
    with open(save_dir + 'args.json', 'w') as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)

        
    #Do not rerun the simulations already run
    if args.rewrite: do_hod = True
    else: do_hod = False
    if not os.path.isfile(save_dir + 'power_rsd.npy') : do_hod = True
    if not do_hod: continue

    
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

    
    halos = halos.sort('Mass', reverse=True)
    hmass = halos['Mass'].compute()
    print("Halo read. Mass: ", hmass)
    hsize = hmass.shape[0]

    #Setup HOD
    mcut, m1 = hodtools.setup_hod(halos, nbar=nbar, satfrac=args.satfrac, bs=bs, alpha_fid=args.alpha)


    #
    ps, ngals, gals, pmus, pells, hods = [], [], [], [], [], []
    ngals = []
    hfracs = []
    hodmodel, hod = None, None
    
    for i_hod in range(args.nhod): 
        print('  HOD %i' % i_hod)
        
        # sample HOD
        iseed = args.seed+i_hod
        theta_hod = hodtools.sample_conditional_HOD(m_hod, np.log10(mcut), m1=np.log10(m1), seed=i_lhc*9999+iseed, ab_scatter=args.abscatter)
        
        #hod = Galaxies.hodGalaxies(halos, theta_hod, seed=0, hod_model=m_hod)
        if hodmodel is None: 
            hodmodel = Galaxies.hodGalaxies_cache(halos, hod_model=m_hod)
            hod = halos.populate(hodmodel, seed=0, **theta_hod)
        else:
            hod.repopulate(**theta_hod)
        
        #calculate some summary numbers on galaxy types
        #galsum = hodtools.galaxy_summary(hod, bs=bs, filename=save_dir+'gals_%i.json'%iseed)
        galsum = hodtools.galaxy_summary(hod, bs=bs, filename=None)
        print(galsum)
        ngals.append(galsum['number density'])
        k, p = hodtools.get_power(hod, pm)
        k, pmu, pell = hodtools.get_power_rsd(hod, pm, Nmu=12)
        
        ps.append(p)
        pmus.append(pmu)
        pells.append(pell)
        gals.append([galsum['total'], galsum['centrals'], galsum['satellites']])
        hods.append([theta_hod[key] for key in theta_hod.keys()])

        #diagnostics to check what halo fraction is used.
        mlims = [10**(hods[-1][0] - j*hods[-1][1]) for j in range(3)]
        sigmin = (hods[-1][0] - np.log10(hmass[-1]))/hods[-1][1]
        hfracs.append([(hmass > mlims[j]).sum()/hsize for j in range(3)] + [sigmin]) 
        print(hfracs[-1])

        
    #
    np.save(save_dir + "power", np.array(ps).real)
    np.save(save_dir + "power_rsd", np.array(pmus).real)
    np.save(save_dir + "power_ell", np.array(pells).real)
    np.save(save_dir + "gals", np.array(gals))
    np.save(save_dir + "hodp", np.array(hods)) 
    np.save(save_dir + "hfracs", np.array(hfracs)) 
    with open(save_dir + "gals.npy.meta", "w") as f:
        f.write( "\n".join(["Total", "Centrals", "Satellites"]))
    with open(save_dir + "hodp.npy.meta", "w") as f:
        f.write( "\n".join(theta_hod.keys()))
    with open(save_dir + "hfracs.npy.meta", "w") as f:
        f.write( "\n".join(["halo frac>Mmin", "halo frac>Mmin-1\sigma", "halo frac>Mmin-2\sigma", "sigma difference in logMmin & log(min halo mass)"]))

    print("mean number denisty : ", np.mean(ngals))

    # #Make PS figure
    # fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    # for i in range(len(ps)):
    #     ax[0].plot(ps[i][0], ps[i][1])
    #     #ax[1].plot(ps[i][0], ps[i][2])
    # for axis in ax:
    #     axis.loglog()
    #     axis.grid(which='both')
    # plt.tight_layout()
    # plt.savefig(save_dir + 'pk.png')
