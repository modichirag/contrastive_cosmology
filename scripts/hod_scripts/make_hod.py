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
parser.add_argument('--id0', type=int, default=0, help='first quijote seed')
parser.add_argument('--id1', type=int, default=2000, help='last quijote seed')
parser.add_argument('--seed', type=int, default=0, help='seed between 0-999')
parser.add_argument('--nhod', type=int, default=10, help='number of HOD sims')
parser.add_argument('--alpha', type=float, default=0.7, help='alpha, slope of satellites')
parser.add_argument('--satfrac', type=float, default=0.2, help='satellite fraction')
parser.add_argument('--nbar', type=float, default=0.0001, help='numbder density of galaxies')
parser.add_argument('--suffix', type=str, default="", help='suffix for parent folder with z and nnbar')
parser.add_argument('--suffix2', type=str, default="", help='suffix for subfolder with model name')
parser.add_argument('--rewrite', type=int, default=0, help='rewrite files which already exist')
parser.add_argument('--fiducial', type=int, default=0, help='for fiducial simulations')
parser.add_argument('--abscatter', type=float, default=0.2, help='scatter of zhengab')
args = parser.parse_args()

print(args)

m_hod = args.model
id0, id1, nhod = args.id0, args.id1, args.nhod
zred, alpha_fid = args.z, args.alpha
nbar, satfrac = args.nbar, args.satfrac
#
bs, nc = 1000, 256
pm = ParticleMesh(Nmesh=[nc, nc, nc], BoxSize=bs, dtype='f8')
if nbar != 0: data_dir = '/mnt/ceph/users/cmodi/contrastive/data/z%02d-N%04d/'%(zred*10, nbar/1e-4)
else: data_dir = '/mnt/ceph/users/cmodi/contrastive/data/z%02d/'%(zred*10)
if args.suffix != "": data_dir = data_dir[:-1] + '-%s/'%args.suffix
os.makedirs(data_dir, exist_ok=True)

#model suffix
namefid, namefinder, namescatter = '', '', ''
if (args.fiducial == 1) : namefid = '-fid'
if args.finder == 'rockstar': namefinder = '-rock'
if args.abscatter != 0.2: namescatter = '-abs%d'%(args.abscatter*10)
if args.suffix2 != "": args.suffix2 =  "-" + args.suffix2
data_dir = data_dir + '%s/'%(args.model + namefid + namefinder + namescatter + args.suffix2)
print("Save in data directory : ", data_dir)
os.makedirs(data_dir, exist_ok=True)
#
for i_lhc in range(id0, id1):
    print('LHC %i' % i_lhc)
    # read in halo catalog
    if args.fiducial: 
        halos = Halos.Quijote_fiducial_HR(i_lhc, z=zred, finder=args.finder)
    else:
        halos = Halos.Quijote_LHC_HR(i_lhc, z=zred, finder=args.finder)
        
    
    print(halos['Mass'].compute())
    halos = halos.sort('Mass', reverse=True)
    hmass = halos['Mass'].compute()
    print(hmass)
    hsize = hmass.shape[0]

    mcut, m1 = hodtools.setup_hod(halos, nbar=nbar, satfrac=satfrac, bs=bs, alpha_fid=alpha_fid)
    ps, ngals, gals, pmus, pells, hods = [], [], [], [], [], []
    ngals = []
    hfracs = []
    save_dir = data_dir + '%04d/'%i_lhc
    os.makedirs(save_dir, exist_ok=True)        

    #
    #
    if args.rewrite: do_hod = True
    else: do_hod = False
    for i_hod in range(nhod):
        if not os.path.isfile(save_dir + 'power_%d.npy'%i_hod) : do_hod = True
    if not do_hod: continue
    #

    hodmodel, hod = None, None
    for i_hod in range(nhod): 
        print('  HOD %i' % i_hod)
        # sample HOD
        iseed = args.seed+i_hod
        theta_hod = hodtools.sample_conditional_HOD(m_hod, np.log10(mcut), m1=np.log10(m1), seed=i_lhc*9999+iseed, ab_scatter=args.abscatter)
        np.save(save_dir+'hodp_%d'%iseed, theta_hod)
        #hod = Galaxies.hodGalaxies(halos, theta_hod, seed=0, hod_model=m_hod)
        if hodmodel is None: 
            hodmodel = Galaxies.hodGalaxies_cache(halos, hod_model=m_hod)
            hod = halos.populate(hodmodel, seed=0, **theta_hod)
        else:
            hod.repopulate(**theta_hod)
        #hod.save(save_dir+'cat_%i.bf' % iseed) 

        #calculate some summary numbers on galaxy types
        galsum = hodtools.galaxy_summary(hod, bs=bs, filename=save_dir+'gals_%i.json'%iseed)
        print(galsum)
        ngals.append(galsum['number density'])
        k, p = hodtools.get_power(hod, pm)
        k, pmu, pell = hodtools.get_power_rsd(hod, pm, Nmu=12)
        #np.save(save_dir+"power_%d"%iseed, p)
        #np.save(save_dir+"power_rsd_%d"%iseed, pmu)
        #np.save(save_dir+"power_ell_%d"%iseed, pell)

        ps.append(p)
        pmus.append(pmu)
        pells.append(pell)
        gals.append([galsum['total'], galsum['centrals'], galsum['satellites']])
        hods.append([theta_hod[key] for key in theta_hod.keys()])
        mlims = [10**(hods[-1][0] - j*hods[-1][1]) for j in range(3)]
        sigmin = (hods[-1][0] - np.log10(hmass[-1]))/hods[-1][1]
        hfracs.append([(hmass > mlims[j]).sum()/hsize for j in range(2)] + [sigmin]) 
        print(hfracs[-1])

    np.save(save_dir + "power", np.array(ps))
    np.save(save_dir + "power_rsd", np.array(pmus))
    np.save(save_dir + "power_ell", np.array(pells))
    np.save(save_dir + "gals", np.array(gals))
    np.save(save_dir + "hodp", np.array(hods)) 
    np.save(save_dir + "hfracs", np.array(hfracs)) 
    #Make PS figure
    print("mean number denisty : ", np.mean(ngals))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4), sharex=True, sharey=True)
    for i in range(len(ps)):
        ax[0].plot(ps[i][0], ps[i][1])
        #ax[1].plot(ps[i][0], ps[i][2])
    for axis in ax:
        axis.loglog()
        axis.grid(which='both')
    plt.tight_layout()
    plt.savefig(save_dir + 'pk.png')
