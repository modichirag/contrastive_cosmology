import os, sys

dir_rockstar = '/mnt/home/cmodi/Research/Projects/rockstar/'
dir_snapshots = '/mnt/ceph/users/fvillaescusa/Quijote/Snapshots/latin_hypercube_HR/'
snap = 2
dir_halos = '/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/Rockstar/snap%d/'%snap #for redshift z=1

def quijote_HR_LHC(i0, i1):
    ''' function to write the config file for rockstar and slurm file to submit the job
    '''

    # first write config file for given simulation
    for i_lhc in range(i0, i1):
        os.makedirs(dir_halos + "%d"%i_lhc, exist_ok=True)

        a = '\n'.join([
            '#Rockstar Halo Finder',
            'FILE_FORMAT = "AREPO"',
            'PARTICLE_MASS = 0       # must specify (in Msun/h) for ART or ASCII',
            '',
            '# You should specify cosmology parameters only for ASCII formats',
            '# For GADGET2 and ART, these parameters will be replaced with values from the',
            '# particle data file',
            '',
            '# For AREPO / GADGET2 HDF5, you would use the following instead:',
            '# Make sure to compile with "make with_hdf5"!',
            'AREPO_LENGTH_CONVERSION = 1e-3',
            'AREPO_MASS_CONVERSION = 1e+10',
            '',
            'MASS_DEFINITION = "vir" ',
            '',
            '#This specifies the use of multiple processors:',
            'PARALLEL_IO=1',
            'PERIODIC = 1',
            '',
            'FORCE_RES = 0.05 #Force resolution of simulation, in Mpc/h',
            '',
            'MIN_HALO_OUTPUT_SIZE = 20 ',
            '',
            'BOX_SIZE = 1000.00 #Mpc',
            '',
            'INBASE = "%s%i"' % (dir_snapshots, i_lhc),
            'FILENAME="snapdir_00%d/snap_00%d.<block>.hdf5"'%(snap, snap),# 'STARTING_SNAP = 0 ', 'NUM_SNAPS = 5 ',
            'NUM_BLOCKS=8',
            '',
            'OUTBASE = "%s%i"' % (dir_halos, i_lhc),
            '',
            'NUM_READERS = 1',
            'NUM_WRITERS = 8',
            'FORK_READERS_FROM_WRITERS = 1',
            'FORK_PROCESSORS_PER_MACHINE = 8'])
        f = open(os.path.join(dir_halos, str(i_lhc), 'quijote_lhc_hr.%i.cfg' % i_lhc), 'w')
        f.write(a)
        f.close()

    ###############################################
    # next, write slurm file for submitting the job
    a = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -p ccm',
        '#SBATCH -J rockstar_%d_%d'%(i0, i1),
        '#SBATCH --ntasks-per-node=28',
        '#SBATCH --time=06:00:00',
        '#SBATCH -o ./logs/rockstar.o%j',
        '',# 'module load intel-mpi/2017.4.196', 'module load openmpi',
        'module load gcc lib/hdf5',
        '',
        'dir_rockstar="%s" # rockstar repo directory' % dir_rockstar,
        ])
    
    for i_lhc in range(i0, i1):

        a += '\n'.join(['',
            '', 
            'echo "%s"' % i_lhc, 
            'dir_snapshot="%s%i" # output directory' % (dir_halos, i_lhc),
            '',
            'mkdir -p $dir_snapshot',
            '',
            '$dir_rockstar/rockstar -c $dir_snapshot/quijote_lhc_hr.%i.cfg &> $dir_snapshot/server%i.dat &' % (i_lhc, i_lhc),
            '',
            'while [ ! -f "$dir_snapshot/auto-rockstar.cfg" ]; do echo "sleeping"; sleep 1; done',
            '',
            '# deploy jobs ',
            '$dir_rockstar/rockstar -c $dir_snapshot/auto-rockstar.cfg >> $dir_snapshot/output.dat 2>&1',
        ])

    f = open('rockstar_quijote_hr_lhc.%i_%i.slurm' % (i0, i1), 'w')
    f.write(a)
    f.close()
    os.system('sbatch rockstar_quijote_hr_lhc.%i_%i.slurm' % (i0, i1))
    os.system('mv rockstar_quijote_hr_lhc.%i_%i.slurm ./jobs/rockstar_quijote_hr_lhc.%i_%i.slurm ' %(i0, i1, i0, i1))
    #os.system('rm rockstar_quijote_hr_lhc.%i.slurm' % i_lhc)
    return None





def quijote_HR_LHC_pid(i0, i1):
    ''' function to write the config file for rockstar and slurm file to submit the job 
    '''
    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -p ccm',
        '#SBATCH -J rockstarpid_%d_%d'%(i0, i1),
        '#SBATCH --ntasks-per-node=28',
        '#SBATCH --time=05:59:59',
        '#SBATCH -o logs/rockstar_pid.o%j',
        '',# 'module load intel-mpi/2017.4.196', 'module load openmpi',
        'module load gcc lib/hdf5',
        '',  
        ''])
    
    n_run = 0 
    for i_lhc in range(i0, i1): 

        if not os.path.isdir(os.path.join(dir_halos, str(i_lhc))): 
            os.system('mkdir %s' % os.path.join(dir_halos, str(i_lhc)))

        # next, write slurm file for submitting the job
        slurm += '\n'.join([
            '', 
            'echo "%s"' % i_lhc, 
            'dir_rockstar="%s" # rockstar repo directory' % dir_rockstar,
            'dir_snapshot="%s%i" # output directory' % (dir_halos, i_lhc),
            '',
            '$dir_rockstar/util/find_parents $dir_snapshot/out_0.list 1000 > $dir_snapshot/out_0_pid.list', 
            #'', #'python /mnt/home/chahn/projects/simbig/cmass/bin/postprocess_rockstar.py %i' % i_lhc, 
            ''])
        n_run += 1

    if n_run == 0: return None

    f = open('rockstar_quijote_hr_lhc_pid.%i_%i.slurm' % (i0, i1), 'w')
    f.write(slurm)
    f.close()
    os.system('sbatch rockstar_quijote_hr_lhc_pid.%i_%i.slurm' % (i0, i1))
    os.system('mv rockstar_quijote_hr_lhc_pid.%i_%i.slurm ./jobs/rockstar_quijote_hr_lhc_pid.%i_%i.slurm' % (i0, i1, i0, i1))
    return None




def quijote_HR_LHC_postprocess(i0, i1):
    ''' function to write the config file for rockstar and slurm file to submit the job 
    '''
    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -p ccm',
        '#SBATCH -J postpr_%d_%d'%(i0, i1),
        '#SBATCH --ntasks-per-node=1',
        '#SBATCH --time=05:59:59',
        '#SBATCH -o logs/postprocess_rock.o%j',
        '',# 'module load intel-mpi/2017.4.196', 'module load openmpi',
        'module load gcc lib/hdf5',
        '',  
        ''])
    
    n_run = 0 
    for i_lhc in range(i0, i1): 
        fbf = os.path.join(dir_halos, str(i_lhc), 'quijote_LHC_HR%i.%d.rockstar.bf' % (i_lhc, snap))

        if not os.path.isdir(os.path.join(dir_halos, str(i_lhc))): 
            os.system('mkdir %s' % os.path.join(dir_halos, str(i_lhc)))

        # next, write slurm file for submitting the job
        slurm += '\n'.join([
            '', 
            'echo "%s"' % i_lhc, 
            '',
            'time srun python -u postprocess_rockstar.py lhc %i' % i_lhc, 
            ''])
        n_run += 1

    if n_run == 0: return None

    f = open('rockstar_quijote_hr_lhc_postprocess.%i_%i.slurm' % (i0, i1), 'w')
    f.write(slurm)
    f.close()
    os.system('sbatch rockstar_quijote_hr_lhc_postprocess.%i_%i.slurm' % (i0, i1))
    os.system('mv rockstar_quijote_hr_lhc_postprocess.%i_%i.slurm ./jobs/rockstar_quijote_hr_lhc_postprocess.%i_%i.slurm' % (i0, i1, i0, i1))
    return None


for j in range(2000): 
    if os.path.exists('/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/Rockstar//%d/snapshot_2.bf'%j): 
        pass
    else:
        print(j)
        #quijote_HR_LHC_postprocess(j, (j+1))
        #quijote_HR_LHC(j, j+1)
        #quijote_HR_LHC_pid(j, j+1)
        
for i in range(0, 40): 
    #quijote_HR_LHC(i, i+1)
    #quijote_HR_LHC_pid(i*50, (i+1)*50)
    #quijote_HR_LHC_postprocess(i*50, (i+1)*50)
    pass

# ## To loop over and rerun missed simulations due to exceeding runtime
# for i in range(1, 40): 
#     i0, i1 = i*50, (i+1)*50
#     for j in range(i0, i1):
#         if os.path.exists('/mnt/ceph/users/cmodi/Quijote/latin_hypercube_HR/Rockstar/snap2/%d/output.dat'%j): 
#             pass
#         else:
#             print(j)
#             break
#     print(j, i1)
#     quijote_HR_LHC(j, i1)



