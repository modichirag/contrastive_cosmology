import os, sys
import time 

##for i in {0..2000..50}; do j=$((i+50)); echo $i $j; sbatch flscript1.sh  $i  $j ; done


def make_hod(script, i0, i1, model, nhod, z, seed, nbar, fiducial, finder, abscatter, rewrite):
    
    ''' function to write the config file for rockstar and slurm file to submit the job 
    '''
    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -p ccm',
        '#SBATCH -c 1',
        '#SBATCH -t 180',
        '#SBATCH -J make_hod',
        '#SBATCH -o logs/makehod.o%j',
        '',
        'module --force purge',
        'module load modules-traditional',
        'module load cuda/11.0.3_450.51.06',
        'module load cudnn/v8.0.4-cuda-11.0',
        'module load slurm',
        'module load gcc',
        'module load openmpi',
        'source activate defpyn',
        '',  
        '',
        'id0=%s'%i0,
        'id1=%s'%i1,
        'model=%s'%model,
        'nhod=%s'%nhod,
        'z=%s'%z,
        'seed=%s'%seed,
        'nbar=%s'%nbar,
        'fiducial=%s'%fiducial,
        'finder=%s'%finder,
        'abscatter=%0.2f'%abscatter,
        'rewrite=%s'%rewrite,
        '',
        'time srun -N 1 -n 1 python -u %s --model $model --z $z --id0 $id0 --id1 $id1 --seed $seed --nhod $nhod --nbar $nbar  --fiducial $fiducial --finder $finder --abscatter $abscatter --rewrite $rewrite'%script,
        ''
    ])
    
    f = open('makehod.%i_%i.slurm' % (i0, i1), 'w')
    f.write(slurm)
    f.close()
    os.system('sbatch makehod.%i_%i.slurm' % (i0, i1))
    os.system('mv makehod.%i_%i.slurm ./submitjobs/makehod.%i_%i.slurm' % (i0, i1, i0, i1))
    return None


script = 'make_hod.py'
model = "zheng07_ab"
nhod = 10
z = 1.0
seed = 0
nbar = 0.00055
fiducial = 0
finder = 'rockstar'
rewrite = 1
abscatter = 0.9

sim_per_job=50
#for i0 in range(0, 1, sim_per_job):
for i0 in range(0, 2000, sim_per_job):

    i1 = i0 + sim_per_job
    print(i0, i1)

    make_hod(script, i0, i1, 
             model=model, 
             nhod=nhod, 
             z=z, 
             seed=seed, 
             nbar=nbar, 
             fiducial=fiducial, 
             finder=finder,
             abscatter=abscatter,
             rewrite=rewrite)   # 


# ### Rerun missed simulations
# path = '/mnt/ceph/users/cmodi/contrastive/data/z%d-N%04d/%s'%(z*10, nbar*1e4, model)
# if finder == 'rockstar': path = path + '-rock/'
# else: path = path + '/' 
# checkdate = False
# daterun = 29

# for i0 in range(0, 2000, sim_per_job):

#     i1 = i0 + sim_per_job
#     rerun = 0 

#     for j in range(i0, i1):
#         if os.path.exists(path + '/%04d/power_ell.npy'%j): 
#             mtime = os.path.getmtime(path + '/%04d/power_ell.npy'%j)
#             date = time.ctime(mtime).split(" ")[2]
#             if checkdate & (int(date) < daterun): 
#                 print(date, j, i1)
#                 rerun = 1
#                 break
#             pass

#         else:
#             print(j, i1)
#             rerun = 1
#             break

#     if rerun:
#         make_hod(script, j, i1,
#                  model=model, 
#                  nhod=nhod, 
#                  z=z, 
#                  seed=seed, 
#                  nbar=nbar, 
#                  fiducial=fiducial, 
#                  finder=finder, 
#                  rewrite=rewrite)   # 
            
