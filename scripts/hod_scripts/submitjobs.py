import os, sys

##for i in {0..2000..50}; do j=$((i+50)); echo $i $j; sbatch flscript1.sh  $i  $j ; done


def make_hod(script, i0, i1, model, nhod, z, seed, nbar, fiducial, finder, rewrite):
    
    ''' function to write the config file for rockstar and slurm file to submit the job 
    '''
    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -p ccm',
        '#SBATCH -c 1',
        '#SBATCH -t 120',
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
        'rewrite=%s'%rewrite,
        '',
        'time srun -N 1 -n 1 python -u %s --model $model --z $z --id0 $id0 --id1 $id1 --seed $seed --nhod $nhod --nbar $nbar  --fiducial $fiducial --finder $finder --rewrite $rewrite'%script,
        ''
    ])
    
    f = open('makehod.%i_%i.slurm' % (i0, i1), 'w')
    f.write(slurm)
    f.close()
    os.system('sbatch makehod.%i_%i.slurm' % (i0, i1))
    os.system('mv makehod.%i_%i.slurm ./jobs/makehod.%i_%i.slurm' % (i0, i1, i0, i1))
    return None



script = 'make_hod.py'
model = "zheng07"
nhod = 10
z = 1.0
seed = 0
nbar = 0.0001
fiducial = 0
finder = 'rockstar'
rewrite = 1

#for i0 in range(0, 50, 50):
for i0 in range(50, 2000, 50):

    i1 = i0 + 50
    print(i0, i1)

    make_hod(script, i0, i1, 
             model=model, 
             nhod=nhod, 
             z=z, 
             seed=seed, 
             nbar=nbar, 
             fiducial=fiducial, 
             finder=finder, 
             rewrite=rewrite)   # 
