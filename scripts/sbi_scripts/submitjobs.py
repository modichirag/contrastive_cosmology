import os, sys

##for i in {0..2000..50}; do j=$((i+50)); echo $i $j; sbatch flscript1.sh  $i  $j ; done
#python -u analysis.py --dataloader hod_ells_offset --datapath z10-N0001/zheng07_ab/  --logit 0 --suffix offset-nolog 


def analysis(dataloader, datapath, suffix, jobname):
    
    ''' function to write the config file for rockstar and slurm file to submit the job 
    '''
    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -p ccm',
        '#SBATCH -c 1',
        '#SBATCH -t 60',
        '#SBATCH -J %s'%jobname,
        '#SBATCH -o logs/.o%j',
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
        'time srun -N 1 -n 1 python -u analysis.py --dataloader %s --datapath %s --logit 0 --suffix %s'%(dataloader, datapath, suffix),
        ''
    ])
    
    f = open('analysis.slurm' , 'w')
    f.write(slurm)
    f.close()
    os.system('sbatch analysis.slurm' )
    #os.system('rm analysis.slurm ' )
    return None



dataloader = "hod_ells_offset"
suffix = "offset-nolog"

# for datapath in ['z10-N0001/zheng07_ab/', 
#                  'z10-N0001/zheng07_velab/', 
#                  'z10-N0001/zheng07-rock/', 
#                  'z10-N0001/zheng07_ab-rock/', 
#                  'z10-N0001/zheng07_velab-rock/'
#                  ]:

for datapath in ['z10-N0004/zheng07/', 
                 'z10-N0004/zheng07-rock/', 
                 ]:

    jobname = datapath.split('/')[1].split('_')[-1]
    print(datapath.split('/')[1].split('_')[-1])
    analysis(dataloader, datapath, suffix, jobname=jobname)
