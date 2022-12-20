import os, sys

##for i in {0..2000..50}; do j=$((i+50)); echo $i $j; sbatch flscript1.sh  $i  $j ; done
#python -u analysis.py --dataloader hod_ells_offset --datapath z10-N0001/zheng07_ab/  --logit 0 --suffix offset-nolog 

#def analysis(dataloader, datapath, suffix, jobname, retrain, nlayers, nhidden, kmax, standardize, ells, submit=True, srun=False):
#analysis(dataloader, datapath, suffix, jobname=jobname, retrain=retrain, nlayers=nlayers, nhidden=nhidden, standardize=standardize, kmax=kmax, ells=ells, submit=submit, srun=srun)

def analysis(command, jobname ):
    ''' function to write the config file for rockstar and slurm file to submit the job 
    '''

    
    slurm = '\n'.join([
        '#!/bin/bash',
        '#SBATCH -p ccm',
        '#SBATCH -c 1',
        '#SBATCH -t 120',
        '#SBATCH -J %s'%jobname,
        '#SBATCH -o logs/sbi.o%j',
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
        #'time srun -N 1 -n 1 %s'%(command),
        '%s'%(command),
        ''
    ])

    f = open('analysis.slurm' , 'w')
    f.write(slurm)
    f.close()
    os.system('sbatch analysis.slurm' )
    os.system('mv analysis.slurm ./scripts/analysis.slurm' )
    return None




def diff_analysis(modelpath, jobname, submit=True):
    ''' function to write the config file for rockstar and slurm file to submit the job 
    '''
    if submit: 
        slurm = '\n'.join([
        '#!/bin/bash',
            '#SBATCH -p ccm',
            '#SBATCH -c 1',
            '#SBATCH -t 60',
            '#SBATCH -J %s'%jobname,
            '#SBATCH -o logs/sbi.o%j',
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
            'time srun -N 1 -n 1 python -u diff_analysis.py --modelpath %s'%(modelpath),
            ''
        ])

        f = open('diff_analysis.slurm' , 'w')
        f.write(slurm)
        f.close()
        os.system('sbatch diff_analysis.slurm' )
        os.system('rm diff_analysis.slurm ' )
    else:
        os.system('time python -u diff_analysis.py --modelpath %s'%(modelpath))

    return None


def submit_analysis(submit=False, srun=False):
    dataloader = "hod_ells_offset"
    #dataloader = "hod_ells"
    z = 1.0
    nbar = 0.00055
    folder = '//z%d-N%04d/%s/'%(z*10, nbar*1e4, '%s')
    fiducial = 0
    seed = 0
    #
    nlayers = 5
    nhidden = 32
    kmax = 0.3
    
    retrain = 1
    #models = ['zheng07', 'zheng07_ab', 'zheng07_velab']
    finders = ['fof', 'rockstar']
    models = ['zheng07']
    finder = 'rockstar'
    standardize = 1
    ells = "024" #set directly in the analysis code
    fithod = 0
    logit = 0
    suffix = "offset-nolog-nn-nohod2-check"
    #suffix = "nolog-nn"
    if standardize == 0: suffix = "offset-nolog-nowhite"

    if srun: sjob = 'srun -N 1 -n 1'
    else: sjob = ''

    for kmax in [0.5]:
        for model in models:

            datapath = folder%model
            if finder == 'rockstar': datapath = datapath[:-1] + '-rock/'
            print(datapath)

            command = ''
            #for nhidden in [32, 64, 128]:
            for nhidden in [32]:

                jobname = model+"%d"%nhidden
                print(jobname)
                #command =  'python -u analysis.py --dataloader %s --datapath %s --logit 0 --suffix %s --nlayers %d --nhidden %d --standardize %d --kmax %0.2f  --retrain %d --ells %s --fithod %d'%(dataloader, datapath, suffix, nlayers, nhidden, standardize, kmax, retrain, ells, fithod)
                command = command + 'time %s python -u analysis_nn.py --dataloader %s --datapath %s --logit 0 --suffix %s --nlayers %d --nhidden %d --standardize %d --kmax %0.2f  --retrain %d --ells %s --fithod %d  \n'%(sjob, dataloader, datapath, suffix, nlayers, nhidden, standardize, kmax, retrain, ells, fithod)
                print(command)


            if submit: 
                analysis(command, jobname)
            elif srun: 
                os.system('time srun %s'%command)
            else:
                os.system('time %s'%command)



def submit_diff_analysis(submit, srun=False):

    dataloader = "hod_ells_offset"
    z = 1.0
    nbar = 0.00055
    fiducial = 0
    seed = 0
    nlayers = 5
    nhidden = 32
    kmax = 0.3
    folder = 'z%d-N%04d/%s/'%(z*10, nbar*1e4, '%s')

    #odels = ['zheng07', 'zheng07_ab', 'zheng07_velab']
    models = ['zheng07']
    finders = ['fof', 'rockstar']
    finder = 'rockstar'
    standardize = 1
    if standardize == 0:     suffix = "offset-nolog-nowhite"
    else:  suffix = "offset-nolog-ells02"

    modelpaths = ['maf-kmax50-nl05-nh32-s000-ells-nn', 
                  'maf-kmax50-nl05-nh32-s000-ells-nolog-nn', 
                  'maf-kmax50-nl05-nh32-s000-ells-nolog-nn-nohod', 
                  'maf-kmax50-nl05-nh32-s000-ells-offset-nolog-nn-nohod',
                  'maf-kmax50-nl05-nh32-s000-ells-offset-nolog-nn']

    for model in models:
        #or nhidden in [32, 64, 128]:
        for f in modelpaths:
            modelpath = folder%model
            if finder == 'rockstar': modelpath = modelpath[:-1] + '-rock/'
            #modelpath = modelpath + 'maf-kmax%02d-nl%02d-nh%02d-s000-ells-%s/'%(kmax*100, nlayers, nhidden, suffix)
            modelpath = modelpath + f + '/'

            print(modelpath)
            
            if srun: os.system('time srun python -u diff_analysis.py --modelpath %s'%(modelpath))
            else: diff_analysis(modelpath, jobname='diffanalys', submit=submit)



if __name__=="__main__":

    #submit_diff_analysis(submit=False, srun=False)
    submit_analysis(submit=False, srun=False)
