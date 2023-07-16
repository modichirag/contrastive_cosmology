#!/bin/bash
#SBATCH --nodes=1
#SBATCH -n 4
#SBATCH --partition=ccm
#SBATCH -C skylake
#SBATCH --time=6:00:00
#SBATCH --job-name=wb_bspec
#SBATCH -o ../logs/%x.o%j

# Start from an "empty" module collection.
module purge

# Load in what we need to execute mpirun.
module load modules/2.0-20220630
module load openmpi  gcc/7.5.0
source activate ptorch

# We assume this executable is in the directory from which you ran sbatch.
# Run 1 job per task
JOBNAME=$SLURM_JOB_NAME            # re-use the job-name specified above
N_JOB=$SLURM_NTASKS                # create as many jobs as tasks
echo $N_JOB

config_data=$1
nmodels=20

time mpirun -n ${N_JOB} python -u wandb_bispec.py ${config_data} ${nmodels} 
