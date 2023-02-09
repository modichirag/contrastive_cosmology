#!/bin/bash
#SBATCH --nodes=1
#SBATCH -n 4
#SBATCH --partition=ccm
#SBATCH -C skylake
#SBATCH --time=4:00:00
#SBATCH --job-name=wandb
#SBATCH -o logs/%x.o%j

# Start from an "empty" module collection.
module purge

# Load in what we need to execute mpirun.
module load gcc/7.5.0 openmpi
source activate ptorch

# We assume this executable is in the directory from which you ran sbatch.
# Run 1 job per task
JOBNAME=$SLURM_JOB_NAME            # re-use the job-name specified above
N_JOB=$SLURM_NTASKS                # create as many jobs as tasks
echo $N_JOB

#sweep_id="in2xjz5u"
#config_data="./configs/config_hodells_quijote.yaml"
config_data=$1
nmodels=10

time mpirun -n ${N_JOB} python -u wandb_hodells.py ${config_data} ${nmodels} 

# i0=0
# i1=1
# echo "run for range"
# echo $i0 $i1
# echo "start loop"
# # for((i=${i0} ; i<=${i1} ; i+=${N_JOB}))
# # do
# #     echo $i $((i+N_JOB))
# #     echo "inner loop"
# #     iend=$((i+N_JOB))
# #     for ((j=$i ; j<$iend ; j+=1))
# #     do
# #         j1=$((j+1))
# #         echo $j $j1
# #         time python -u wandb_hodells.py ${sweep_id} ${nmodels} ${config_data} &
# #     done
# #     echo "inner loop exit"
# #     wait 
# # done
# for((i=${i0} ; i<=${i1} ; i+=${N_JOB}))
# do
#     echo $i $((i+N_JOB))
#     echo "inner loop"
#     time mpirun -n ${N_JOB} python -u wandb_hodells.py ${sweep_id} ${nmodels} ${config_data} 
#     wait 
# done

# #Wait for all
# wait
 
# echo
# echo "All done. Checking results:"

# #time python -u make_hod.py --z $z --finder FoF --model zheng07 --id0 $j --id1 $((j+1)) --nhod $nhod --simulation quijote &
