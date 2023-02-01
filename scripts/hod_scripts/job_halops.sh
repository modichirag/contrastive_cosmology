#!/bin/bash
#SBATCH --nodes=1
#SBATCH -n 10
#SBATCH --partition=ccm
#SBATCH -C skylake
#SBATCH --time=4:00:00
#SBATCH --job-name=halops
#SBATCH -o logs/%x.o%j

# Start from an "empty" module collection.
module purge

# Load in what we need to execute mpirun.
module load gcc/7.5.0 openmpi
source activate tf210

# We assume this executable is in the directory from which you ran sbatch.

# Run 1 job per task
JOBNAME=$SLURM_JOB_NAME            # re-use the job-name specified above
N_JOB=$SLURM_NTASKS                # create as many jobs as tasks
echo $N_JOB

i0=$1
i1=$2
echo "run for range"
echo $i0 $i1
echo "start loop"

for((i=${i0} ; i<=${i1} ; i+=${N_JOB}))
do
    echo $i $((i+N_JOB))
    echo "inner loop"
    iend=$((i+N_JOB))
    for ((j=$i ; j<$iend ; j+=1))
    do
        j1=$((j+1))
        echo $j $j1
        #time python -u halo_power.py --z 0.5 --id0 $j --id1 $((j+1)) --simulation fastpm  & 
        time python -u halo_power.py --z 0.5 --id0 $j --id1 $((j+1)) --simulation quijote  & 
    done
    echo "inner loop exit"
    wait 
done

#Wait for all
wait
 
echo
echo "All done. Checking results:"
