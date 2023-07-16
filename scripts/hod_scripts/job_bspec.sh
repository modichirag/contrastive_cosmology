#!/bin/bash
#SBATCH --nodes=1
#SBATCH -n 10
#SBATCH --partition=ccm
#SBATCH -C skylake
#SBATCH --time=8:00:00  ###for 50 sims
#SBATCH --job-name=bspec
#SBATCH -o logs/%x.o%j

# Start from an "empty" module collection.
module purge

# Load in what we need to execute mpirun.
module load modules/2.0-20220630
module load gcc/7.5.0 openmpi/1.10.7 fftw
source activate defpyn


# We assume this executable is in the directory from which you ran sbatch.

# Run 1 job per task
JOBNAME=$SLURM_JOB_NAME            # re-use the job-name specified above
N_JOB=$SLURM_NTASKS                # create as many jobs as tasks
echo $N_JOB

i0=$1                           # read in from command line
i1=$2                           # read in from command line
z=1.0
nbar=0.00045
nhod=10
simulation="quijote"
model="zheng07_velab"
finder="FoF" 
rewrite=1

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
        time python -u make_hod_bspec.py --z $z --finder $finder --model $model --id0 $j --id1 $((j+1)) --nhod $nhod --nbar $nbar  --simulation $simulation --rewrite $rewrite &
    done
    echo "inner loop exit"
    wait 
done

#Wait for all
wait
 
echo
echo "All done. Checking results:"

#time python -u make_hod.py --z $z --finder FoF --model zheng07 --id0 $j --id1 $((j+1)) --nhod $nhod --simulation quijote &
