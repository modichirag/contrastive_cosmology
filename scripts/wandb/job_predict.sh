#!/bin/bash
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --partition=ccm
#SBATCH -C skylake
#SBATCH --time=2:00:00
#SBATCH --job-name=predsim
#SBATCH -o ./logs/%x.o%j

# Start from an "empty" module collection.
module purge

# Load in what we need to execute mpirun.
module load modules/2.0-20220630
module load gcc/7.5.0 openmpi/1.10.7
source activate ptorch

# We assume this executable is in the directory from which you ran sbatch.
# Run 1 job per task
JOBNAME=$SLURM_JOB_NAME            # re-use the job-name specified above
N_JOB=$SLURM_NTASKS                # create as many jobs as tasks
echo $N_JOB

i0=$1
i1=$2
ihod=0
echo "LHC index" $i0 $i1

# for((i=$i0 ; i<=$i1 ; i+=1))
# do
#     echo $i $ihod
#     time python -u predict_sim.py $i $ihod 'Rockstar'  'zheng07_velab' 'zheng07'
#     time python -u predict_sim.py $i $ihod 'Rockstar'  'zheng07' 'zheng07'
#     time python -u predict_sim.py $i $ihod 'Rockstar'  'zheng07_velab' 'zheng07_velab'
#     time python -u predict_sim.py $i $ihod 'Rockstar'  'zheng07' 'zheng07_velab'
# done

for((i=$i0 ; i<=$i1 ; i+=1))
do
    echo $i $ihod
    #time python -u predict_sim.py $i $ihod 'FoF'  'zheng07_velab' 'zheng07'
    #time python -u predict_sim.py $i $ihod 'FoF'  'zheng07' 'zheng07'
    time python -u predict_sim.py $i $ihod 'FoF'  'zheng07_velab' 'zheng07_velab'
    #time python -u predict_sim.py $i $ihod 'FoF'  'zheng07' 'zheng07_velab'
done


# for((i=0 ; i<=100 ; i+=1))
# do
#     echo "inner loop"
#     for ((j=${i0} ; j<${i1} ; j+=1))
#     do
#         echo $i $j
#         time python -u predict_sim.py $i $j 
#         #time python -u predict_sim2.py $i $j
#     done
#     wait 
#     echo "done for simulation " $i
# done

wait

echo
echo "All done. "
