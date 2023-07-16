#!/bin/bash
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --partition=ccm
#SBATCH -C skylake
#SBATCH --time=8:00:00
#SBATCH --job-name=rankells
#SBATCH -o ../logs/%x.o%j

# Start from an "empty" module collection.
module purge

# Load in what we need to execute mpirun.
module load modules/2.0-20220630
module load gcc/7.5.0 openmpi/1.10.7
source activate ptorch

## time python -u rank_statisitcs.py {simulation of data} {finder of data} {hod of data} {hod of training}

#time python -u rank_statistics.py quijote FoF zheng07_velab zheng07_velab 10
#time python -u rank_statistics.py quijote Rockstar zheng07_velab zheng07_velab 10
#time python -u rank_statistics.py quijote Rockstar zheng07_velab zheng07_velab 20
#time python -u rank_statistics.py quijote Rockstar zheng07_velab zheng07_velab 5

#time python -u rank_statistics.py quijote FoF zheng07_velab zheng07 10 
#time python -u rank_statistics.py quijote Rockstar zheng07_velab zheng07 10 

#time python -u rank_statistics.py quijote FoF zheng07  zheng07_velab 10 
#time python -u rank_statistics.py quijote Rockstar  zheng07 zheng07_velab 10  
time python -u rank_statistics.py quijote Rockstar  zheng07 zheng07 10  

#time python -u rank_statistics.py fastpm FoF zheng07_velab zheng07_velab  10 
#time python -u rank_statistics.py fastpm FoF zheng07_velab zheng07  10  

#time python -u rank_statistics.py fastpm FoF zheng07 zheng07_velab  10 
#time python -u rank_statistics.py fastpm FoF zheng07 zheng07  10  

wait
