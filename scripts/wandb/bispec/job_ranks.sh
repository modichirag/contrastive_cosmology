#!/bin/bash
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH --partition=ccm
#SBATCH -C skylake
#SBATCH --time=8:00:00
#SBATCH --job-name=rankbk
#SBATCH -o ../logs/%x.o%j

# Start from an "empty" module collection.
module purge

# Load in what we need to execute mpirun.
module load modules/2.0-20220630
module load gcc/7.5.0 openmpi/1.10.7
source activate ptorch

ens=10


#time python -u rank_statistics.py quijote Rockstar zheng07_velab zheng07_velab $ens
#time python -u rank_statistics.py quijote FoF zheng07_velab zheng07_velab $ens

#time python -u rank_statistics.py quijote Rockstar zheng07 zheng07 $ens
time python -u rank_statistics.py quijote Rockstar zheng07 zheng07_velab $ens

#time python -u rank_statistics.py quijote FoF zheng07_velab zheng07  $ens
#time python -u rank_statistics.py quijote Rockstar zheng07_velab zheng07 $ens

#time python -u rank_statistics.py quijote FoF zheng07  zheng07_velab $ens
#time python -u rank_statistics.py quijote Rockstar  zheng07 zheng07_velab $ens

wait
