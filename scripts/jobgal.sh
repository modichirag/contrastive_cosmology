#!/bin/bash
#SBATCH -p ccm
#SBATCH -c 1
#SBATCH -t 150
#SBATCH -J paint

module --force purge
module load modules-traditional
module load cuda/11.0.3_450.51.06
module load cudnn/v8.0.4-cuda-11.0
module load slurm
module load gcc
module load openmpi
source activate defpyn

id0="$1"
id1="$2"
echo $id0 $id1

time srun -N 1 -n 1 python -u paint_gal.py --id0 $id0 --id1 $id1

##for i in {0..2000..50}; do j=$((i+50)); echo $i $j; sbatch flscript1.sh  $i  $j ; done
