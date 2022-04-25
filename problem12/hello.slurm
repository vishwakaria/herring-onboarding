#!/bin/bash
#SBATCH -N 2
# Run python script
srun python hello.py $SLURM_NODEID