#!/bin/bash

#SBATCH -c 1
#SBATCH -N 1
#SBATCH -t 0-04:00                  # wall time (D-HH:MM)
#SBATCH -G a100:2
#SBATCH --mem=120G
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)

module load mvapich2-2.3.7-gcc-11.2.0
MV2_USE_ALIGNED_ALLOC=1
# export JULIA_CUDA_MEMORY_POOL=none
time mpirun -n 2 julia --project --check-bounds=no -O3 /scratch/tpatel28/topo_mag/EW_sim/main.jl 
# time julia --project --check-bounds=no -O3 /scratch/tpatel28/topo_mag/EW_sim/main.jl 
# time python vtkgenscript.py