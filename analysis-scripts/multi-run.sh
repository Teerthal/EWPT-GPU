#!/bin/bash

#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --mem=10G                    # amount of RAM requested in GiB (2^40)
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -t 00-00:10                  # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)

# MV2_USE_ALIGNED_ALLOC=1
# module load mvapich2-2.3.7-gcc-11.2.0
# time mpirun -n 4 julia main_rk4_multiple.jl
# module unload mvapich2-2.3.7-gcc-11.2.0

# module load mpich/4.1.2
# module load intel/parallel-studio-2020.4
# time mpirun -np 64 julia main_rk4_multiple.jl $1 $2 $3 $4 $5 $6 $7 $8 $9
# mpirun -n 216 -launcher slurm julia --project --check-bounds=no -O3 main_rk4_multiple.jl $1 $2 $3 $4 $5 $6 $7 $8 $9
# module unload intel/parallel-studio-2020.4
# module unload mpich/4.1.2

##fixed cuda precompilation issue?##
export JULIA_CUDA_USE_COMPAT=false

ml julia
ml hdf5/1.14.0-hpcx-2.13.1

srun --export=ALL --mpi=pmix julia load_cuda.jl