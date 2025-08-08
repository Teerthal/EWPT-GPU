#!/bin/bash

#SBATCH -N 1
#SBATCH --tasks-per-node=2
#SBATCH --mem=100G                    # amount of RAM requested in GiB (2^40)
#SBATCH -G 2
#SBATCH -t 0-04:00                  # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)

# MV2_USE_ALIGNED_ALLOC=1
# module load mvapich2-2.3.7-gcc-11.2.0
# time mpirun -n 4 julia main_rk4_multiple.jl
# module unload mvapich2-2.3.7-gcc-11.2.0

##Parameter Order##
##${gamma} ${no_bubbles} ${# time steps} ${dx} ${_dt} ${# Lattice points} ${gp2} ${nsnaps} ${T}##

module load mpich/4.1.2
time mpirun -np 2 julia main_rk4_multiple.jl 0.0 5 500 0.1 25.0 128 0.99 10 0.25
module unload mpich/4.1.2
