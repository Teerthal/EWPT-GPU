#!/bin/bash

#SBATCH -N 1
#SBATCH --tasks-per-node=2
#SBATCH --mem=100G                    # amount of RAM requested in GiB (2^40)
#SBATCH -G 2
#SBATCH -t 0-04:00                  # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)

module load intel/parallel-studio-2020.4
#time mpirun -np 4 julia main_rk4_multiple.jl 0.0 5 500 0.1 25.0 128 0.99 10 0.25
time mpiexec.hydra -np 4 ./numactl.gpu
