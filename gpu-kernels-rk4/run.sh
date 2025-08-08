#!/bin/bash

#SBATCH -c 1                        # number of tasks your job will spawn
#SBATCH --mem=100G                    # amount of RAM requested in GiB (2^40)
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80
#SBATCH -t 0-3:00                  # wall time (D-HH:MM)
#SBATCH -p general
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)

# MV2_USE_ALIGNED_ALLOC=1
# module load mvapich2-2.3.7-gcc-11.2.0
# time mpirun -n 4 julia main_rk4_multiple.jl
# module unload mvapich2-2.3.7-gcc-11.2.0

# time julia main_rk4.jl $1 $2 $3 $4 $5 $6 $7 $8 $9

time julia --project --check-bounds=no -O3 main_rk4.jl $1 $2 $3 $4 $5 $6 $7 $8 $9