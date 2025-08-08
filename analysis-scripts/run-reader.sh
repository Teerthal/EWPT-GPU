#!/bin/bash

#SBATCH -c 1                        # number of tasks your job will spawn
#SBATCH --mem=60G                    # amount of RAM requested in GiB (2^40)
#SBATCH -p general
#SBATCH -t 0-24:00                  # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)

# MV2_USE_ALIGNED_ALLOC=1
# module load mvapich2-2.3.7-gcc-11.2.0
# time mpirun -n 4 julia main_rk4_multiple.jl
# module unload mvapich2-2.3.7-gcc-11.2.0

julia --project --check-bounds=no -O3 general-reader.jl ${1}
