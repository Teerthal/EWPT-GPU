#!/bin/bash

#SBATCH -c 1                        # number of tasks your job will spawn
#SBATCH --mem=40G                    # amount of RAM requested in GiB (2^40)
#SBATCH -p htc
#SBATCH -C epyc                     ##prevents sapphirerapids error, maybe
#SBATCH -t 0-4:00                  # wall time (D-HH:MM)
#SBATCH -o ./analysis_slurms_files/slurm.%A_%a.out             # STDOUT (%A_%a = JobId_TaskID)
#SBATCH -e ./analysis_slurms_files/slurm.%A_%a.err             # STDERR (%A_%a = JobId_TaskID)
#SBATCH --export=NONE
#SBATCH --array=1-51

# MV2_USE_ALIGNED_ALLOC=1
# module load mvapich2-2.3.7-gcc-11.2.0
# time mpirun -n 4 julia main_rk4_multiple.jl
# module unload mvapich2-2.3.7-gcc-11.2.0

# julia --project --check-bounds=no -O3 general-reader.jl ${1} ${2}
ml julia
ml hdf5/1.14.0-hpcx-2.13.1
export JULIA_CUDA_USE_COMPAT=false

i=${SLURM_ARRAY_TASK_ID}

echo "starting task ${i}.."
# sleep $(( (RANDOM % 3) + 1))
julia --project --check-bounds=no -O3 general-reader.jl ${1} ${i} ${2}
