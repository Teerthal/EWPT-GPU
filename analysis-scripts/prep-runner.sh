#!/bin/bash

#SBATCH -c 1                        # number of tasks your job will spawn
#SBATCH --mem=40G                    # amount of RAM requested in GiB (2^40)
#SBATCH -p htc
#SBATCH -C epyc                     ##prevents sapphirerapids error, maybe
#SBATCH -t 0-4:00                  # wall time (D-HH:MM)
#SBATCH -o ./analysis_slurms_files/slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e ./analysis_slurms_files/slurm.%j.err             # STDERR (%j = JobId)
#SBATCH --export=NONE
# MV2_USE_ALIGNED_ALLOC=1
# module load mvapich2-2.3.7-gcc-11.2.0
# time mpirun -n 4 julia main_rk4_multiple.jl
# module unload mvapich2-2.3.7-gcc-11.2.0

# julia --project --check-bounds=no -O3 general-reader.jl ${1} ${2}
ml julia
ml hdf5/1.14.0-hpcx-2.13.1
export JULIA_CUDA_USE_COMPAT=false



N=1

# (
# for idx in {1..51}; do 
#    ((i=i%N)); ((i++==0)) && wait
#     julia --project --check-bounds=no -O3 general-reader.jl ${1} ${idx} & 
# done
# )


for i in {1..51}; do        ##slurm-array-task-id
    (
        # .. do your stuff here
        echo "starting task $i.."
        # sleep $(( (RANDOM % 3) + 1))
        julia --project --check-bounds=no -O3 general-reader.jl ${1} ${i} ${2}
    ) &

    # allow to execute up to $N jobs in parallel
    if [[ $(jobs -r -p | wc -l) -ge $N ]]; then
        # now there are $N jobs already running, so wait here for any job
        # to be finished so there is a place to start next one.
        wait -n
    fi

done

# no more jobs to be started but wait for pending jobs
# (all need to be finished)
wait

echo "all done"
