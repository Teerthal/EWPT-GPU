#!/bin/bash

#SBATCH -c 20
#SBATCH -t 0-04:00                  # wall time (D-HH:MM)
#SBATCH -q public -p htc
#SBATCH --mem=0
#SBATCH -A tpatel28            # Account to pull cpu hours from (commented out)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)

time julia -t 20 hdf5_to_vtk.jl