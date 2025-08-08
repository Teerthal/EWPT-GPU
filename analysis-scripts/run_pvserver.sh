#!/bin/bash

#SBATCH -c 1
#SBATCH --mem=10G
#SBATCH -p general -q private
#SBATCH -G 1
#SBATCH -t 0-4:00                  # wall time (D-HH:MM)
#SBATCH -o slurm.%j.out             # STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err             # STDERR (%j = JobId)
module load paraview-5.10.1-gcc-11.2.0
pvserver -p 33333
module unload paraview-5.10.1-gcc-11.2.0
