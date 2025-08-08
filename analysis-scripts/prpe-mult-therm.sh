#!/bin/bash

# gamma=0.0
# no_bubbles=10

# method="ker-rk4"
# gamma="$1"
# no_bubbles="$2"
# nte="$3"
# dx="$4"
# _dt="$5"
# latx="$6"
# gp2="$7"
# nsnaps="$8"
# T="$9"
# beta_W="${10}"
# beta_Y="${11}"

for seed in {1..25} #{2..25}
do
echo ${seed}
sbatch prep-runner-array.sh ${seed} ${1}      ###arg-1: para list entry
# julia --project --check-bounds=no -O3 general-reader.jl ${seed}
done

#slurm:16240812-test parallel bash on highmem for run 1
