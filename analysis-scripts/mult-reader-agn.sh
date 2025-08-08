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

for seed in {1..25}
do
sbatch run-reader.sh ${seed}
# julia --project --check-bounds=no -O3 general-reader.jl ${seed}
done