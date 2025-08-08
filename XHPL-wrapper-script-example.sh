#!/bin/bash

host=$(hostname -s)
nproc=$(nproc --all) 
RANK_PER_NODE=4
CPU_CORES_PER_RANK=$((nproc/RANK_PER_NODE))
NUMA_RADIUS=$((CPU_CORES_PER_RANK/2))
echo "$host :: CPU_CORES_PER_RANK=$CPU_CORES_PER_RANK"

#export OMP_PROC_BIND=TRUE

export OMP_NUM_THREADS=$CPU_CORES_PER_RANK
export MKL_NUM_THREADS=$CPU_CORES_PER_RANK
export LD_LIBRARY_PATH=$HPL_DIR:$LD_LIBRARY_PATH

export MONITOR_GPU=0
export GPU_TEMP_WARNING=80
export GPU_CLOCK_WARNING=1189
export GPU_POWER_WARNING=500
export GPU_PCIE_GEN_WARNING=3
export GPU_PCIE_WIDTH_WARNING=16

export TRSM_CUTOFF=10000000
export GPU_DGEMM_SPLIT=1.0
APP="julia /home/tpatel28/topo_mag/EW_sim/kernels-rk4/mult-test/main_rk4_multiple.jl 0.0 5 500 0.1 25.0 128 0.99 10 0.25"

lrank=$OMPI_COMM_WORLD_LOCAL_RANK

export CUDA_VISIBLE_DEVICES=$lrank
cpubind=$(nvidia-smi topo -m | awk "/^GPU$lrank/ {range=\$(NF-1); split(range,a,\"-\"); printf(\"%d-%d\",a[1]-$NUMA_RADIUS,a[2])}")

numactl --physcpubind=$cpubind $APP

#case ${lrank} in
#[0])
#  export CUDA_VISIBLE_DEVICES=0
## numactl --physcpubind=0-7  \
#  numactl --physcpubind=12-23  \
#  $APP
#  ;;
#[1])
#  export CUDA_VISIBLE_DEVICES=1
## numactl --physcpubind=11-18 \
#  numactl --physcpubind=0-11  \
#  $APP
#  ;;
#[2])
#  export CUDA_VISIBLE_DEVICES=2
## numactl --physcpubind=24-31  \
#  numactl --physcpubind=36-47  \
#  $APP
#  ;;
#[3])
#  export CUDA_VISIBLE_DEVICES=3
## numactl --physcpubind=35-42 \
#  numactl --physcpubind=24-35 \
#  $APP
#  ;;
#
#esac
