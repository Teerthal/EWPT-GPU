#!/bin/bash

# gamma=0.0
no_bubbles=$2
method="$1"

echo "No bubbles: " $no_bubbles

master_dir=/home/tpatel28/topo_mag/EW_sim
echo $master_dir

for gamma in 0.0 0.1 #0.0 0.1 0.5 #1.0 2.0 #0.00001
do

echo "Gamma: " $gamma

if [ $method == "igg-cn" ]; then
    echo "Method: " $method
    code_dir=igg-cn
    cd ${master_dir}/runs/many-bubbles-test/large-runs
    run_dir=${method}-gamma-${gamma}-bubbles-${no_bubbles}
    mkdir $run_dir
    cp ${master_dir}/${code_dir}/* ${master_dir}/runs/many-bubbles-test/large-runs/${run_dir}/
    cd ${master_dir}/runs/many-bubbles-test/large-runs/${run_dir}
    sbatch sbatchscript.sh ${gamma} ${no_bubbles}
fi

if [ $method == "igg-rk4" ]; then
    echo "Method: " $method
    code_dir=igg-rk4
    cd ${master_dir}/runs/many-bubbles-test/large-runs
    run_dir=${method}-gamma-${gamma}-bubbles-${no_bubbles}
    mkdir $run_dir
    cp ${master_dir}/${code_dir}/* ${master_dir}/runs/many-bubbles-test/large-runs/${run_dir}/
    cd ${master_dir}/runs/many-bubbles-test/large-runs/${run_dir}
    sbatch sbatchscript.sh ${gamma} ${no_bubbles}
fi

if [ $method == "ker-cn" ]; then
    echo "Method: " $method
    code_dir=kernels-4thOrder-CN
    cd ${master_dir}/runs/many-bubbles-test/large-runs
    run_dir=${method}-gamma-${gamma}-bubbles-${no_bubbles}
    mkdir $run_dir
    cp ${master_dir}/${code_dir}/* ${master_dir}/runs/many-bubbles-test/large-runs/${run_dir}/
    cd ${master_dir}/runs/many-bubbles-test/large-runs/${run_dir}
    sbatch run.sh ${gamma} ${no_bubbles}
fi

if [ $method == "ker-rk4" ]; then
    echo "Method: " $method
    code_dir=kernels-rk4
    cd ${master_dir}/runs/many-bubbles-test/large-runs
    run_dir=${method}-gamma-${gamma}-bubbles-${no_bubbles}
    mkdir $run_dir
    cp ${master_dir}/${code_dir}/* ${master_dir}/runs/many-bubbles-test/large-runs/${run_dir}/
    cd ${master_dir}/runs/many-bubbles-test/large-runs/${run_dir}
    sbatch run.sh ${gamma} ${no_bubbles}
fi

done